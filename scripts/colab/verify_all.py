#!/usr/bin/env python3
"""One-shot GPU verification for fLLM, made for a Google Colab cell.

Run from a repository checkout (or copy the script anywhere):

    python3 scripts/colab/verify_all.py              # full run on a GPU runtime
    python3 scripts/colab/verify_all.py --cpu-only   # CPU checks only, no GPU needed
    python3 scripts/colab/verify_all.py --dry-run    # print the plan, touch nothing

The script acquires an fLLM binary (by default the CUDA asset of the latest
GitHub release, verified by its sha256; alternatives: --binary PATH and
--build-from-source), then runs, step by step and independent of each other:

  1. hardware detection (nvidia-smi, compute capability) and which checks apply
  2. the full `selftest-gpu` (ok/fail counts and the failed checks)
  3. greedy GPU-vs-CPU text equality for a small dense model, a small MoE model
     and a GPTQ import (one quantization family next to Q4_K_M)
  4. decode and prefill throughput at 355, 1863 and 8000 prompt tokens, with a
     warning (never a failure) of more than 10 percent below reference values
     from scripts/colab/reference.json - run noise between two runs is around
     15 percent, so this is a hint
  5. optional features, each skipped with a reason when the binary does not
     offer it (detected via --help, a string probe in the binary or a probe
     run, never via a version number): CUDA graphs on/off, FP16 activations
     (FLLM_ACT_DTYPE), prompt-lookup speculation, FP8 KV cache, vision model
     with an image, embedding model, multi-GPU self test

Every step has a time limit and can fail or be skipped without killing the
run.  The result is a PASS/FAIL/SKIP table on stdout plus verify_result.json
and verify_result.md in the work directory.  The network is used only for the
one-time download of the binary and of the test models (Hugging Face); all
checks run against a local server on 127.0.0.1.

Nothing here needs a pip install; it is Python 3 standard library only.
"""

import argparse
import atexit
import base64
import hashlib
import json
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zlib

# ---------------------------------------------------------------------------
# Constants (change these here, not in the middle of the script)
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
REFERENCE_FILE = os.path.join(SCRIPT_DIR, "reference.json")

RELEASE_REPO = "Marco-Lu2508/fLLM"
CUDA_ASSET = "fllm-linux-x86_64-cuda.tar.gz"
MUSL_ASSET = "fllm-linux-x86_64-musl.tar.gz"

PORT = 8973
CONTEXTS = (355, 1863, 8000)
SPEED_GEN_TOKENS = 64
EQUALITY_GEN_TOKENS = 32
EQUALITY_PROMPT = "The capital of France is"
MOE_PROMPT = "Water is made of hydrogen and"
REPETITIVE_PROMPT = "Repeat after me: alpha beta gamma delta epsilon zeta eta theta. " * 24
IMPORT_GEN_TOKENS = 24

# Small public test models.  Sizes are the compressed download sizes.
MODELS = {
    "dense": {
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "quant": "Q4_K_M",
    },
    "moe": {
        "repo": "allenai/OLMoE-1B-7B-0924-Instruct-GGUF",
        "file": "olmoe-1b-7b-0924-instruct-q4_k_m.gguf",
        "quant": "Q4_K_M",
    },
    "vision": {
        "repo": "ggml-org/SmolVLM-500M-Instruct-GGUF",
        "file": "SmolVLM-500M-Instruct-Q8_0.gguf",
        "mmproj": "mmproj-SmolVLM-500M-Instruct-f16.gguf",
    },
    "embed": {
        "repo": "ggml-org/bge-small-en-v1.5-Q8_0-GGUF",
        "file": "bge-small-en-v1.5-q8_0.gguf",
    },
}

# Quantized-checkpoint import (downloaded and converted by fLLM itself when
# used as the model argument).  Set FLLM_VERIFY_IMPORT_MODEL to test another.
IMPORT_MODEL = os.environ.get(
    "FLLM_VERIFY_IMPORT_MODEL", "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4"
)
# There is no small public ModelOpt NVFP4 checkpoint known to this script yet;
# point FLLM_VERIFY_NVFP4_MODEL at one to enable the NVFP4 import check.
NVFP4_MODEL = os.environ.get("FLLM_VERIFY_NVFP4_MODEL", "")

SELFTEST_TIMEOUT = 3600
MODEL_TIMEOUT = 2400          # one model load plus a few generations
DOWNLOAD_TIMEOUT = 3600
BUILD_TIMEOUT = 10800
SERVER_START_TIMEOUT = 900

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

T0 = time.time()
SERVERS = []
BINARY = None
MODE = "gpu"          # "gpu", "cpu-only" or "dry-run"
WORK = None
RUN_DIR = None
CACHE = {}            # facts shared between steps (hardware, backends, texts)
RESULTS = []          # one record per step


def log(msg):
    print("[%6.0fs] %s" % (time.time() - T0, msg), flush=True)


class StepSkipped(Exception):
    pass


class StepFailed(Exception):
    pass


class StepTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise StepTimeout()


def sh(cmd, timeout=300, env=None, cwd=None, check=False):
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        timeout=timeout,
        env=env,
        cwd=cwd,
    )
    if check and proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-5:])
        raise StepFailed("`%s` failed with rc=%d: %s" % (" ".join(cmd[:4]), proc.returncode, tail))
    return proc


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest, timeout=DOWNLOAD_TIMEOUT):
    """Stream url to dest (skip when present); returns dest."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        log("    already downloaded: %s" % os.path.basename(dest))
        return dest
    tmp = dest + ".part"
    log("    downloading %s" % url)
    req = urllib.request.Request(url, headers={"User-Agent": "fllm-verify-all"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as fh:
        total, seen, last = resp.headers.get("Content-Length"), 0, 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            seen += len(chunk)
            if seen - last > (256 << 20):
                last = seen
                if total:
                    log("    ... %d of %d MB" % (seen >> 20, int(total) >> 20))
                else:
                    log("    ... %d MB" % (seen >> 20))
    os.replace(tmp, dest)
    return dest


def bin_has_string(needle):
    """Chunked search for a byte string in the acquired binary."""
    needle = needle.encode()
    with open(BINARY, "rb") as fh:
        tail = b""
        while True:
            chunk = fh.read(8 << 20)
            if not chunk:
                return False
            if needle in tail + chunk:
                return True
            tail = chunk[-(len(needle) - 1):] if len(needle) > 1 else b""


def help_text():
    if "help" not in CACHE:
        proc = sh([BINARY, "--help"], timeout=120)
        CACHE["help"] = proc.stdout if proc.stdout else ""
    return CACHE["help"]


def has_flag(flag):
    return flag in help_text()


def supports_flag(flag):
    """True when the flag is documented in --help or present in the binary.

    The usage text does not list every option (for example --mmproj is
    accepted but not printed), so fall back to a string search.
    """
    if has_flag(flag):
        return True
    try:
        return bin_has_string(flag)
    except OSError:
        return False


def have_gpu():
    return MODE == "gpu" and CACHE.get("gpu_count", 0) > 0


def gpu_steps_reason():
    """Why GPU device steps cannot run here (or None)."""
    if MODE == "cpu-only":
        return "cpu-only mode (--cpu-only)"
    if CACHE.get("gpu_count", 0) <= 0:
        return "no NVIDIA GPU detected (run with --cpu-only for the CPU subset)"
    if not CACHE.get("cuda_compiled", False):
        raise StepFailed(
            "the binary has no cuda backend (compiled: %s); the linux assets before the"
            " CUDA packaging cannot load the NVIDIA driver either (static musl) - use the"
            " fllm-linux-x86_64-cuda asset, --binary or --build-from-source"
            % ", ".join(CACHE.get("backends", []))
        )
    return None


def require_gpu_step():
    reason = gpu_steps_reason()
    if reason:
        raise StepSkipped(reason)


# ---------------------------------------------------------------------------
# HTTP / server helpers
# ---------------------------------------------------------------------------

def http_get(path, timeout=5):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path),
                                 headers={"User-Agent": "fllm-verify-all"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def http_post(path, body, timeout=600):
    data = json.dumps(body).encode()
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path), data=data,
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "fllm-verify-all"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:200]
        except Exception:
            pass
        raise StepFailed("POST %s: HTTP %s%s" % (path, exc.code,
                                                 ": " + detail if detail else "")) from None


def start_server(model, device="cuda", extra_args=(), env_extra=None, tag="srv",
                 timeout=SERVER_START_TIMEOUT, cpu_threads=None):
    stop_servers()
    env = dict(os.environ)
    env["HF_HOME"] = os.path.join(WORK, "hf")
    if env_extra:
        env.update(env_extra)
    cmd = [BINARY, "serve", "--model", model, "--port", str(PORT),
           "--ctx", "8192", "--max-seqs", "1", "--device", device]
    if device == "cpu":
        threads = cpu_threads or max(2, (os.cpu_count() or 4) // 2)
        cmd += ["--threads", str(threads)]
    cmd += list(extra_args)
    logfile = os.path.join(RUN_DIR, "server-%s.log" % tag)
    fh = open(logfile, "wb")
    log("    starting server: %s" % " ".join(cmd[:7] + cmd[-2:]))
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)
    SERVERS.append(proc)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise StepFailed("server exited with rc=%s:\n%s" % (proc.returncode, log_tail(logfile)))
        try:
            status, _ = http_get("/health", timeout=2)
            if status == 200:
                log("    server ready (%s)" % tag)
                return proc
        except Exception:
            pass
        time.sleep(1)
    raise StepFailed("server did not become ready within %ds:\n%s" % (timeout, log_tail(logfile)))


def log_tail(path, lines=6):
    try:
        with open(path, "rb") as fh:
            data = fh.read().decode("utf-8", "replace")
        return "\n".join(data.splitlines()[-lines:])
    except OSError:
        return "(no log)"


def server_stderr_scan(logfile, pattern):
    """Search the merged server log for a pattern (for feature state lines)."""
    try:
        with open(logfile, "rb") as fh:
            data = fh.read().decode("utf-8", "replace")
    except OSError:
        return []
    return [line for line in data.splitlines() if re.search(pattern, line)]


def stop_server(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def stop_servers():
    while SERVERS:
        stop_server(SERVERS.pop())


atexit.register(stop_servers)


def complete(prompt, max_tokens, ignore_eos=True, temperature=0, timeout=600):
    body = {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature,
            "ignore_eos": ignore_eos, "stream": False}
    data = http_post("/v1/completions", body, timeout=timeout)
    choices = data.get("choices") or []
    if not choices:
        raise StepFailed("completion returned no choices: %s" % json.dumps(data)[:200])
    return choices[0].get("text", "")


def stream_prompt(prompt, max_tokens, timeout=900):
    """Stream one completion; returns (prompt_seconds, decode_tps, events)."""
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
                       "ignore_eos": True, "stream": True}).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/v1/completions" % PORT, data=body,
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "fllm-verify-all"})
    started = time.time()
    first = None
    events = 0
    last = started
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload in ("", "[DONE]"):
                if payload == "[DONE]":
                    break
                continue
            if first is None:
                first = time.time()
            events += 1
            last = time.time()
    if first is None or events == 0:
        raise StepFailed("stream produced no tokens")
    decode_tps = (events - 1) / max(last - first, 1e-6) if events > 1 else 0.0
    return first - started, decode_tps, events


def token_count(text):
    data = http_post("/tokenize", {"content": text, "add_special": False}, timeout=60)
    tokens = data.get("tokens")
    if isinstance(tokens, list):
        return len(tokens)
    if isinstance(tokens, int):
        return tokens
    if isinstance(data.get("count"), int):
        return data["count"]
    raise StepFailed("cannot read token count from /tokenize: %s" % json.dumps(data)[:200])


def make_prompt(n_tokens):
    """Build a prompt of about n_tokens tokens using /tokenize for sizing."""
    base = "The quick brown fox jumps over the lazy dog near the river bank. "
    per = max(1, token_count(base))
    text = base * max(1, n_tokens // per)
    count = token_count(text)
    filler = "Additionally, here is one more simple sentence to lengthen the prompt. "
    for _ in range(24):
        if abs(count - n_tokens) <= max(2, n_tokens // 64):
            break
        diff = n_tokens - count
        if diff <= 0:
            break
        words = filler.split()
        chunk = " ".join(words[:max(1, min(len(words), diff * 2))])
        text = text + chunk + " "
        new_count = token_count(text)
        if new_count == count:
            break
        count = new_count
    return text, count


def make_png64(width=32, height=32):
    """A small RGB PNG as base64 (checked pattern), pure stdlib."""
    rows = b""
    for y in range(height):
        row = bytearray([0])  # filter type 0
        for x in range(width):
            row += bytes(((x * 8) % 256, (y * 8) % 256, 200 if (x // 8 + y // 8) % 2 else 40))
        rows += bytes(row)

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(rows, 9))
    png += chunk(b"IEND", b"")
    return base64.b64encode(png).decode()


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return "first difference at char %d: ...%r vs ...%r" % (
                i, a[max(0, i - 30):i + 30], b[max(0, i - 30):i + 30])
    return "lengths %d vs %d" % (len(a), len(b))


def model_path(key):
    """Download (once) and return the GGUF path of a MODELS entry."""
    cache_key = "model_" + key
    if cache_key not in CACHE:
        entry = MODELS[key]
        CACHE[cache_key] = download(
            "https://huggingface.co/%s/resolve/main/%s" % (entry["repo"], entry["file"]),
            os.path.join(WORK, "models", entry["file"]))
    return CACHE[cache_key]


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_hw():
    gpus = []
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            proc = sh([smi, "--query-gpu=name,driver_version,memory.total,compute_cap",
                       "--format=csv,noheader"], timeout=60)
            if proc.returncode == 0:
                for line in proc.stdout.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        gpus.append({"name": parts[0], "driver": parts[1],
                                     "memory": parts[2],
                                     "compute_cap": parts[3] if len(parts) > 3 else ""})
        except Exception as exc:
            log("    nvidia-smi query failed: %s" % exc)
    cc = None
    if gpus:
        m = re.match(r"(\d+)\.(\d+)", gpus[0].get("compute_cap", ""))
        if m:
            cc = int(m.group(1)) * 10 + int(m.group(2))
    CACHE["gpu_count"] = len(gpus)
    CACHE["gpus"] = gpus
    CACHE["cc"] = cc
    gate = {}
    if cc is not None:
        gate = {"cuda_graphs": True, "fp8_kv": cc >= 89, "fa3_hopper": cc >= 90,
                "nvfp4": cc >= 120}
    elif gpus:
        gate = {"cuda_graphs": True, "fp8_kv": None, "fa3_hopper": None, "nvfp4": None}
    CACHE["gates"] = gate
    detail = "%d GPU(s)" % len(gpus)
    if gpus:
        detail += ": " + "; ".join("%s (driver %s, %s, CC %s)" % (
            g["name"], g["driver"], g["memory"], g.get("compute_cap") or "?") for g in gpus)
        if cc is None:
            detail += " - compute capability readable only from `devices` output"
    else:
        detail += " (nvidia-smi not found or no NVIDIA GPU)"
    notes = []
    if gate:
        notes.append("gated checks: FP8 KV %s, Hopper/FA3 paths %s, NVFP4 %s" % (
            "on" if gate["fp8_kv"] else "off (needs sm_89+)",
            "on" if gate["fa3_hopper"] else "off (needs sm_90)",
            "on" if gate["nvfp4"] else "off (needs sm_120)"))
    if MODE == "cpu-only":
        notes.append("cpu-only mode: GPU device steps will be skipped")
    return {"detail": detail, "data": {"gpus": gpus, "cc": cc, "gates": gate},
            "warnings": notes}


def step_binary():
    assert BINARY is not None
    proc = sh([BINARY, "--help"], timeout=120)
    if proc.returncode != 0:
        raise StepFailed("`--help` exited with rc=%d" % proc.returncode)
    digest = sha256_of(BINARY)
    size = os.path.getsize(BINARY)
    return {"detail": "%s (%.0f MB, sha256 %s...)" % (
        CACHE.get("binary_source", "?"), size / 1048576.0, digest[:16]),
        "data": {"path": BINARY, "sha256": digest, "size_bytes": size,
                 "source": CACHE.get("binary_source", "?")}}


def step_devices():
    json_data = None
    proc = sh([BINARY, "devices", "--json"], timeout=300)
    if proc.returncode == 0 and proc.stdout.lstrip().startswith("{"):
        try:
            json_data = json.loads(proc.stdout)
        except Exception:
            json_data = None
    if json_data is not None:
        backends = [str(b) for b in json_data.get("compiled_backends", [])]
        text = proc.stdout
    else:
        proc = sh([BINARY, "devices"], timeout=300)
        if proc.returncode != 0:
            raise StepFailed("`devices` exited with rc=%d:\n%s" % (
                proc.returncode, "\n".join(proc.stdout.splitlines()[-5:])))
        text = proc.stdout
        m = re.search(r"Compiled backends:\s*(.+)", text)
        backends = [b.strip() for b in m.group(1).split(",")] if m else []
    CACHE["backends"] = backends
    CACHE["cuda_compiled"] = "cuda" in backends
    if CACHE.get("cc") is None:
        m2 = re.search(r"\bsm_(\d+)\b", text)
        if m2:
            CACHE["cc"] = int(m2.group(1))
            if CACHE.get("gates"):
                cc = CACHE["cc"]
                CACHE["gates"] = {"cuda_graphs": True, "fp8_kv": cc >= 89,
                                  "fa3_hopper": cc >= 90, "nvfp4": cc >= 120}
    detail = "compiled backends: %s" % (", ".join(backends) or "?")
    warnings = []
    if have_gpu() and not CACHE["cuda_compiled"]:
        warnings.append("cuda is NOT compiled into this binary while a GPU is present")
    return {"detail": detail, "data": {"backends": backends}, "warnings": warnings}


def step_selftest_gpu():
    require_gpu_step()
    out_json = os.path.join(RUN_DIR, "selftest-gpu.json")
    proc = sh([BINARY, "selftest-gpu", "--device", "cuda", "--json", out_json],
              timeout=SELFTEST_TIMEOUT)
    text = proc.stdout or ""
    data = {"json": out_json}
    parsed = None
    if os.path.exists(out_json):
        try:
            with open(out_json) as fh:
                parsed = json.load(fh)
        except Exception as exc:
            data["json_error"] = str(exc)
    if isinstance(parsed, dict) and "checks" in parsed:
        checks = int(parsed.get("checks", 0))
        failed = int(parsed.get("failed", -1))
        skipped = int(parsed.get("skipped", 0))
        ok = int(parsed.get("ok_count", 0))
        data.update({"checks": checks, "failed": failed, "skipped": skipped,
                     "ok_count": ok, "backend": parsed.get("backend"),
                     "device": parsed.get("device_name"), "error": parsed.get("error"),
                     "groups": parsed.get("groups")})
        fails = []
        for row in parsed.get("results", []):
            if row.get("status") != "ok" or row.get("pass") == "failed":
                fails.append("%s/%s %s: %s" % (row.get("group"), row.get("op"),
                                               row.get("shape"), row.get("message")))
        data["failed_lines"] = fails[:10]
    else:
        m = None
        for m in re.finditer(r"(\d+) checks, (\d+) failed, (\d+) skipped \(unsupported\), (\d+) ok", text):
            pass
        if proc.returncode != 0 and not m:
            raise StepFailed("selftest-gpu exited with rc=%d without a summary:\n%s" % (
                proc.returncode, "\n".join(text.splitlines()[-8:])))
        checks = int(m.group(1)) if m else 0
        failed = int(m.group(2)) if m else -1
        skipped = int(m.group(3)) if m else 0
        ok = int(m.group(4)) if m else 0
        fails = [line.strip() for line in text.splitlines()
                 if re.search(r"\bfail", line, re.I) and "0 failed" not in line.lower()]
        if m:
            fails = [l for l in fails if l != m.group(0).strip()]
        data.update({"checks": checks, "failed": failed, "skipped": skipped,
                     "ok_count": ok, "failed_lines": fails[:10]})
    status_ok = proc.returncode == 0 and failed == 0 and checks > 0
    detail = "%d checks, %d failed, %d skipped (unsupported), %d ok" % (
        checks, failed, skipped, ok)
    if data.get("failed_lines"):
        detail += "; e.g. " + " | ".join(data["failed_lines"][:3])
    if not status_ok:
        raise StepFailed(detail)
    return {"detail": detail, "data": data}


def equality_pair(model, device_a, device_b, prompt, gen_tokens, tag_a="gpu", tag_b="cpu",
                  extra_a=(), extra_b=(), env_a=None, env_b=None, timeout=MODEL_TIMEOUT):
    """Run the same greedy request on two devices; returns (text_a, text_b, logfile_a)."""
    start_server(model, device=device_a, extra_args=extra_a, env_extra=env_a,
                 tag=tag_a, timeout=timeout)
    text_a = complete(prompt, gen_tokens, timeout=timeout)
    logfile_a = os.path.join(RUN_DIR, "server-%s.log" % tag_a)
    start_server(model, device=device_b, extra_args=extra_b, env_extra=env_b,
                 tag=tag_b, timeout=timeout)
    text_b = complete(prompt, gen_tokens, timeout=timeout)
    stop_servers()
    return text_a, text_b, logfile_a


def step_dense():
    require_gpu_step()
    path = model_path("dense")
    gpu_text, cpu_text, _ = equality_pair(path, "cuda", "cpu", EQUALITY_PROMPT,
                                          EQUALITY_GEN_TOKENS)
    CACHE["dense_cpu_text"] = cpu_text
    CACHE["dense_gpu_text"] = gpu_text
    if gpu_text != cpu_text:
        raise StepFailed("GPU and CPU greedily diverge: " + first_diff(cpu_text, gpu_text))
    return {"detail": "identical greedy text on cuda and cpu (%d tokens): %r" % (
        EQUALITY_GEN_TOKENS, gpu_text[:60]),
        "data": {"text": gpu_text, "text_cpu": cpu_text}}


def step_moe():
    require_gpu_step()
    path = model_path("moe")
    gpu_text, cpu_text, _ = equality_pair(path, "cuda", "cpu", MOE_PROMPT,
                                          EQUALITY_GEN_TOKENS, tag_a="moe-gpu", tag_b="moe-cpu")
    if gpu_text != cpu_text:
        raise StepFailed("GPU and CPU greedily diverge: " + first_diff(cpu_text, gpu_text))
    return {"detail": "identical greedy text on cuda and cpu (%d tokens): %r" % (
        EQUALITY_GEN_TOKENS, gpu_text[:60]), "data": {"text": gpu_text}}


def _import_text(device, tag, timeout=MODEL_TIMEOUT):
    start_server(IMPORT_MODEL, device=device, tag=tag, timeout=timeout)
    text = complete(EQUALITY_PROMPT, IMPORT_GEN_TOKENS, timeout=timeout)
    return text


def step_import():
    require_gpu_step()
    cpu_text = None
    converted = None
    try:
        gpu_text = _import_text("cuda", "import-gpu")
        converted = server_stderr_scan(os.path.join(RUN_DIR, "server-import-gpu.log"),
                                       r"convert|import|quant")
        cpu_text = _import_text("cpu", "import-cpu")
    except StepFailed as exc:
        msg = str(exc)
        if re.search(r"unsupported|not supported|unknown|unrecognized|cannot convert",
                     msg, re.I):
            raise StepSkipped("the binary cannot import this checkpoint: %s" % msg.splitlines()[0])
        raise
    stop_servers()
    if gpu_text != cpu_text:
        raise StepFailed("GPU and CPU greedily diverge on the imported checkpoint: "
                         + first_diff(cpu_text, gpu_text))
    detail = "imported %s, identical greedy text on cuda and cpu (%d tokens)" % (
        _short_model_name(IMPORT_MODEL), IMPORT_GEN_TOKENS)
    return {"detail": detail, "data": {"text": gpu_text, "converted_lines": converted[:5]}}


def _short_model_name(model):
    return model.split("/")[-1] if "/" in model else os.path.basename(model)


def step_nvfp4():
    require_gpu_step()
    if not NVFP4_MODEL:
        raise StepSkipped("no small public NVFP4 test checkpoint configured; set"
                          " FLLM_VERIFY_NVFP4_MODEL=<hf repo> to enable this check")
    cc = CACHE.get("cc")
    if cc is not None and cc < 120:
        raise StepSkipped("NVFP4 tensor cores need sm_120 (this GPU is sm_%d); the Marlin"
                          " fallback is covered by the GPTQ import check" % cc)
    try:
        gpu_text = _import_text("cuda", "nvfp4-gpu")
    except StepFailed as exc:
        msg = str(exc)
        if re.search(r"unsupported|not supported|unknown|unrecognized|cannot convert",
                     msg, re.I):
            raise StepSkipped("the binary cannot import this checkpoint: %s" % msg.splitlines()[0])
        raise
    stop_servers()
    return {"detail": "imported %s and generated %d tokens" % (_short_model_name(NVFP4_MODEL),
                                                               IMPORT_GEN_TOKENS),
            "data": {"text": gpu_text}}


def step_speed():
    require_gpu_step()
    path = model_path("dense")
    start_server(path, device="cuda", tag="speed", timeout=MODEL_TIMEOUT)
    measurements = {}
    graph_lines = []
    for ctx in CONTEXTS:
        prompt, tokens = make_prompt(ctx)
        log("    context %d: prompt has %d tokens" % (ctx, tokens))
        prefill_s, decode_tps, events = stream_prompt(prompt, SPEED_GEN_TOKENS)
        prefill_tps = tokens / max(prefill_s, 1e-6)
        measurements[str(ctx)] = {
            "prompt_tokens": tokens, "prefill_seconds": round(prefill_s, 3),
            "prefill_tps": round(prefill_tps, 1), "decode_tps": round(decode_tps, 1),
            "generated_tokens": events,
        }
        log("    context %d: prefill %.0f tok/s, decode %.1f tok/s" % (ctx, prefill_tps, decode_tps))
    graph_lines = server_stderr_scan(os.path.join(RUN_DIR, "server-speed.log"),
                                     r"CUDA graphs for decode steps")
    stop_servers()
    warnings = []
    ref = reference_for_card()
    model_file = os.path.basename(path)
    ref_model = (ref or {}).get("models", {}).get(model_file)
    data = {"measurements": measurements, "graph_state_lines": graph_lines,
            "reference_card": (ref or {}).get("match"), "reference_model": ref_model}
    if ref_model:
        graphs_on = any("graphs for decode steps: on" in l for l in graph_lines)
        ref_value = ref_model.get("decode_tok_s_cuda_graph_on" if graphs_on
                                  else "decode_tok_s_short_ctx")
        ref_kind = "CUDA graphs on" if graphs_on else "graphs off"
        measured = measurements[str(CONTEXTS[0])]["decode_tps"]
        if ref_value:
            loss = 1.0 - measured / float(ref_value)
            if loss > 0.10:
                warnings.append(
                    "decode at context %d is %.1f tok/s, %.0f%% below the documented %s"
                    " reference (%.0f tok/s) for %s - run noise is around 15%%, so treat this"
                    " as a hint" % (CONTEXTS[0], measured, loss * 100, ref_kind, ref_value,
                                    model_file))
            data["reference_used"] = {"kind": ref_kind, "decode_tok_s": ref_value}
    else:
        log("    no reference values for this card/model; recording measurements only")
    return {"detail": "decode %.1f/%.1f/%.1f tok/s, prefill %.0f/%.0f/%.0f tok/s at 355/1863/8000"
                      % (measurements[str(CONTEXTS[0])]["decode_tps"],
                         measurements[str(CONTEXTS[1])]["decode_tps"],
                         measurements[str(CONTEXTS[2])]["decode_tps"],
                         measurements[str(CONTEXTS[0])]["prefill_tps"],
                         measurements[str(CONTEXTS[1])]["prefill_tps"],
                         measurements[str(CONTEXTS[2])]["prefill_tps"]),
            "data": data, "warnings": warnings}


def reference_for_card():
    try:
        with open(REFERENCE_FILE) as fh:
            ref = json.load(fh)
    except Exception as exc:
        log("    reference.json not readable: %s" % exc)
        return None
    names = " ".join(g["name"] for g in CACHE.get("gpus", [])).lower()
    for card in ref.get("cards", []):
        for match in card.get("match", []):
            if match.lower() in names:
                return card
    return None


def step_cudagraph():
    require_gpu_step()
    path = model_path("dense")
    state_lines = {}
    texts = {}
    for value, tag in (("0", "graph-off"), ("1", "graph-on")):
        start_server(path, device="cuda", env_extra={"FLLM_CUDA_GRAPH": value},
                     tag=tag, timeout=MODEL_TIMEOUT)
        texts[value] = complete(EQUALITY_PROMPT, 16, timeout=MODEL_TIMEOUT)
        state_lines[value] = server_stderr_scan(os.path.join(RUN_DIR, "server-%s.log" % tag),
                                                r"CUDA graph")
        stop_servers()
    if not state_lines["0"] and not state_lines["1"]:
        raise StepSkipped("the binary never reports CUDA graph state (feature not in this"
                          " build); no FLLM_CUDA_GRAPH switch")
    warnings = []
    if texts["0"] != texts["1"]:
        raise StepFailed("text with CUDA graphs off and on differs: "
                         + first_diff(texts["0"], texts["1"]))
    for value, tag in (("0", "off"), ("1", "on")):
        if not state_lines[value]:
            warnings.append("no 'CUDA graphs' line seen with FLLM_CUDA_GRAPH=%s" % value)
    if texts.get("0") and CACHE.get("dense_cpu_text") and texts["0"] != CACHE["dense_cpu_text"]:
        warnings.append("graph-off text differs from the CPU text recorded in the dense"
                        " step (prompt/length differ, so this is informational)")
    first_line = (state_lines["1"] or state_lines["0"])[:1]
    return {"detail": "texts identical with graphs off/on; %s" % (
        first_line[0].strip() if first_line else "state line missing"),
        "data": {"state_lines": state_lines, "text": texts["0"]}, "warnings": warnings}


def step_spec():
    require_gpu_step()
    if not supports_flag("--speculative-lookup"):
        raise StepSkipped("this binary has no --speculative-lookup option")
    path = model_path("dense")
    start_server(path, device="cuda", tag="spec-base", timeout=MODEL_TIMEOUT)
    base = complete(REPETITIVE_PROMPT, 48, timeout=MODEL_TIMEOUT)
    stop_servers()
    start_server(path, device="cuda", extra_args=("--speculative-lookup", "4"),
                 tag="spec-on", timeout=MODEL_TIMEOUT)
    spec = complete(REPETITIVE_PROMPT, 48, timeout=MODEL_TIMEOUT)
    stop_servers()
    if base != spec:
        raise StepFailed("prompt-lookup speculation changed the greedy text: "
                         + first_diff(base, spec))
    return {"detail": "greedy text identical with and without --speculative-lookup 4",
            "data": {"text": spec}}


def step_fp8kv():
    require_gpu_step()
    if not supports_flag("--kv-cache-type"):
        raise StepSkipped("this binary has no --kv-cache-type option")
    cc = CACHE.get("cc")
    if cc is not None and cc < 89:
        raise StepSkipped("FP8 KV cache needs sm_89+ (this GPU is sm_%d)" % cc)
    path = model_path("dense")
    start_server(path, device="cuda", extra_args=("--kv-cache-type", "fp8_e4m3"),
                 tag="fp8kv", timeout=MODEL_TIMEOUT)
    text = complete(EQUALITY_PROMPT, 16, timeout=MODEL_TIMEOUT)
    stop_servers()
    warnings = []
    if CACHE.get("dense_gpu_text") and not CACHE["dense_gpu_text"].startswith(text[:8]):
        warnings.append("fp8 KV text start differs from the fp16 KV text; FP8 KV is an"
                        " approximation, the first tokens usually stay equal")
    return {"detail": "generated with --kv-cache-type fp8_e4m3: %r" % text[:40],
            "data": {"text": text}, "warnings": warnings}


def step_actdtype():
    require_gpu_step()
    if not bin_has_string("FLLM_ACT_DTYPE"):
        raise StepSkipped("this binary has no FLLM_ACT_DTYPE support (FP16 activations are"
                          " not part of this build)")
    path = model_path("dense")
    start_server(path, device="cuda", env_extra={"FLLM_ACT_DTYPE": "f16"},
                 tag="actdtype", timeout=MODEL_TIMEOUT)
    text = complete(EQUALITY_PROMPT, 16, timeout=MODEL_TIMEOUT)
    stop_servers()
    warnings = []
    if CACHE.get("dense_cpu_text") and not CACHE["dense_cpu_text"].startswith(text[:8]):
        warnings.append("text with FLLM_ACT_DTYPE=f16 starts differently from the CPU"
                        " baseline - expected for reduced-precision activations, check by hand")
    return {"detail": "generated with FLLM_ACT_DTYPE=f16: %r" % text[:40],
            "data": {"text": text}, "warnings": warnings}


def step_vision():
    reason = gpu_steps_reason()
    cpu_fallback = reason is not None and MODE == "cpu-only"
    if reason and not cpu_fallback:
        raise StepSkipped(reason)
    if not supports_flag("--mmproj"):
        raise StepSkipped("this binary has no --mmproj option")
    model = download("https://huggingface.co/%s/resolve/main/%s"
                     % (MODELS["vision"]["repo"], MODELS["vision"]["file"]),
                     os.path.join(WORK, "models", MODELS["vision"]["file"]))
    mmproj = download("https://huggingface.co/%s/resolve/main/%s"
                      % (MODELS["vision"]["repo"], MODELS["vision"]["mmproj"]),
                      os.path.join(WORK, "models", MODELS["vision"]["mmproj"]))
    device = "cpu" if cpu_fallback else "cuda"
    start_server(model, device=device, extra_args=("--mmproj", mmproj),
                 tag="vision", timeout=MODEL_TIMEOUT)
    image = "data:image/png;base64," + make_png64()
    diagnostics = http_post("/v1/images/embeddings", {"image_url": image}, timeout=600)
    text_ok = False
    chat_note = ""
    try:
        chat = http_post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "What colors do you see? Answer in one short sentence."},
                {"type": "image_url", "image_url": {"url": image}}]}],
            "max_tokens": 32, "temperature": 0}, timeout=600)
        choice = (chat.get("choices") or [{}])[0]
        msg = choice.get("message", {}).get("content", "")
        text_ok = bool(msg)
        chat_note = "chat answer: %r" % (msg[:60] or "")
    except Exception as exc:
        chat_note = "chat with image failed (%s); the diagnostic image endpoint is the" \
                    " deterministic check" % exc
    stop_servers()
    if not diagnostics:
        raise StepFailed("the image diagnostic endpoint returned nothing")
    if not text_ok:
        log("    " + chat_note)
    return {"detail": "image embeddings endpoint answered (%s); %s"
                      % ("models" if "data" in diagnostics else json.dumps(diagnostics)[:40],
                         chat_note),
            "data": {"chat_ok": text_ok, "diagnostics_keys": sorted(diagnostics)[:8]},
            "warnings": [] if text_ok else [chat_note]}


def step_embed():
    reason = gpu_steps_reason()
    cpu_fallback = reason is not None and MODE == "cpu-only"
    if reason and not cpu_fallback:
        raise StepSkipped(reason)
    if not supports_flag("--embeddings"):
        raise StepSkipped("this binary has no --embeddings option")
    model = download("https://huggingface.co/%s/resolve/main/%s"
                     % (MODELS["embed"]["repo"], MODELS["embed"]["file"]),
                     os.path.join(WORK, "models", MODELS["embed"]["file"]))
    device = "cpu" if cpu_fallback else "cuda"
    start_server(model, device=device, extra_args=("--embeddings",),
                 tag="embed", timeout=MODEL_TIMEOUT)
    data = http_post("/v1/embeddings",
                     {"input": ["hello world", "the cat sat on the mat"]},
                     timeout=600)
    stop_servers()
    rows = data.get("data") or []
    if len(rows) != 2 or not rows[0].get("embedding"):
        raise StepFailed("unexpected embeddings response: %s" % json.dumps(data)[:200])
    vec_a, vec_b = rows[0]["embedding"], rows[1]["embedding"]
    warnings = []
    if len(vec_a) != 384:
        warnings.append("vector length %d, bge-small is usually 384" % len(vec_a))
    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    norm = (sum(x * x for x in vec_a) ** 0.5) * (sum(y * y for y in vec_b) ** 0.5)
    cos = dot / norm if norm else 0.0
    if cos < 0.4 or cos > 0.99:
        warnings.append("cosine between the two sentences is %.3f, expected something in"
                        " between" % cos)
    return {"detail": "two embeddings, length %d, cosine %.3f" % (len(vec_a), cos),
            "data": {"dim": len(vec_a), "cosine": round(cos, 4)}, "warnings": warnings}


def step_multigpu():
    if MODE == "cpu-only":
        raise StepSkipped("cpu-only mode")
    count = CACHE.get("gpu_count", 0)
    if count < 2:
        raise StepSkipped("single GPU (Colab has one); multi-GPU self test needs 2+ devices")
    if "selftest-comm" in help_text():
        proc = sh([BINARY, "selftest-comm"], timeout=900)
        text = proc.stdout or ""
        if "SKIP nccl" in text:
            raise StepSkipped("selftest-comm reports: " +
                              "; ".join(l for l in text.splitlines() if "SKIP nccl" in l))
        if proc.returncode != 0:
            raise StepFailed("selftest-comm exited with rc=%d:\n%s"
                             % (proc.returncode, "\n".join(text.splitlines()[-6:])))
        return {"detail": "selftest-comm passed on %d GPUs" % count, "data": {}}
    if "selftest-parallel" in help_text():
        proc = sh([BINARY, "selftest-parallel", "--all"], timeout=1800)
        if proc.returncode != 0:
            raise StepFailed("selftest-parallel --all exited with rc=%d" % proc.returncode)
        return {"detail": "selftest-parallel --all passed", "data": {}}
    raise StepSkipped("this binary has no multi-GPU self test (selftest-comm/selftest-parallel)")


def step_selftest_cpu():
    if MODE != "cpu-only":
        raise StepSkipped("GPU mode: CPU tiers are not the target here; use --cpu-only")
    proc = sh([BINARY, "selftest-cpu", "--tiers", "all"], timeout=SELFTEST_TIMEOUT)
    text = proc.stdout or ""
    tiers = re.findall(r"^tier (\S+)\s+(\d+) checks, (\d+) failed", text, re.M)
    failed = sum(int(t[2]) for t in tiers)
    detail = "; ".join("%s: %s checks, %s failed" % t for t in tiers) or \
        "\n".join(text.splitlines()[-5:])
    data = {"tiers": [{"tier": t[0], "checks": int(t[1]), "failed": int(t[2])}
                      for t in tiers]}
    if proc.returncode != 0 or failed > 0 or not tiers:
        raise StepFailed(detail)
    return {"detail": detail, "data": data}


STEPS = [
    ("hw", "Hardware", step_hw, 120),
    ("binary", "Binary", step_binary, 300),
    ("devices", "Devices", step_devices, 300),
    ("selftest", "selftest-gpu", step_selftest_gpu, SELFTEST_TIMEOUT + 60),
    ("selftestcpu", "selftest-cpu", step_selftest_cpu, SELFTEST_TIMEOUT + 60),
    ("dense", "Dense Q4_K_M", step_dense, 2 * MODEL_TIMEOUT),
    ("moe", "MoE Q4_K_M", step_moe, 2 * MODEL_TIMEOUT),
    ("import", "GPTQ import", step_import, 2 * MODEL_TIMEOUT),
    ("nvfp4", "NVFP4 import", step_nvfp4, MODEL_TIMEOUT),
    ("speed", "Speed vs reference", step_speed, MODEL_TIMEOUT + 900),
    ("cudagraph", "CUDA graphs", step_cudagraph, 2 * MODEL_TIMEOUT),
    ("spec", "Spec decoding", step_spec, 2 * MODEL_TIMEOUT),
    ("fp8kv", "FP8 KV cache", step_fp8kv, MODEL_TIMEOUT),
    ("actdtype", "FP16 activations", step_actdtype, MODEL_TIMEOUT),
    ("vision", "Vision model", step_vision, MODEL_TIMEOUT + 600),
    ("embed", "Embeddings", step_embed, MODEL_TIMEOUT),
    ("multigpu", "Multi-GPU", step_multigpu, SELFTEST_TIMEOUT),
]


def run_step(key, title, fn, timeout):
    log("=== %s ===" % title)
    started = time.time()
    record = {"key": key, "name": title, "status": "FAIL", "detail": "", "seconds": 0.0,
              "warnings": [], "data": {}}
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(timeout))
    try:
        result = fn()
        record.update(status="PASS", detail=result.get("detail", ""),
                      data=result.get("data", {}), warnings=result.get("warnings", []))
    except StepSkipped as exc:
        record.update(status="SKIP", detail=str(exc))
    except StepTimeout:
        record.update(status="FAIL", detail="timed out after %ds" % timeout)
    except StepFailed as exc:
        record.update(status="FAIL", detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - one broken step must not kill the run
        record.update(status="FAIL", detail="%s: %s" % (type(exc).__name__, exc))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        stop_servers()
    record["seconds"] = round(time.time() - started, 1)
    RESULTS.append(record)
    log("--- %s: %s (%s) %s" % (title, record["status"], record["seconds"], record["detail"][:120]))
    return record


# ---------------------------------------------------------------------------
# Binary acquisition
# ---------------------------------------------------------------------------

def github_release(tag):
    url = "https://api.github.com/repos/%s/releases/%s" % (RELEASE_REPO, tag)
    req = urllib.request.Request(url, headers={"User-Agent": "fllm-verify-all",
                                               "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise StepFailed("cannot read release '%s' from GitHub (HTTP %d)" % (tag, exc.code))


def acquire_from_release(tag, asset_name):
    release = github_release(tag)
    assets = {a["name"]: a for a in release.get("assets", [])}
    if asset_name not in assets:
        raise StepFailed(
            "release %s has no %s (assets: %s); releases made before the CUDA packaging"
            " never have it - pass --binary or --build-from-source"
            % (release.get("tag_name", tag), asset_name, ", ".join(sorted(assets)) or "none"))
    asset = assets[asset_name]
    dest = os.path.join(WORK, asset_name)
    download(asset["browser_download_url"], dest)
    digest = sha256_of(dest)
    sha_name = asset_name + ".sha256"
    if sha_name in assets:
        sha_file = download(assets[sha_name]["browser_download_url"], dest + ".sha256")
        with open(sha_file) as fh:
            expected = fh.read().split()[0].strip().lower()
        if expected != digest:
            raise StepFailed("sha256 mismatch for %s: expected %s, got %s"
                             % (asset_name, expected, digest))
    else:
        log("    note: the release has no %s; checksum not verifiable" % sha_name)
    extract_dir = os.path.join(WORK, "bin")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(dest) as tar:
        try:
            tar.extractall(extract_dir, filter="data")
        except TypeError:  # Python < 3.12
            tar.extractall(extract_dir)
    binary = os.path.join(extract_dir, "fllm-rs")
    if not os.path.exists(binary):
        raise StepFailed("%s does not contain fllm-rs" % asset_name)
    os.chmod(binary, 0o755)
    return binary, "release %s: %s" % (release.get("tag_name", tag), asset_name)


def acquire_from_source():
    log("    building from source (slow; needs nvcc for the cuda feature)")
    if os.path.exists(os.path.join(REPO_ROOT, "rust", "engine", "Cargo.toml")):
        src = REPO_ROOT
    else:
        src = os.path.join(WORK, "src")
        if not os.path.exists(src):
            sh(["git", "clone", "--depth", "1",
                "https://github.com/%s.git" % RELEASE_REPO, src], timeout=600, check=True)
    if not shutil.which("cargo"):
        log("    installing rustup")
        sh(["bash", "-lc", "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal"],
           timeout=900, check=True)
        os.environ["PATH"] = os.path.expanduser("~/.cargo/bin") + os.pathsep + os.environ["PATH"]
    if shutil.which("apt-get") and os.geteuid() == 0:
        sh(["bash", "-lc", "apt-get update -qq && apt-get install -y -qq --no-install-recommends"
            " cmake clang pkg-config build-essential >/dev/null 2>&1 || true"], timeout=900)
    cc = CACHE.get("cc")
    archs = ("%d" % cc) if cc else "75;80;86;89;90;100;120"
    env = dict(os.environ)
    env["FLLM_CUDA_ARCHS"] = archs
    env["CARGO_BUILD_JOBS"] = str(max(1, (os.cpu_count() or 2) // 2))
    sh(["cargo", "build", "--release", "--locked", "-p", "fllm-engine", "--features",
        "rust-gpu,cuda", "--bin", "fllm-rs"], timeout=BUILD_TIMEOUT, check=True, env=env,
       cwd=src)
    # the Cargo workspace is at the repository root (older layouts had it in rust/): look in both and ask cargo
    candidates = [os.path.join(src, "target", "release", "fllm-rs"), os.path.join(src, "rust", "target", "release", "fllm-rs")]
    try:
        import json as _json
        meta = subprocess.run(["cargo", "metadata", "--format-version", "1", "--no-deps"], cwd=src, capture_output=True, text=True, timeout=120)
        candidates.insert(0, os.path.join(_json.loads(meta.stdout)["target_directory"], "release", "fllm-rs"))
    except Exception:  # noqa: BLE001 - the two fixed places are enough
        pass
    binary = next((c for c in candidates if os.path.exists(c)), candidates[-1])
    if not os.path.exists(binary):
        raise StepFailed("build finished but fllm-rs is in none of: %s" % ", ".join(candidates))
    return binary, "built from source (FLLM_CUDA_ARCHS=%s)" % archs


def acquire_binary(args):
    global BINARY
    if args.binary:
        path = os.path.abspath(os.path.expanduser(args.binary))
        if not os.path.exists(path):
            raise StepFailed("--binary %s does not exist" % path)
        if not os.access(path, os.X_OK):
            try:
                os.chmod(path, 0o755)
            except OSError as exc:
                raise StepFailed("--binary %s is not executable and cannot be made"
                                 " executable: %s" % (path, exc))
        BINARY, source = path, "--binary"
    elif args.build_from_source:
        BINARY, source = acquire_from_source()
    else:
        asset = MUSL_ASSET if args.cpu_only else CUDA_ASSET
        try:
            BINARY, source = acquire_from_release(args.version, asset)
        except StepFailed:
            if args.cpu_only or asset == MUSL_ASSET:
                raise
            log("    %s is not in the release; falling back to %s" % (CUDA_ASSET, MUSL_ASSET))
            BINARY, source = acquire_from_release(args.version, MUSL_ASSET)
            CACHE["cuda_compiled"] = False
    CACHE["binary_source"] = source


# ---------------------------------------------------------------------------
# Plan, reports, main
# ---------------------------------------------------------------------------

def default_work_dir():
    if os.path.isdir("/content"):
        return os.path.join("/content", "fllm_verify")
    return os.path.join(REPO_ROOT, "fllm_verify")


def print_plan(args):
    print("fLLM verification plan (dry run)")
    print("  mode: %s" % ("cpu-only" if args.cpu_only else "gpu"))
    print("  work directory: %s" % os.path.abspath(args.work or default_work_dir()))
    print("  binary: %s" % ("--binary " + args.binary if args.binary else
                             "build from source" if args.build_from_source else
                             "release %s asset %s" % (args.version,
                                                      MUSL_ASSET if args.cpu_only else CUDA_ASSET)))
    print("  steps:")
    print("    - %-12s %s (runs first)" % ("acquire", "get the fllm-rs binary"))
    for key, title, _fn, _t in STEPS:
        if key not in args.steps_set:
            continue
        print("    - %-12s %s" % (key, title))
    print("  one-time downloads (only for the steps above):")
    if not args.cpu_only:
        print("    - the release binary asset")
        print("    - %s/%s (~2 GB)" % (MODELS["dense"]["repo"], MODELS["dense"]["file"]))
        print("    - %s/%s (~4.2 GB)" % (MODELS["moe"]["repo"], MODELS["moe"]["file"]))
        print("    - %s (converted by fLLM, ~1.2 GB)" % IMPORT_MODEL)
    print("    - %s/%s (~0.5 GB) and its mmproj" % (MODELS["vision"]["repo"], MODELS["vision"]["file"]))
    print("    - %s/%s (~0.15 GB)" % (MODELS["embed"]["repo"], MODELS["embed"]["file"]))
    print("  no check itself needs the network after these downloads.")


def print_table():
    widths = [max(len(r["name"]), 8) for r in RESULTS] or [8]
    width = max(widths)
    print()
    print("| %-*s | Result | Time | Details |" % (width, "Step"))
    print("|%s|--------|------|---------|" % ("-" * (width + 2)))
    for r in RESULTS:
        detail = r["detail"].replace("\n", " ").replace("|", "/")[:110]
        print("| %-*s | %-6s | %5.0fs | %s |" % (width, r["name"], r["status"],
                                                 r["seconds"], detail))
    passed = sum(1 for r in RESULTS if r["status"] == "PASS")
    failed = sum(1 for r in RESULTS if r["status"] == "FAIL")
    skipped = sum(1 for r in RESULTS if r["status"] == "SKIP")
    print()
    print("Summary: %d PASS, %d FAIL, %d SKIP" % (passed, failed, skipped))


def write_reports(args):
    result = {
        "schema": 1,
        "script": "scripts/colab/verify_all.py",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": MODE,
        "binary": {"source": CACHE.get("binary_source"),
                   "sha256": sha256_of(BINARY) if BINARY else None},
        "hardware": {"gpus": CACHE.get("gpus", []), "compute_capability": CACHE.get("cc"),
                     "gates": CACHE.get("gates", {})},
        "steps": RESULTS,
        "summary": {"pass": sum(1 for r in RESULTS if r["status"] == "PASS"),
                    "fail": sum(1 for r in RESULTS if r["status"] == "FAIL"),
                    "skip": sum(1 for r in RESULTS if r["status"] == "SKIP")},
    }
    json_path = os.path.join(WORK, "verify_result.json")
    with open(json_path, "w") as fh:
        json.dump(result, fh, indent=2)
    md_path = os.path.join(WORK, "verify_result.md")
    with open(md_path, "w") as fh:
        fh.write("# fLLM verification %s\n\n" % result["generated"])
        fh.write("Mode: %s, binary: %s\n\n" % (result["mode"], result["binary"]["source"]))
        fh.write("| Step | Result | Time | Details |\n|------|--------|------|---------|\n")
        for r in RESULTS:
            fh.write("| %s | %s | %.0fs | %s |\n" % (
                r["name"], r["status"], r["seconds"],
                r["detail"].replace("\n", " ").replace("|", "/")[:200]))
        for r in RESULTS:
            if r["warnings"]:
                fh.write("\nWarnings for %s:\n" % r["name"])
                for warning in r["warnings"]:
                    fh.write("- %s\n" % warning)
    log("wrote %s" % json_path)
    log("wrote %s" % md_path)
    return json_path


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="One-shot GPU verification for fLLM (Colab friendly).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and touch nothing")
    parser.add_argument("--cpu-only", action="store_true",
                        help="run only the checks that do not need a GPU")
    parser.add_argument("--binary", metavar="PATH", help="use this fllm-rs binary")
    parser.add_argument("--build-from-source", action="store_true",
                        help="build the binary from the checkout instead of downloading")
    parser.add_argument("--version", default="latest",
                        help="release tag to download (or 'latest')")
    parser.add_argument("--work", default=None,
                        help="work directory for downloads, logs and results")
    parser.add_argument("--steps", default="all",
                        help="comma list of step keys (default: all; keys in --dry-run)")
    return parser.parse_args(argv)


def main(argv=None):
    global MODE, WORK, RUN_DIR
    args = parse_args(argv if argv is not None else sys.argv[1:])
    mode = "dry-run" if args.dry_run else "cpu-only" if args.cpu_only else "gpu"
    MODE = mode
    default_work = default_work_dir()
    WORK = os.path.abspath(args.work or default_work)
    RUN_DIR = os.path.join(WORK, "logs")

    keys = [k for k, _t, _f, _s in STEPS]
    if args.steps != "all":
        wanted = [s.strip() for s in args.steps.split(",") if s.strip()]
        unknown = [s for s in wanted if s not in keys]
        if unknown:
            print("unknown step(s): %s (known: %s)" % (", ".join(unknown), ", ".join(keys)),
                  file=sys.stderr)
            return 2
        args.steps_set = set(wanted)
    else:
        args.steps_set = set(keys)

    if args.dry_run:
        print_plan(args)
        return 0

    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(os.path.join(WORK, "models"), exist_ok=True)
    log("work directory: %s" % WORK)

    acquire_timeout = BUILD_TIMEOUT if args.build_from_source else 7200
    run_step("acquire", "Acquire binary", lambda: step_acquire(args), acquire_timeout)
    if not BINARY:
        record = RESULTS[-1] if RESULTS else {"detail": "binary acquisition failed"}
        log("no usable binary: %s" % record["detail"])
        print_table()
        write_reports(args)
        return 2

    for key, title, fn, timeout in STEPS:
        if key not in args.steps_set:
            continue
        run_step(key, title, fn, timeout)

    print_table()
    json_path = write_reports(args)
    failed = sum(1 for r in RESULTS if r["status"] == "FAIL")
    if failed:
        log("%d step(s) failed - see %s" % (failed, json_path))
        return 1
    return 0


def step_acquire(args):
    acquire_binary(args)
    return {"detail": CACHE.get("binary_source"), "data": {"path": BINARY}}


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        stop_servers()
        sys.exit(130)
