#!/usr/bin/env python3
"""Build fLLM with the TPU backend on Google Colab and check it on a TPU (v2-8 or v5e-1 runtime).

Run it in a Colab notebook that has a TPU runtime (Runtime > Change runtime type > TPU), after cloning or uploading the fLLM
repository:

    !python fllm/scripts/colab/tpu_check.py --model-uri hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf

What it does, step by step (each step is reported as ok / FAILED / skipped, a failure does not stop the later steps that can still
run):

  1. environment   Python, CPU count, memory, TPU device files, whether JAX/libtpu are already imported in a running kernel.
  2. rust          installs the Rust toolchain with rustup when `cargo` is missing.
  3. libtpu        finds the PJRT plugin `libtpu.so` of the `libtpu` pip package (installs the package when it is missing).
  4. build         `cargo build --release -p fllm-engine --features tpu-pjrt` (a Colab VM needs 10-25 minutes).
  5. devices       `fllm-rs devices`: the TPU cores must be listed.
  6. selftest      `fllm-rs selftest-gpu --device tpu --json ...` with exact arithmetic (FLLM_TPU_EXACT=1): every operation against
                   the CPU backend.
  7. model         loads a small GGUF model on the TPU (server mode), greedy-decodes a prompt and compares the text with the CPU
                   (the same model, temperature 0). Needs --model PATH or --model-uri hf://...
  8. report        writes tpu_check_report.json (all outputs, timings, the first lines of every log) next to the script.

Nothing here is verified by the authors on a TPU (level L0 of the project); the report is what the first real run sends back.
Only one process can open a TPU: do not run JAX or another TPU program at the same time (restart the runtime if a notebook cell
imported jax), and do not start the script twice.
"""

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = []  # (step, status, seconds, note)
REPORT = {"steps": {}, "env": {}}


def log(message):
    print(message, flush=True)


def run(cmd, cwd=None, env=None, timeout=None, log_file=None, check=True):
    """Run a command, stream its output to the console and to `log_file`, return (returncode, full output)."""
    log(f"$ {' '.join(str(c) for c in cmd)}")
    merged = dict(os.environ)
    merged.update(env or {})
    process = subprocess.Popen(
        [str(c) for c in cmd], cwd=cwd, env=merged, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace"
    )
    lines = []
    start = time.time()
    try:
        for line in process.stdout:
            lines.append(line)
            sys.stdout.write(line)
            if timeout and time.time() - start > timeout:
                process.kill()
                lines.append(f"[tpu_check] killed after {timeout} s\n")
                break
        process.wait(timeout=30)
    except Exception:
        process.kill()
        raise
    output = "".join(lines)
    if log_file:
        Path(log_file).write_text(output)
    if check and process.returncode != 0:
        raise RuntimeError(f"command failed with exit code {process.returncode}: {' '.join(str(c) for c in cmd)}")
    return process.returncode, output


def step(name):
    """Decorator: run a step, record its status and never let an exception escape."""

    def wrap(fn):
        def inner(*args, **kwargs):
            log(f"\n===== {name} =====")
            start = time.time()
            try:
                note = fn(*args, **kwargs)
                status = "skipped" if note and str(note).startswith("skipped") else "ok"
                RESULTS.append((name, status, time.time() - start, note or ""))
                return True if status == "ok" else None
            except Exception as error:  # noqa: BLE001 - the report must survive every failure
                RESULTS.append((name, "FAILED", time.time() - start, str(error)))
                log(f"[tpu_check] {name} FAILED: {error}")
                return False

        return inner

    return wrap


def meminfo():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


@step("environment")
def check_environment(args):
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "memory_mb": meminfo(),
        "accel_devices": sorted(str(p) for p in Path("/dev").glob("accel*")) + sorted(str(p) for p in Path("/dev").glob("vfio/*")),
        "tpu_env": {k: v for k, v in os.environ.items() if k.startswith(("TPU_", "COLAB_TPU", "XRT_TPU", "PJRT_"))},
        "jax_imported_here": "jax" in sys.modules,
    }
    REPORT["env"] = info
    log(json.dumps(info, indent=2))
    if not info["accel_devices"] and not info["tpu_env"]:
        raise RuntimeError("no TPU device files (/dev/accel*, /dev/vfio) and no TPU environment: select a TPU runtime")
    return f"{info['cpus']} CPUs, {info['memory_mb']} MB, {len(info['accel_devices'])} device files"


@step("rust")
def ensure_rust(args):
    cargo_bin = Path.home() / ".cargo" / "bin"
    os.environ["PATH"] = f"{cargo_bin}{os.pathsep}{os.environ['PATH']}"
    if shutil.which("cargo"):
        run(["cargo", "--version"])
        return "cargo present"
    run(["bash", "-c", "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal"], timeout=900)
    if not shutil.which("cargo"):
        raise RuntimeError("cargo is still missing after rustup")
    run(["cargo", "--version"])
    return "installed with rustup"


def libtpu_path():
    """Path of libtpu.so from the `libtpu` package, or None."""
    code = (
        "import os, sys\n"
        "try:\n"
        "    import libtpu\n"
        "except ImportError:\n"
        "    sys.exit(3)\n"
        "get = getattr(libtpu, 'get_library_path', None)\n"
        "path = get() if get else os.path.join(os.path.dirname(libtpu.__file__), 'libtpu.so')\n"
        "print(path)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    path = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    return path if path and Path(path).is_file() else None


@step("libtpu")
def ensure_libtpu(args):
    if args.libtpu:
        path = args.libtpu
    else:
        path = libtpu_path()
        indexes = [None, "https://storage.googleapis.com/libtpu-releases/index.html", "https://storage.googleapis.com/jax-releases/libtpu_releases.html"]
        for index in indexes:
            if path:
                break
            cmd = [sys.executable, "-m", "pip", "install", "--quiet", "libtpu"]
            if index:
                cmd += ["-f", index]
            code, _ = run(cmd, check=False, timeout=900)
            if code == 0:
                path = libtpu_path()
    if not path or not Path(path).is_file():
        raise RuntimeError("libtpu.so not found: pip install libtpu (or pass --libtpu /path/to/libtpu.so)")
    os.environ["FLLM_TPU_LIBRARY"] = path
    REPORT["env"]["libtpu"] = path
    log(f"libtpu: {path} ({Path(path).stat().st_size // (1 << 20)} MB)")
    return path


def source_dir(args):
    root = Path(args.source).resolve() if args.source else REPO
    if not (root / "rust" / "engine" / "Cargo.toml").is_file():
        raise RuntimeError(f"{root} is not an fLLM checkout (rust/engine/Cargo.toml missing); pass --source DIR")
    return root


@step("build")
def build(args):
    if args.bin:
        if not Path(args.bin).is_file():
            raise RuntimeError(f"--bin {args.bin} does not exist")
        return f"skipped (using {args.bin})"
    root = source_dir(args)
    env = {"FLLM_SKIP_GPU_KERNELS": "1", "CARGO_BUILD_JOBS": str(args.jobs or max(1, min(os.cpu_count() or 2, 8)))}
    cmd = ["cargo", "build", "--release", "-p", "fllm-engine", "--features", args.features]
    run(cmd, cwd=root, env=env, timeout=args.build_timeout, log_file=HERE / "tpu_check_build.log")
    return "built"


def binary(args):
    if args.bin:
        return Path(args.bin)
    root = source_dir(args)
    for candidate in (root / "target" / "release" / "fllm-rs", root / "rust" / "target" / "release" / "fllm-rs"):
        if candidate.is_file():
            return candidate
    raise RuntimeError("fllm-rs binary not found after the build")


TPU_ENV = {"FLLM_TPU_EXACT": "1", "FLLM_TPU_DEBUG": "1"}


@step("devices")
def devices(args):
    exe = binary(args)
    _, output = run([exe, "devices"], env=TPU_ENV, timeout=600, log_file=HERE / "tpu_check_devices.log")
    REPORT["steps"]["devices_output"] = output[-4000:]
    if "tpu" not in output.lower():
        raise RuntimeError("`fllm-rs devices` does not list a TPU; see tpu_check_devices.log (plugin not found or client creation failed)")
    return "TPU listed"


@step("selftest")
def selftest(args):
    exe = binary(args)
    json_path = HERE / "tpu_selftest.json"
    json_path.unlink(missing_ok=True)
    code, output = run(
        [exe, "selftest-gpu", "--device", "tpu", "--json", json_path],
        env=TPU_ENV,
        timeout=args.selftest_timeout,
        log_file=HERE / "tpu_check_selftest.log",
        check=False,
    )
    summary = {}
    if json_path.is_file():
        summary = json.loads(json_path.read_text())
        REPORT["steps"]["selftest"] = summary
    failures = [line for line in output.splitlines() if line.rstrip().endswith("FAIL") or " FAIL:" in line]
    REPORT["steps"]["selftest_failures"] = failures[:200]
    if code != 0 or not summary.get("ok"):
        raise RuntimeError(f"selftest-gpu: {summary.get('failed', '?')} of {summary.get('checks', '?')} checks failed (exit {code}); first failures: " + " | ".join(failures[:5]))
    return f"{summary.get('checks')} checks passed"


def http(method, url, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def serve_and_ask(exe, model, device, port, prompt, tokens, env, log_file, ready_timeout):
    """Start the server on `device`, ask the prompt twice (greedy), stop the server. Returns (text, first_seconds, second_seconds)."""
    cmd = [str(exe), "serve", "--model", str(model), "--device", device, "--port", str(port), "--host", "127.0.0.1", "--ctx", "512", "--max-seqs", "1"]
    merged = dict(os.environ)
    merged.update(env)
    handle = open(log_file, "w")
    server = subprocess.Popen(cmd, env=merged, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + ready_timeout
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"the {device} server exited with code {server.returncode}; see {log_file}")
            try:
                http("GET", base + "/health", timeout=5)
                break
            except (urllib.error.URLError, ConnectionError, OSError):
                if time.time() > deadline:
                    raise RuntimeError(f"the {device} server was not ready after {ready_timeout} s; see {log_file}")
                time.sleep(2)
        model_id = http("GET", base + "/v1/models")["data"][0]["id"]
        body = {"model": model_id, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": tokens}
        timings, text = [], ""
        for _ in range(2):
            start = time.time()
            answer = http("POST", base + "/v1/chat/completions", body, timeout=3600)
            timings.append(time.time() - start)
            text = answer["choices"][0]["message"]["content"]
        return text, timings[0], timings[1]
    finally:
        try:
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(server.pid), signal.SIGKILL)
        handle.close()


@step("model")
def model_smoke(args):
    if not args.model and not args.model_uri:
        return "skipped (no --model / --model-uri)"
    exe = binary(args)
    model = args.model
    if not model:
        _, output = run([exe, "pull", args.model_uri], timeout=3600, env={})
        model = output.strip().splitlines()[-1].strip()
    if not Path(model).is_file():
        raise RuntimeError(f"model file {model} not found")
    tpu_text, first, second = serve_and_ask(exe, model, "tpu", args.port, args.prompt, args.tokens, TPU_ENV, HERE / "tpu_check_server_tpu.log", args.ready_timeout)
    log(f"TPU answer ({first:.1f} s with compilation, {second:.1f} s warm): {tpu_text!r}")
    cpu_text, cpu_first, _ = serve_and_ask(exe, model, "cpu", args.port + 1, args.prompt, args.tokens, {}, HERE / "tpu_check_server_cpu.log", args.ready_timeout)
    log(f"CPU answer ({cpu_first:.1f} s): {cpu_text!r}")
    REPORT["steps"]["model"] = {
        "model": str(model),
        "prompt": args.prompt,
        "tpu_text": tpu_text,
        "cpu_text": cpu_text,
        "tpu_seconds_cold": first,
        "tpu_seconds_warm": second,
        "cpu_seconds": cpu_first,
        "identical": tpu_text == cpu_text,
    }
    if tpu_text != cpu_text:
        common = next((i for i, (a, b) in enumerate(zip(tpu_text, cpu_text)) if a != b), min(len(tpu_text), len(cpu_text)))
        raise RuntimeError(f"the TPU text differs from the CPU text after {common} characters (level L2 not reached)")
    return f"identical to the CPU ({args.tokens} tokens max), warm run {second:.1f} s"


def write_report(args):
    REPORT["results"] = [{"step": n, "status": s, "seconds": round(t, 1), "note": note} for n, s, t, note in RESULTS]
    out = Path(args.report)
    out.write_text(json.dumps(REPORT, indent=2))
    log("\n===== summary =====")
    for name, status, seconds, note in RESULTS:
        log(f"{name:<12} {status:<8} {seconds:7.1f} s  {note}")
    log(f"\nreport: {out}")
    log("send back: " + ", ".join(str(HERE / n) for n in ("tpu_check_report.json", "tpu_selftest.json", "tpu_check_devices.log", "tpu_check_selftest.log", "tpu_check_server_tpu.log")))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", help="fLLM checkout (default: the checkout this script is in)")
    parser.add_argument("--bin", help="use this fllm-rs binary instead of building one (it must be built with --features tpu-pjrt)")
    parser.add_argument("--features", default="tpu-pjrt", help="cargo features (default: tpu-pjrt)")
    parser.add_argument("--jobs", type=int, help="cargo build jobs (default: the CPU count, at most 8)")
    parser.add_argument("--libtpu", help="path of libtpu.so (default: from the libtpu pip package)")
    parser.add_argument("--model", help="a GGUF file to run on the TPU and on the CPU")
    parser.add_argument("--model-uri", help="hf://owner/repo/file.gguf, fetched with `fllm-rs pull`")
    parser.add_argument("--prompt", default="Write one sentence about the sea.")
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--build-timeout", type=int, default=3600)
    parser.add_argument("--selftest-timeout", type=int, default=3600)
    parser.add_argument("--ready-timeout", type=int, default=900)
    parser.add_argument("--report", default=str(HERE / "tpu_check_report.json"))
    parser.add_argument("--skip", nargs="*", default=[], choices=["build", "devices", "selftest", "model"], help="steps to leave out")
    args = parser.parse_args()

    check_environment(args)
    ensure_rust(args)
    ensure_libtpu(args)
    if "build" not in args.skip:
        build(args)
    else:
        RESULTS.append(("build", "skipped", 0.0, "--skip build"))
    ok = {r[0]: r[1] for r in RESULTS}
    if ok.get("build") == "FAILED":
        log("[tpu_check] the build failed: the remaining steps need the binary")
    else:
        if "devices" not in args.skip:
            devices(args)
        if "selftest" not in args.skip:
            selftest(args)
        if "model" not in args.skip:
            model_smoke(args)
    write_report(args)
    sys.exit(1 if any(status == "FAILED" for _, status, _, _ in RESULTS) else 0)


if __name__ == "__main__":
    main()
