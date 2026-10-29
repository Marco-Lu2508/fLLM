#!/usr/bin/env python3
"""Compare inference engines on one machine with the same client, model file and prompts.

    python3 scripts/bench_compare.py --model MODEL.gguf --out results.jsonl  CONFIG [CONFIG ...]

A CONFIG is  label=engine:binary[:device]  with engine one of
    fllm       binary = fllm-rs            device = cpu | cuda | vulkan | vulkan@1 | ...   (fllm-rs serve --device; "@" stands for ":" in a config)
    llamacpp   binary = llama-server       device = cpu | gpu   (gpu: -ngl 99)
    ollama     binary = ollama             device = auto        (the GPU backend Ollama finds in its own library directory)
    vllm       binary = python of the vLLM environment   device = model directory (a Hugging Face snapshot; --model is ignored)
    sglang     binary = python of the SGLang environment device = model directory
    flm        binary = flm                device = FastFlowLM model tag, e.g. llama3.2@3b ("@" stands for ":")   (AMD NPU; its own model files)
Environment variables for one config can be given as  label=engine:binary:device:KEY=VAL,KEY=VAL.

Every engine is started once per config, warmed up, then asked `--runs` times for a continuation of a different ~`--prompt-words` word text with
`--gen` new tokens (temperature 0, streaming). Prompt speed = prompt tokens / time to the first token (this includes the first generated token),
decode speed = (generated tokens - 1) / (last token - first token). The median of the runs is reported. Every request is a new text, so no prompt
cache helps; the three engines see exactly the same texts.
"""
import argparse, json, os, signal, statistics, subprocess, sys, time, urllib.request

PORT = 18123
WORDS = ("river stone window garden copper winter harvest lantern meadow compass velvet thunder orchard marble whisper anchor canyon ember "
         "falcon glacier harbor island jasmine kettle lagoon monsoon nectar oyster pebble quartz ripple saffron timber umbra violet willow zephyr").split()


def make_prompt(seed, words):
    # deterministic pseudo-random text; a different one per seed so that nothing is cached between runs
    x = seed * 2654435761 % (1 << 32)
    out = []
    for _ in range(words):
        x = (x * 1103515245 + 12345) % (1 << 31)
        out.append(WORDS[x % len(WORDS)])
    return "Here is a long list of words: " + " ".join(out) + ". Now, in a few paragraphs, a story that uses some of these words begins:"


def post(url, body, timeout=900):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def one_run(base, model, prompt, gen, engine=""):
    body = {"model": model, "prompt": prompt, "max_tokens": gen, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": True, "min_tokens": gen}
    if engine == "llamacpp":
        body["cache_prompt"] = False
    t0 = time.time()
    first = last = None
    chunks = 0
    usage = {}
    with post(base + "/v1/completions", body) as r:
        for raw in r:
            line = raw.decode("utf8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                j = json.loads(data)
            except ValueError:
                continue
            if j.get("usage"):
                usage = j["usage"]
            for c in j.get("choices", []):
                if c.get("text"):
                    now = time.time()
                    if first is None:
                        first = now
                    last = now
                    chunks += 1
    if first is None:
        raise RuntimeError("no token received")
    ptok = usage.get("prompt_tokens") or 0
    ctok = usage.get("completion_tokens") or chunks
    return {"prompt_tokens": ptok, "gen_tokens": ctok, "ttft": first - t0,
            "pp": ptok / (first - t0) if ptok else 0.0,
            "tg": (ctok - 1) / (last - first) if ctok > 1 and last > first else 0.0}


def wait_ready(base, path, proc, seconds=600):
    end = time.time() + seconds
    while time.time() < end:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(base + path, timeout=3).read()
            return True
        except Exception:
            time.sleep(1.5)
    return False


def start(engine, binary, device, model, env_extra, log):
    env = dict(os.environ)
    env.update(env_extra)
    base = f"http://127.0.0.1:{PORT}"
    if engine == "fllm":
        cmd = [binary, "serve", "--model", model, "--model-name", "m", "--device", device, "--port", str(PORT)]
        ready, name = "/ready", "m"
    elif engine == "llamacpp":
        cmd = [binary, "-m", model, "--port", str(PORT), "--host", "127.0.0.1", "-c", "4096", "-ngl", "99" if device != "cpu" else "0",
               "--no-webui", "--parallel", "1"]
        ready, name = "/health", "m"
    elif engine == "ollama":
        env["OLLAMA_HOST"] = f"127.0.0.1:{PORT}"
        env["OLLAMA_MODELS"] = os.path.expanduser("~/bench/ollama-models")
        env["OLLAMA_NUM_PARALLEL"] = "1"
        env["OLLAMA_KEEP_ALIVE"] = "30m"
        env["OLLAMA_CONTEXT_LENGTH"] = "4096"
        cmd = [binary, "serve"]
        ready, name = "/api/tags", "bench"
    elif engine in ("vllm", "sglang"):
        # binary = the python of the engine's environment, device = the model directory (a Hugging Face snapshot, e.g. an AWQ checkpoint)
        if engine == "vllm":
            cmd = [binary, "-m", "vllm.entrypoints.openai.api_server", "--model", device, "--served-model-name", "m", "--port", str(PORT),
                   "--max-model-len", "4096", "--gpu-memory-utilization", "0.85", "--dtype", "float16"]
        else:
            cmd = [binary, "-m", "sglang.launch_server", "--model-path", device, "--served-model-name", "m", "--port", str(PORT),
                   "--context-length", "4096", "--mem-fraction-static", "0.8", "--dtype", "float16",
                   "--attention-backend", "triton", "--sampling-backend", "pytorch"]
        ready, name = "/health", "m"
    elif engine == "flm":
        # FastFlowLM on an AMD NPU: the model is a FastFlowLM tag (device field, "llama3.2@3b" stands for llama3.2:3b), not the GGUF file
        cmd = [binary, "serve", device, "--port", str(PORT), "--pmode", "performance"]
        ready, name = "/v1/models", device
    else:
        raise SystemExit(f"unknown engine {engine}")
    proc = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env, start_new_session=True)
    if not wait_ready(base, ready, proc, 1200 if engine in ("vllm", "sglang") else 600):
        stop(proc)
        raise RuntimeError(f"{engine} did not start, see {log}")
    if engine == "ollama":
        mf = os.path.expanduser("~/bench/Modelfile")
        open(mf, "w").write(f"FROM {model}\nPARAMETER num_ctx 4096\n")
        subprocess.run([binary, "create", "bench", "-f", mf], env=env, check=True, capture_output=True)
    return proc, base, name


def stop(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    time.sleep(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="bench_results.jsonl")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--prompt-words", type=int, default=380, help="about 512 tokens")
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--tag", default="", help="free text stored with every result (machine, model)")
    ap.add_argument("configs", nargs="+")
    a = ap.parse_args()
    print(f"model {os.path.basename(a.model)}  prompt ~{a.prompt_words} words  gen {a.gen}  runs {a.runs}", flush=True)
    for cfg in a.configs:
        label, spec = cfg.split("=", 1)
        parts = spec.split(":")
        engine, binary = parts[0], parts[1]
        device = parts[2].replace("@", ":") if len(parts) > 2 else "auto"
        env_extra = dict(kv.split("=", 1) for kv in parts[3].split(",")) if len(parts) > 3 and parts[3] else {}
        row = {"label": label, "engine": engine, "device": device, "model": os.path.basename(a.model), "tag": a.tag}
        try:
            proc, base, name = start(engine, os.path.expanduser(binary), device, a.model, env_extra, f"/tmp/bench_{label}.log")
            try:
                one_run(base, name, make_prompt(99, 60), 16, engine)                # warm-up (shader / graph / kernel compilation, model load)
                one_run(base, name, make_prompt(98, a.prompt_words), 16, engine)
                runs = [one_run(base, name, make_prompt(i + 1, a.prompt_words), a.gen, engine) for i in range(a.runs)]
            finally:
                stop(proc)
            row.update(pp=round(statistics.median(r["pp"] for r in runs), 1), tg=round(statistics.median(r["tg"] for r in runs), 1),
                       prompt_tokens=runs[0]["prompt_tokens"], gen_tokens=runs[0]["gen_tokens"], ttft=round(statistics.median(r["ttft"] for r in runs), 2))
        except Exception as e:
            row["error"] = str(e)[:200]
        print(json.dumps(row), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
