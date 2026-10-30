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

With `--prompt-words 60,380,1500` and `--concurrency 1,4,8` the same server is measured for every prompt length and for every number of requests
sent at the same moment (batching): the row then holds one entry per combination, `agg_tps` being all generated tokens divided by the time until the
last request finished (the throughput a service would see), `tg` the decode speed one request got, `ttft` the median time to the first token.

Every engine is started once per config, warmed up, then asked `--runs` times for a continuation of a different ~`--prompt-words` word text with
`--gen` new tokens (temperature 0, streaming). Prompt speed = prompt tokens / time to the first token (this includes the first generated token),
decode speed = (generated tokens - 1) / (last token - first token). The median of the runs is reported. Every request is a new text, so no prompt
cache helps; the three engines see exactly the same texts.
"""
import argparse, json, os, signal, statistics, subprocess, sys, tempfile, threading, time, urllib.request

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


def start(engine, binary, device, model, env_extra, log, slots=1, ctx=4096):
    env = dict(os.environ)
    env.update(env_extra)
    base = f"http://127.0.0.1:{PORT}"
    if engine == "fllm":
        cmd = [binary, "serve", "--model", model, "--model-name", "m", "--device", device, "--port", str(PORT)]
        ready, name = "/ready", "m"
    elif engine == "llamacpp":
        cmd = [binary, "-m", model, "--port", str(PORT), "--host", "127.0.0.1", "-c", str(ctx * slots), "-ngl", "99" if device != "cpu" else "0",
               "--no-webui", "--parallel", str(slots)]
        ready, name = "/health", "m"
    elif engine == "ollama":
        env["OLLAMA_HOST"] = f"127.0.0.1:{PORT}"
        env["OLLAMA_MODELS"] = os.path.expanduser("~/bench/ollama-models")
        env["OLLAMA_NUM_PARALLEL"] = str(slots)
        env["OLLAMA_KEEP_ALIVE"] = "30m"
        env["OLLAMA_CONTEXT_LENGTH"] = str(ctx)
        cmd = [binary, "serve"]
        ready, name = "/api/tags", "bench"
    elif engine in ("vllm", "sglang"):
        # binary = the python of the engine's environment, device = the model directory (a Hugging Face snapshot, e.g. an AWQ checkpoint)
        if engine == "vllm":
            cmd = [binary, "-m", "vllm.entrypoints.openai.api_server", "--model", device, "--served-model-name", "m", "--port", str(PORT),
                   "--max-model-len", str(ctx), "--gpu-memory-utilization", "0.85", "--dtype", "float16"]
        else:
            cmd = [binary, "-m", "sglang.launch_server", "--model-path", device, "--served-model-name", "m", "--port", str(PORT),
                   "--context-length", str(ctx), "--mem-fraction-static", "0.8", "--dtype", "float16",
                   "--attention-backend", "triton", "--sampling-backend", "pytorch"]
        ready, name = "/health", "m"
    elif engine == "flm":
        # FastFlowLM on an AMD NPU: the model is a FastFlowLM tag (device field, "llama3.2@3b" stands for llama3.2:3b), not the GGUF file
        cmd = [binary, "serve", device, "--port", str(PORT), "--pmode", "performance"]
        ready, name = "/v1/models", device
    else:
        raise SystemExit(f"unknown engine {engine}")
    proc = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env, **({} if os.name == "nt" else {"start_new_session": True}))
    if not wait_ready(base, ready, proc, 1200 if engine in ("vllm", "sglang") else 600):
        stop(proc)
        raise RuntimeError(f"{engine} did not start, see {log}")
    if engine == "ollama":
        mf = os.path.expanduser("~/bench/Modelfile")
        open(mf, "w").write(f"FROM {model}\nPARAMETER num_ctx {ctx}\n")
        subprocess.run([binary, "create", "bench", "-f", mf], env=env, check=True, capture_output=True)
    return proc, base, name


def burst(base, model, words, gen, engine, n, seed):
    """n requests at the same moment, each with its own text."""
    res, errs = [None] * n, []

    def work(i):
        try:
            res[i] = one_run(base, model, make_prompt(seed * 100 + i, words), gen, engine)
        except Exception as e:  # noqa: BLE001
            errs.append(str(e)[:100])

    t0 = time.time()
    ts = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.time() - t0
    ok = [r for r in res if r]
    if not ok:
        raise RuntimeError("all requests failed: " + (errs[0] if errs else "?"))
    return {"n": n, "ok": len(ok), "wall": wall, "agg_tps": sum(r["gen_tokens"] for r in ok) / wall,
            "tg": statistics.median(r["tg"] for r in ok), "ttft": statistics.median(r["ttft"] for r in ok),
            "prompt_tokens": ok[0]["prompt_tokens"]}


def stop(proc):
    if os.name == "nt":
        # no process groups here: taskkill /T ends the server and anything it started
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        try:
            proc.wait(timeout=20)
        except Exception:
            pass
        time.sleep(3)
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    time.sleep(3)


def split_config(spec):
    """engine:binary[:device[:ENV]] -> parts; a Windows drive letter in the binary path (fllm:C:/x/fllm-rs.exe:cpu) is not a separator."""
    parts = spec.split(":")
    if len(parts) > 2 and len(parts[1]) == 1 and parts[1].isalpha() and parts[2][:1] in ("\\", "/"):
        parts[1:3] = [parts[1] + ":" + parts[2]]
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="bench_results.jsonl")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--prompt-words", default="380", help="about 512 tokens; a comma separated list measures several prompt lengths")
    ap.add_argument("--concurrency", default="1", help="comma separated numbers of simultaneous requests (batching)")
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--tag", default="", help="free text stored with every result (machine, model)")
    ap.add_argument("configs", nargs="+")
    a = ap.parse_args()
    lengths = [int(x) for x in a.prompt_words.split(",")]
    levels = [int(x) for x in a.concurrency.split(",")]
    ctx = max(4096, int(max(lengths) * 1.6) + a.gen + 256)
    print(f"model {os.path.basename(a.model)}  prompt ~{lengths} words  concurrency {levels}  gen {a.gen}  runs {a.runs}", flush=True)
    for cfg in a.configs:
        label, spec = cfg.split("=", 1)
        parts = split_config(spec)
        engine, binary = parts[0], parts[1]
        device = parts[2].replace("@", ":") if len(parts) > 2 else "auto"
        env_extra = dict(kv.split("=", 1) for kv in parts[3].split(",")) if len(parts) > 3 and parts[3] else {}
        row = {"label": label, "engine": engine, "device": device, "model": os.path.basename(a.model), "tag": a.tag}
        try:
            proc, base, name = start(engine, os.path.expanduser(binary), device, a.model, env_extra, os.path.join(tempfile.gettempdir(), f"bench_{label}.log"), max(levels), ctx)
            try:
                one_run(base, name, make_prompt(99, 60), 16, engine)                # warm-up (shader / graph / kernel compilation, model load)
                grid = []
                for words in lengths:
                    one_run(base, name, make_prompt(98, words), 16, engine)
                    for n in levels:
                        if n == 1:
                            runs = [one_run(base, name, make_prompt(i + 1, words), a.gen, engine) for i in range(a.runs)]
                            grid.append({"n": 1, "words": words, "prompt_tokens": runs[0]["prompt_tokens"], "pp": round(statistics.median(r["pp"] for r in runs), 1),
                                         "tg": round(statistics.median(r["tg"] for r in runs), 1), "ttft": round(statistics.median(r["ttft"] for r in runs), 2),
                                         "agg_tps": round(statistics.median(r["tg"] for r in runs), 1), "gen_tokens": runs[0]["gen_tokens"]})
                        else:
                            try:
                                b = burst(base, name, words, a.gen, engine, n, 7 + n)
                                grid.append({"n": n, "words": words, "prompt_tokens": b["prompt_tokens"], "agg_tps": round(b["agg_tps"], 1), "tg": round(b["tg"], 1),
                                             "ttft": round(b["ttft"], 2), "ok": b["ok"]})
                            except Exception as e:  # noqa: BLE001
                                grid.append({"n": n, "words": words, "error": str(e)[:120]})
            finally:
                stop(proc)
            first = next((g for g in grid if g["n"] == 1), grid[0])
            row.update(pp=first.get("pp", 0), tg=first.get("tg", 0), prompt_tokens=first.get("prompt_tokens", 0), gen_tokens=first.get("gen_tokens", 0),
                       ttft=first.get("ttft", 0), grid=grid)
        except Exception as e:
            row["error"] = str(e)[:200]
        print(json.dumps(row), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
