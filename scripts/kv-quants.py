#!/usr/bin/env python3
"""Two different quantizations of the same model on ONE disk-tier folder, and a changed word.

Part 1 (quants): the Q8_0 server writes the KV blocks of a prompt to the folder. A Q4_K_M server (same architecture, same geometry, other
weights, so other KV values) is started on the SAME folder and asked the same prompt. If it reads the Q8_0 blocks, `cached` is > 0 and its
answer can differ from what Q4_K_M computes alone (the control runs Q4_K_M with an empty folder).
Part 2 (a changed word): one server, a long prompt, then the same prompt with one word changed at three places (start, middle, end).

Usage: kv-quants.py <fllm-rs> <q8.gguf> <q4.gguf>
"""
import json, os, shutil, signal, subprocess, sys, time, urllib.error, urllib.request

BIN, Q8, Q4 = sys.argv[1:4]
PORT = 8191
DIR = "/tmp/fllm-kvquants"


def doc(k=0, sections=22, word=None, at=None):
    parts = []
    for i in range(1, sections):
        line = f"Document {k} section {i}: unit {k}-{i} records value {(i * 37 + k * 11) % 101} and reports it every {i % 7 + 2} seconds."
        if word is not None and i == at:
            line = line.replace("records", word)
        parts.append(line)
    return " ".join(parts)


def serve(tag, model, disk):
    env = {**os.environ, "FLLM_OUTPUT_CACHE": "off"}
    for k in ("FLLM_KV_DISK_DIR", "FLLM_KV_RAM_BYTES"):
        env.pop(k, None)
    if disk:
        env["FLLM_KV_DISK_DIR"] = disk
        env["FLLM_KV_DISK_BYTES"] = str(512 << 20)
    proc = subprocess.Popen([BIN, "serve", "--model", model, "--device", "cpu", "--port", str(PORT), "--speculative", "off", "--ctx", "4096"],
                            env=env, stdout=subprocess.DEVNULL, stderr=open(f"/tmp/fllm-kvquants-{tag}.log", "w"), start_new_session=True)
    for _ in range(180):
        time.sleep(1)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            return proc
        except Exception:
            if proc.poll() is not None:
                raise SystemExit(f"{tag}: server exited, see /tmp/fllm-kvquants-{tag}.log")
    raise SystemExit(f"{tag}: not ready")


def stop(proc):
    os.killpg(proc.pid, signal.SIGTERM)
    proc.wait(timeout=60)


def call(path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=900))
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{path}: HTTP {e.code} {e.read()[:300]}")


def ask(text, question="What does the first unit record?"):
    mid = call("/v1/models")["data"][0]["id"]
    t = time.perf_counter()
    r = call("/v1/chat/completions", {"model": mid, "max_tokens": 24, "temperature": 0, "messages": [{"role": "system", "content": text}, {"role": "user", "content": question}]})
    return time.perf_counter() - t, r["usage"]["prompt_tokens"], (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0), r["choices"][0]["message"]["content"]


def quants():
    shutil.rmtree(DIR, ignore_errors=True)
    os.makedirs(DIR)
    p = serve("q8", Q8, DIR)
    try:
        t, n, c, text_q8 = ask(doc())
        time.sleep(3)
        print(f"[Q8_0 writes]   {t:.2f} s, {n} tokens, cached {c}, files on disk {len(os.listdir(DIR))}", flush=True)
    finally:
        stop(p)
    p = serve("q4-shared", Q4, DIR)
    try:
        t, n, c, text_shared = ask(doc())
        print(f"[Q4_K_M, same folder]  {t:.2f} s, cached {c} of {n}, reloaded {call('/admin/kv-tier').get('reloaded_blocks')}, hits {call('/admin/kv-tier').get('hits')}", flush=True)
    finally:
        stop(p)
    shutil.rmtree(DIR, ignore_errors=True)
    p = serve("q4-alone", Q4, None)
    try:
        t, n, c, text_alone = ask(doc())
        print(f"[Q4_K_M, empty folder (control)]  {t:.2f} s, cached {c} of {n}", flush=True)
    finally:
        stop(p)
    print(f"[quants] Q4 answer with the shared folder equals Q4 alone: {text_shared == text_alone}; Q8 answer equals Q4 alone: {text_q8 == text_alone}", flush=True)


def changed_word(model, tag):
    p = serve(tag, model, None)
    try:
        t0, n, c0, _ = ask(doc())
        print(f"[{tag}] original, cold: {t0:.2f} s, {n} tokens, cached {c0}", flush=True)
        for label, at in (("changed word in the first section", 1), ("changed word in the middle", 11), ("changed word near the end", 20)):
            t, n2, c, _ = ask(doc(word="logs", at=at))
            print(f"[{tag}] {label}: {t:.2f} s, cached {c} of {n2}", flush=True)
        t, n2, c, _ = ask(doc())
        print(f"[{tag}] the original again: {t:.2f} s, cached {c} of {n2}", flush=True)
    finally:
        stop(p)


quants()
changed_word(Q8, "word-q8")
changed_word(Q4, "word-q4")
