#!/usr/bin/env python3
"""Aggregate generation speed of a running fLLM server under N simultaneous requests: PORT [STREAMS [WORDS [MAX_TOKENS]]].

Each stream sends a different short prompt and asks for MAX_TOKENS (default 64) new tokens; the script prints the tokens generated per
second over all streams together. With the NPU offload (`FLLM_NPU_HOST=1 FLLM_XDNA_MIN_ROWS=4`) the decode steps of several sequences form
a batch of rows that the NPU computes in one call.
"""
import json
import random
import sys
import threading
import time
import urllib.request


def model_name(port: int) -> str:
    return json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=30))["data"][0]["id"]


def one(port: int, model: str, words: int, max_tokens: int, out: list) -> None:
    text = " ".join(random.choice("alpha beta gamma delta river cloud engine paper".split()) + str(random.randint(0, 9999)) for _ in range(words))
    body = {"model": model, "messages": [{"role": "user", "content": "Continue this list: " + text}], "max_tokens": max_tokens, "temperature": 0.7, "stream": False}
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    reply = json.load(urllib.request.urlopen(request, timeout=1800))
    out.append(reply["usage"]["completion_tokens"])


def main() -> None:
    port = int(sys.argv[1])
    streams = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    words = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    model = model_name(port)
    random.seed(time.time())
    for round_ in range(2):
        out: list = []
        start = time.time()
        threads = [threading.Thread(target=one, args=(port, model, words, max_tokens, out)) for _ in range(streams)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        seconds = time.time() - start
        print(f"round {round_}: {streams} streams, {sum(out)} tokens in {seconds:.2f} s = {sum(out) / seconds:.1f} tok/s together")


if __name__ == "__main__":
    main()
