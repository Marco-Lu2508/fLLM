#!/usr/bin/env python3
"""Prints the greedy answer of a running fLLM server to one fixed long prompt: PORT [MAX_TOKENS].

Run it against a server on the processor and against one with the NPU offload (for example FLLM_NPU_HOST=1), then compare the two
texts (`diff`). The NPU computes in bf16, so the text can differ after some tokens; the first lines should agree.
"""
import json
import sys
import urllib.request

PROMPT = (
    "Below is a short technical note. Read it and answer the question at the end in two sentences.\n\n"
    + " ".join(
        f"Section {i}: the pump station number {i} moves {40 + 3 * i} liters per minute through a pipe of {12 + i} centimeters, "
        f"and its operator writes the pressure reading {1.5 + i / 10:.1f} bar into a logbook every {5 + i % 4} minutes."
        for i in range(1, 15)
    )
    + "\n\nQuestion: Which station moves the most water per minute, and how much?"
)


def main() -> None:
    port = int(sys.argv[1])
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    model = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=30))["data"][0]["id"]
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": max_tokens, "temperature": 0, "stream": False}
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    reply = json.load(urllib.request.urlopen(request, timeout=1800))
    print(f"prompt tokens: {reply['usage']['prompt_tokens']}")
    print(reply["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
