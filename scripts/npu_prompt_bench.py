#!/usr/bin/env python3
"""Times prompt processing on a running fLLM server: PORT WORDS [MAX_TOKENS [REPEATS]].

Every request uses a new random text (the server has an output cache), asks for MAX_TOKENS (default 1) new tokens and prints the
prompt length, the total time and the prompt tokens per second (total time includes one generated token, so use it for long prompts).
Example: start `fllm-rs serve --device npu:0 --port 18461 ...`, then `python3 scripts/npu_prompt_bench.py 18461 380 1 3`.
"""
import json
import random
import sys
import time
import urllib.request

VOCAB = "alpha beta gamma delta river mountain cloud engine paper window garden silver copper violet orange market bridge forest ocean".split()


def model_name(port: int) -> str:
    return json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=30))["data"][0]["id"]


def one(port: int, words: int, max_tokens: int) -> tuple[int, int, float]:
    text = " ".join(random.choice(VOCAB) + str(random.randint(0, 99999)) for _ in range(words))
    body = {
        "model": model_name(port),
        "messages": [{"role": "user", "content": "Summarize in one sentence: " + text}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    start = time.time()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    reply = json.load(urllib.request.urlopen(request, timeout=1800))
    usage = reply["usage"]
    return usage["prompt_tokens"], usage["completion_tokens"], time.time() - start


def main() -> None:
    port, words = int(sys.argv[1]), int(sys.argv[2])
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    repeats = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    random.seed(time.time())
    for i in range(repeats):
        prompt, completion, seconds = one(port, words, max_tokens)
        print(f"run {i}: prompt {prompt} tokens, {completion} generated, {seconds:.2f} s, {prompt / seconds:.0f} prompt tokens/s overall")


if __name__ == "__main__":
    main()
