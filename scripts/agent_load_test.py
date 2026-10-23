#!/usr/bin/env python3
"""Fire N parallel chat requests that share a long prefix and report time and tokens/s.

Mimics sub-agents: the same system prompt and tool list, a different task at the end.
Standard library only.

  python3 scripts/agent_load_test.py --url http://127.0.0.1:8000 -n 8 --prefix-words 1500
  python3 scripts/agent_load_test.py -n 8 --warm          # send one request first so the prefix is cached
  python3 scripts/agent_load_test.py -n 8 --tool-grammar  # also set tool_call_grammar
"""
import argparse
import json
import threading
import time
import urllib.request

TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_shell",
        "description": "Run a shell command and return its output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}]


def request(url, body, key):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), headers)
    start = time.time()
    with urllib.request.urlopen(req, timeout=3600) as response:
        data = json.load(response)
    return time.time() - start, data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--key", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("-n", type=int, default=4, help="parallel requests")
    ap.add_argument("--prefix-words", type=int, default=800, help="length of the shared system prompt")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--warm", action="store_true", help="send one request first, then the parallel ones")
    ap.add_argument("--tool-grammar", action="store_true", help="set tool_call_grammar")
    args = ap.parse_args()

    system = "You are a coding agent. Follow the rules below.\n" + " ".join(
        "rule%d" % i for i in range(args.prefix_words))

    def body(task):
        out = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": task}],
            "tools": TOOLS,
            "max_tokens": args.max_tokens,
            "temperature": 0,
        }
        if args.model:
            out["model"] = args.model
        if args.tool_grammar:
            out["tool_call_grammar"] = True
        return out

    if args.warm:
        seconds, data = request(args.url, body("warm-up"), args.key)
        print("warm-up: %.2fs, prompt %d tokens" % (seconds, data["usage"]["prompt_tokens"]))

    results = [None] * args.n

    def worker(i):
        try:
            results[i] = request(args.url, body("task number %d: list the files" % i), args.key)
        except Exception as error:  # report, do not hide
            results[i] = error

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.n)]
    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - start

    generated = cached = prompt = failed = 0
    for r in results:
        if isinstance(r, Exception) or r is None:
            failed += 1
            print("failed:", r)
            continue
        usage = r[1]["usage"]
        generated += usage["completion_tokens"]
        prompt += usage["prompt_tokens"]
        cached += usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    print("requests %d (failed %d), wall time %.2fs" % (args.n, failed, wall))
    print("prompt tokens %d, reused from the prefix cache %d" % (prompt, cached))
    print("generated %d tokens, %.1f tokens/s overall" % (generated, generated / wall if wall else 0))


if __name__ == "__main__":
    main()
