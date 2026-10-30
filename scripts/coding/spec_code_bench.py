#!/usr/bin/env python3
"""Measure what the built-in lookup speculation (--speculative ngram, on by default) gains on code-editing requests.

A coding agent mostly asks the model to repeat a file with a change, which is the case lookup speculation is built for:
the model's next words are often a copy of text that is already in the prompt. The script sends such requests, reads the
speculation counters from /metrics before and after, and prints tokens per second and the acceptance ratio. Start the
server twice, once with `--speculative off` and once without the flag, to see the difference.

  python scripts/coding/spec_code_bench.py --url http://127.0.0.1:8000 --runs 4
"""
import argparse
import json
import os
import random
import re
import time
import urllib.request

FILE = '''import json
from dataclasses import dataclass


@dataclass
class Order:
    id: int
    customer: str
    items: list
    paid: bool = False

    def total(self):
        return sum(item["price"] * item["qty"] for item in self.items)


def load_orders(path):
    with open(path) as handle:
        raw = json.load(handle)
    return [Order(o["id"], o["customer"], o["items"], o.get("paid", False)) for o in raw]


def unpaid_total(orders):
    result = 0
    for order in orders:
        if not order.paid:
            result += order.total()
    return result


def by_customer(orders):
    groups = {}
    for order in orders:
        groups.setdefault(order.customer, []).append(order)
    return groups
'''


def get(url, key):
    request = urllib.request.Request(url, headers={"authorization": f"Bearer {key}"} if key else {})
    return urllib.request.urlopen(request, timeout=60).read().decode()


def counters(base, key):
    text = get(base + "/metrics", key)
    out = {}
    for name in ("fllm_spec_proposed_tokens_total", "fllm_spec_accepted_tokens_total", "fllm_spec_steps_total"):
        match = re.search(rf"^{name} ([0-9.e+-]+)", text, re.M)
        out[name] = float(match.group(1)) if match else 0.0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("FLLM_BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--api-key", default=os.environ.get("FLLM_API_KEY"))
    ap.add_argument("--model", default=os.environ.get("FLLM_MODEL"))
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=300)
    args = ap.parse_args()
    base = args.url.rstrip("/")
    tasks = [
        "Add type hints to every function and return the complete file.",
        "Make unpaid_total use sum() and return the complete file.",
        "Add a docstring to every function and return the complete file.",
        "Rename by_customer to group_by_customer and return the complete file.",
    ]
    before = counters(base, args.api_key)
    rows = []
    for i in range(args.runs):
        code = f"# build {random.random():.9f}\n" + FILE  # a fresh first line: no output cache, no cached prefix
        body = {"messages": [{"role": "user", "content": f"{tasks[i % len(tasks)]}\n\n```python\n{code}```"}],
                "max_tokens": args.max_tokens, "temperature": 0.0, "stream": False}
        if args.model:
            body["model"] = args.model
        request = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                         {"content-type": "application/json", **({"authorization": f"Bearer {args.api_key}"} if args.api_key else {})})
        started = time.perf_counter()
        reply = json.load(urllib.request.urlopen(request, timeout=1800))
        elapsed = time.perf_counter() - started
        timings = reply.get("timings", {})
        rows.append((reply["usage"]["prompt_tokens"], reply["usage"]["completion_tokens"], timings.get("predicted_per_second"), elapsed))
        print(f"run {i + 1}: prompt {rows[-1][0]} tokens, wrote {rows[-1][1]} tokens at {rows[-1][2]:.1f} tok/s" if rows[-1][2] else f"run {i + 1}: {rows[-1]}", flush=True)
    after = counters(base, args.api_key)
    proposed = after["fllm_spec_proposed_tokens_total"] - before["fllm_spec_proposed_tokens_total"]
    accepted = after["fllm_spec_accepted_tokens_total"] - before["fllm_spec_accepted_tokens_total"]
    speeds = [r[2] for r in rows if r[2]]
    print(f"decode speed: {sum(speeds) / max(len(speeds), 1):.2f} tok/s mean over {len(speeds)} runs")
    print(f"speculation: {proposed:.0f} draft tokens proposed, {accepted:.0f} accepted"
          + (f" ({100 * accepted / proposed:.0f} %)" if proposed else " (none proposed: speculation is off, or nothing matched)"))


if __name__ == "__main__":
    main()
