#!/usr/bin/env python3
"""Benchmark client for fLLM and for whatever it is compared with.

It talks plain HTTP with Streaming, so the same numbers come out of fLLM, vLLM, SGLang or llama.cpp — that is the point:
a comparison is only worth something if one client measures all of them the same way.

Measured per request: time to first token (TTFT), time to last token, tokens, tokens/s, and the reported prompt tokens.
Reported: p50 / p95 / p99 and the aggregate throughput at the given concurrency.

Usage:
    python3 scripts/bench.py --target fllm=http://127.0.0.1:8000 --target vllm=http://127.0.0.1:8001 \\
        --model llama --requests 32 --concurrency 4 --max-tokens 128 --prompt-tokens 512

    # a fixed prompt list instead of generated ones
    python3 scripts/bench.py --target fllm=http://127.0.0.1:8000 --prompts benchmarks/prompts.jsonl

Exit code 0 means every request succeeded; 1 means at least one failed (the failure is in the JSON).
"""
import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# A sentence with common words: the tokenizer of any model handles it, and the prompt length is predictable enough.
WORDS = ("the system runs a model on the graphics card and the scheduler decides which request is next "
         "while the cache keeps the blocks of the prompt that were already computed").split()


def generated_prompts(count: int, words: int) -> list:
    """Prompts with different beginnings, so the prefix cache cannot make every request free."""
    out = []
    for index in range(count):
        body = " ".join(WORDS[(index * 7 + step) % len(WORDS)] for step in range(words))
        out.append(f"Request {index}. {body}")
    return out


def load_prompts(path: Path) -> list:
    if path.suffix == ".jsonl":
        return [json.loads(line)["prompt"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [chunk.strip() for chunk in path.read_text(encoding="utf-8").split("\n\n") if chunk.strip()]


def percentile(values: list, fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def one_request(base: str, model: str, prompt: str, max_tokens: int, temperature: float, timeout: float) -> dict:
    """One streaming chat request; returns the numbers or raises."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    request = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    started = time.perf_counter()
    first = None
    tokens = 0
    prompt_tokens = 0
    text = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = event.get("usage") or {}
            if usage.get("prompt_tokens"):
                prompt_tokens = int(usage["prompt_tokens"])
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    if first is None:
                        first = time.perf_counter()
                    tokens += 1
                    text.append(piece)
    ended = time.perf_counter()
    return {
        "ttft_ms": ((first or ended) - started) * 1000.0,
        "total_ms": (ended - started) * 1000.0,
        "completion_tokens": tokens,
        "prompt_tokens": prompt_tokens,
        "text": "".join(text),
    }


def run_target(name: str, base: str, prompts: list, args) -> dict:
    results = []
    errors = []
    lock = threading.Lock()

    def worker(chunk: list):
        for prompt in chunk:
            try:
                result = one_request(base, args.model, prompt, args.max_tokens, args.temperature, args.timeout)
                with lock:
                    results.append(result)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as error:
                with lock:
                    errors.append(str(error))

    started = time.perf_counter()
    threads = []
    size = max(1, len(prompts) // max(1, args.concurrency) + 1)
    for start in range(0, len(prompts), size):
        thread = threading.Thread(target=worker, args=(prompts[start:start + size],))
        thread.start()
        threads.append(thread)
        # keep the concurrency honest instead of starting everything at once
        while sum(1 for t in threads if t.is_alive()) >= args.concurrency:
            time.sleep(0.01)
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - started

    ttft = [r["ttft_ms"] for r in results]
    total = [r["total_ms"] for r in results]
    completion = sum(r["completion_tokens"] for r in results)
    prompt_tokens = sum(r["prompt_tokens"] for r in results)
    return {
        "target": name,
        "base_url": base,
        "requests": len(results),
        "failed": len(errors),
        "errors": errors[:5],
        "wall_seconds": round(wall, 3),
        "ttft_ms": {"p50": round(percentile(ttft, 0.50), 1), "p95": round(percentile(ttft, 0.95), 1), "p99": round(percentile(ttft, 0.99), 1)},
        "total_ms": {"p50": round(percentile(total, 0.50), 1), "p95": round(percentile(total, 0.95), 1), "p99": round(percentile(total, 0.99), 1)},
        "completion_tokens": completion,
        "prompt_tokens": prompt_tokens,
        "tokens_per_second": round(completion / wall, 1) if wall > 0 else 0.0,
        "prompt_tokens_per_second": round(prompt_tokens / wall, 1) if wall > 0 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark fLLM against other OpenAI-compatible servers")
    parser.add_argument("--target", action="append", default=[], help="name=url, repeatable (default fllm=http://127.0.0.1:8000)")
    parser.add_argument("--model", default="", help="model name to send (empty: the server's default)")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt-words", type=int, default=380, help="about 512 tokens at 1.35 tokens per word")
    parser.add_argument("--prompts", type=Path, help="a .jsonl of {prompt} lines or a .txt with paragraphs")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--json", type=Path, help="write the full result here")
    args = parser.parse_args()

    targets = args.target or ["fllm=http://127.0.0.1:8000"]
    prompts = load_prompts(args.prompts) if args.prompts else generated_prompts(args.requests, args.prompt_words)
    if len(prompts) < args.requests:
        prompts = (prompts * ((args.requests // max(1, len(prompts))) + 1))[: args.requests]
    else:
        prompts = prompts[: args.requests]

    print(f"{len(prompts)} requests, concurrency {args.concurrency}, max {args.max_tokens} tokens, temperature {args.temperature}")
    results = []
    for target in targets:
        name, _, base = target.partition("=")
        if not base:
            name, base = "fllm", target
        result = run_target(name or "target", base, prompts, args)
        results.append(result)
        print(
            f"{result['target']:>10}  ttft p50 {result['ttft_ms']['p50']:>8.1f} ms  p99 {result['ttft_ms']['p99']:>9.1f} ms  "
            f"{result['tokens_per_second']:>8.1f} tok/s  {result['completion_tokens']:>6} tokens  failed {result['failed']}"
        )
    if len(results) > 1:
        fastest = max(results, key=lambda r: r["tokens_per_second"])
        print(f"\nfastest: {fastest['target']} with {fastest['tokens_per_second']} tok/s")
    payload = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "results": results}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"written to {args.json}")
    return 0 if all(r["failed"] == 0 for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
