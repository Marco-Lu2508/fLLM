#!/usr/bin/env python3
"""Replay a multi-turn coding-agent session against an fLLM (or any OpenAI-compatible) server and report, per turn,
the prompt size, how much of it came from the prefix cache, the time to the first token and the decode speed.

A coding agent resends the whole conversation on every turn. A server that keeps the key/value cache of the shared
prefix only has to read the new tool result, so the time to first token stays flat as the session grows; one that reads
everything again gets slower with every turn. This script shows which one you have.

  python scripts/coding/replay_session.py --url http://127.0.0.1:8000 --turns 12
  python scripts/coding/replay_session.py --session recorded.json --cold        # same session, cache defeated
  python scripts/coding/replay_session.py --turns 12 --json out.json

--session FILE is {"system": str, "tools": [...], "messages": [...]} in the OpenAI chat format; every assistant
message of it is one turn: the history before it is sent and the server writes the next message (the recorded one is
not used as the answer, it only serves as history for the following turns). Without --session a session is
generated: a system prompt, a tool list and N rounds of "assistant calls read_file / tool returns a source file".

--cold puts a fresh random line in front of the system prompt of every request, so no prefix can be reused: the
difference between a normal and a cold run is what the prefix cache is worth.
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

TOOLS = [
    {"type": "function", "function": {"name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": list(props)}}}
    for name, desc, props in [
        ("read_file", "Read a file from the project and return its text.", {"path": {"type": "string"}}),
        ("write_file", "Create or overwrite a file.", {"path": {"type": "string"}, "content": {"type": "string"}}),
        ("edit_file", "Replace one exact piece of text in a file.", {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}),
        ("list_dir", "List the entries of a folder.", {"path": {"type": "string"}}),
        ("grep", "Search the project for a regular expression.", {"pattern": {"type": "string"}}),
        ("run_command", "Run a shell command and return its output.", {"command": {"type": "string"}}),
    ]
]
SYSTEM = (
    "You are a careful coding agent working in the user's repository. Inspect files with the tools before you change "
    "them, keep edits small, run the tests after a change and report what you did in two sentences. " * 6
)


def source_file(rng, index, funcs):
    """A plausible source file (about 90 tokens per function), different for every index."""
    lines = [f"// module {index}: helpers for the request pipeline"]
    for n in range(funcs):
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        lines.append(f"pub fn step_{index}_{n}(input: &[u32], limit: usize) -> Result<Vec<u32>, String> {{")
        lines.append(f"    let mut out = Vec::with_capacity(limit.min({a}));")
        lines.append(f"    for (i, v) in input.iter().enumerate() {{ if i >= limit {{ break; }} out.push(v * {b} + {n}); }}")
        lines.append("    if out.is_empty() { return Err(String::from(\"nothing to do\")); }")
        lines.append("    Ok(out)\n}")
    return "\n".join(lines)


def synthetic(turns, seed, funcs):
    rng = random.Random(seed)
    messages = [{"role": "user", "content": "Find out how the request pipeline is built and tell me where a new step would go."}]
    for i in range(turns):
        call_id = f"call_{i}"
        path = f"src/pipeline/part_{i}.rs"
        messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": path})}}]})
        messages.append({"role": "tool", "tool_call_id": call_id, "content": source_file(rng, i, funcs)})
    return {"system": SYSTEM, "tools": TOOLS, "messages": messages}


def post_stream(url, key, body, timeout):
    request = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json", **({"authorization": f"Bearer {key}"} if key else {})},
    )
    started = time.perf_counter()
    first = None
    usage = {}
    text = ""
    last = started
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece or delta.get("tool_calls"):
                    if first is None:
                        first = time.perf_counter()
                    last = time.perf_counter()
                    text += piece
    end = time.perf_counter()
    return {"ttft": (first or end) - started, "decode_s": max(last - (first or last), 1e-9), "total": end - started, "usage": usage, "text": text}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("FLLM_BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--api-key", default=os.environ.get("FLLM_API_KEY"))
    ap.add_argument("--model", default=os.environ.get("FLLM_MODEL"))
    ap.add_argument("--session", help="recorded session (JSON); default: a generated one")
    ap.add_argument("--turns", type=int, default=12, help="turns of the generated session")
    ap.add_argument("--max-tokens", type=int, default=24, help="tokens written per turn")
    ap.add_argument("--cold", action="store_true", help="defeat the prefix cache with a fresh first line per request")
    ap.add_argument("--file-funcs", type=int, default=5, help="functions per generated tool result (about 90 tokens each)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--json", help="write the numbers to this file")
    args = ap.parse_args()

    session = json.load(open(args.session, encoding="utf-8")) if args.session else synthetic(args.turns, args.seed, args.file_funcs)
    base = args.url.rstrip("/").removesuffix("/v1")
    messages = session["messages"]
    # a turn per assistant message: the history before it
    cuts = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if not cuts:
        sys.exit("the session has no assistant message to replay")
    rows = []
    nonce = random.Random(os.getpid() ^ int(time.time())).random
    print(f"{'turn':>4} {'prompt':>7} {'cached':>7} {'ttft_s':>8} {'tok/s':>7} {'total_s':>8}")
    for turn, cut in enumerate(cuts, 1):
        system = session["system"]
        if args.cold:
            system = f"[run {nonce():.12f}]\n" + system
        body = {
            "messages": [{"role": "system", "content": system}] + messages[:cut],
            "tools": session.get("tools") or None,
            "max_tokens": args.max_tokens,
            "temperature": 0.2,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if args.model:
            body["model"] = args.model
        body = {k: v for k, v in body.items() if v is not None}
        try:
            result = post_stream(base, args.api_key, body, args.timeout)
        except urllib.error.HTTPError as error:
            sys.exit(f"turn {turn}: HTTP {error.code}: {error.read().decode('utf-8', 'replace')[:300]}")
        usage = result["usage"]
        prompt = usage.get("prompt_tokens", 0)
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        out = usage.get("completion_tokens", 0)
        speed = (out - 1) / result["decode_s"] if out > 1 else 0.0
        rows.append({"turn": turn, "prompt_tokens": prompt, "cached_tokens": cached, "ttft_s": round(result["ttft"], 3), "decode_tok_s": round(speed, 2), "total_s": round(result["total"], 3)})
        print(f"{turn:>4} {prompt:>7} {cached:>7} {result['ttft']:>8.2f} {speed:>7.2f} {result['total']:>8.2f}", flush=True)
    if len(rows) >= 10:
        print(f"\nturn 10: prompt {rows[9]['prompt_tokens']} tokens, {rows[9]['cached_tokens']} from the cache, first token after {rows[9]['ttft_s']} s (turn 1: {rows[0]['ttft_s']} s)")
    if args.json:
        json.dump({"cold": args.cold, "rows": rows}, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
