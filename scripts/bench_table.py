#!/usr/bin/env python3
"""Turn the JSON lines written by bench_compare.py into Markdown tables, one per model and prompt length.

    python3 scripts/bench_table.py results.jsonl > tables.md

Columns: prompt and decode speed of a single stream, then the aggregate generated tokens per second and the median time to the first token
(TTFT) when 4 and 8 requests arrive at the same moment. "-" = not measured, "err" = the engine failed that combination.
"""
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
models = []
for r in rows:
    key = (r["model"], r["tag"])
    if key not in models:
        models.append(key)
for model, tag in models:
    print(f"\n### {model} ({tag})\n")
    rs = [r for r in rows if (r["model"], r["tag"]) == (model, tag)]
    for words in (60, 380, 1500):
        ptok = None
        for r in rs:
            for g in r.get("grid", []):
                if g["words"] == words and "prompt_tokens" in g:
                    ptok = g["prompt_tokens"]
        print(f"\n~{words} words (~{ptok} prompt tokens)\n")
        print("| engine | prompt tok/s (1) | decode tok/s (1) | TTFT s (1) | agg tok/s (4) | TTFT s (4) | agg tok/s (8) | TTFT s (8) |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rs:
            if "error" in r:
                print(f"| {r['label']} | error: {r['error'][:60]} |||||||")
                continue
            d = {g["n"]: g for g in r["grid"] if g["words"] == words}

            def f(n, k):
                g = d.get(n)
                if not g:
                    return "-"
                if "error" in g:
                    return "err"
                return str(g.get(k, "-"))

            print(f"| {r['label']} | {f(1, 'pp')} | {f(1, 'tg')} | {f(1, 'ttft')} | {f(4, 'agg_tps')} | {f(4, 'ttft')} | {f(8, 'agg_tps')} | {f(8, 'ttft')} |")
