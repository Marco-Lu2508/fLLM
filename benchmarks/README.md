# Benchmarks

Numbers, not claims. This folder holds the client and the results it produces; a result file is only added when the
hardware, the model and the load are written next to it.

## Running

```bash
# two servers side by side, same prompts, same client
python3 scripts/bench.py \
  --target fllm=http://127.0.0.1:8000 \
  --target vllm=http://127.0.0.1:8001 \
  --model llama --requests 64 --concurrency 8 --max-tokens 128 --prompt-words 380 \
  --json benchmarks/results/2026-10-27-a100.json
```

Measured per request: time to first token, time to last token, completion tokens, prompt tokens.
Reported: p50/p95/p99 and the aggregate throughput at the given concurrency. The client speaks plain HTTP with
`stream: true`, so the same numbers come out of fLLM, vLLM, SGLang and llama.cpp.

## What a result file must state

- GPU model and count, CPU, RAM, driver/CUDA version
- model (name, quantization, context), server version and flags of **every** target
- request count, concurrency, max tokens, temperature, and whether the prompts are shared or unique
- the raw JSON, not a summary

Without those, a number is not a benchmark, it is advertising.

## Prompts

`--prompts benchmarks/prompts.jsonl` (one `{"prompt": "..."}` per line) for a fixed set that can be compared over time.
Without it the client generates prompts that differ in their beginning, so the prefix cache cannot make the measurement
free — that difference matters: with a shared prefix, fLLM and every engine with a cache will look several times faster,
and the number then measures the cache, not the engine.

## A worked example (the client, not a performance claim)

```bash
fllm-rs make-tiny-model /tmp/tiny.gguf
fllm-rs --model /tmp/tiny.gguf --model-name tiny --device cpu --port 8099 &
python3 scripts/bench.py --target fllm=http://127.0.0.1:8099 --model tiny \
    --requests 8 --concurrency 2 --max-tokens 16 --prompt-words 20 --json bench-tiny.json
```

```
8 requests, concurrency 2, max 16 tokens, temperature 0.0
      fllm  ttft p50    127.9 ms  p99     291.7 ms      99.9 tok/s     128 tokens  failed 0
```

Those numbers describe a random tiny model on a CPU and say nothing about how fast fLLM is with a real model. What they do
show is that the client speaks the same streaming protocol a real run uses, measures TTFT from the first byte of the stream,
and writes a JSON file. Replace the target with your model and the numbers become meaningful.
