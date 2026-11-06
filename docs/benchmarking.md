# Benchmarking

How the speed figures in the README and in [gpu-backends.md](gpu-backends.md) are measured, and how to measure your own. Speed is
level L3 of [test-levels.md](test-levels.md): it only means something with the method written down next to the number.

## What is measured

* **Prefill speed**: prompt tokens per second for a prompt of a given length, from sending the request until the first generated token.
  Measure several prompt lengths; short prompts are dominated by fixed costs and long prompts by attention and memory.
* **Decode speed**: generated tokens per second after the first token, for one user (latency-bound) and for many concurrent users
  (throughput).
* **Time to first token** and the **inter-token interval** as seen by the client.
* **Quality guards**: greedy output identical to the baseline variant, or a stated log-probability difference. A faster variant that
  changes the text is a different result, not a speed-up.

All of this is measured over HTTP against a running `fllm-rs serve`, with a streaming client, because that is what users get.
`rust/engine/examples/perf_decode.rs` measures decoding without the server for kernel work.

## The benchmark client

`fllm-rs bench` is an HTTP client for any OpenAI-compatible server (fLLM, vLLM, llama.cpp, SGLang, Ollama), so one tool and one
workload compare several engines. `fllm-rs bench --help` lists the options; the main ones select the scenarios (prefill, single-user
decode, decode at long context, concurrency), prompt lengths, concurrency levels, generation length, warm-up and repeats, and write a JSON
result with `--json`. `fllm-rs bench-report` compares saved results. The definitions, the JSON schema and the fair-comparison checklist
are in [bench/README.md](../bench/README.md); read the checklist before publishing any comparison.

```sh
fllm-rs serve --model MODEL.gguf --device cuda --port 8000 &
fllm-rs bench --url http://127.0.0.1:8000 --scenarios prefill,decode-single,concurrency \
    --prompt-lens 355,1863 --concurrency 1,4,16 --gen-len 128 --repeats 3 --label "my card" --json result.json
```

## A/B matrix for the CUDA switches

The new CUDA defaults (graphs, fused kernels, NVFP4 activation scale and so on) each have an environment switch back to the old
behaviour ([configuration.md](configuration.md)). `scripts/colab/ab_matrix.py` runs a list of variants against each other on one GPU:
it detects the card, builds fLLM (or reuses a build), runs the self test, then for each model starts one server per variant with a
different set of `FLLM_*` variables, measures prefill (several prompt lengths) and decode with the same client, checks that the text
is identical to the baseline variant, optionally measures vLLM on the same card, and prints a JSON summary. It needs only the Python
standard library and a GPU runtime, and was written for Google Colab, where the README tables were produced. Start with the quick mode
(three variants) and read `--help` for the rest:

```sh
python3 -u scripts/colab/ab_matrix.py --quick
python3 -u scripts/colab/ab_matrix.py --only-model awq --only-variants V0,V8
python3 -u scripts/colab/ab_matrix.py --dry-run      # prints the variants and the server commands, starts nothing
```

`scripts/colab/cpu_confirm.py` repeats the CPU-side checks of the same changes on a CPU machine, and `scripts/colab/nvfp4_quality.py`
scores NVFP4 variants by perplexity against a reference (including vLLM).

## Method rules used for the published tables

* Greedy decoding (temperature 0), one user unless the column says otherwise, the same client for every engine, localhost HTTP.
* One warm-up request, then repeats; report the median and keep the spread. Three repeats show large differences, not small ones.
* Prompt lengths are tokens of the model's own tokenizer; say which. Disable prefix caching or use unique prefixes for prefill
  measurements, because a prefix-cache hit measures the cache.
* Record the version (commit), build features, server flags and environment variables, driver and runtime versions, power limit and
  clocks of the GPU, and whether other work ran on the machine. Shared cloud machines (Colab) vary; treat differences of a few percent
  as noise.
* Compare with another engine only on the same card in the same session, with the same weights or the nearest equivalent quantization,
  and say where the weights differ (for example GGUF Q4_K_M versus AWQ is not like for like).
* Do not compare numbers from different clients or machines. Earlier published comparisons were corrected for exactly that reason.

## Caveats

* GPU power and thermal limits decide sustained prefill speed; isolated kernel peaks are higher than what a whole prompt reaches.
* Decode speed at batch size one is bound by memory bandwidth; compare it with the card's bandwidth before you look for kernel gains.
* Speculative decoding, prompt lookup and the CPU/GPU split change the cadence of output; benchmark them separately and label them.
* A benchmark result states what happened on that machine. It is not a promise for other hardware.

## Reporting

Include the JSON file from `--json` (or the matrix output), the exact commands and the information above in a hardware report issue
([CONTRIBUTING.md](../CONTRIBUTING.md)). The `streaming-benchmark` workflow can run the client on a self-hosted runner ([ci.md](ci.md)).

## Agent load: `fllm-rs bench-agent`

`bench-serve` sends independent requests. A chat front end or a coding agent sends *conversations*: one long system prompt (the tool list)
that is the same for all users, and in every turn the whole history again, the earlier answers included. What a server can reuse between the
turns decides how long a turn takes, and a benchmark of single prompts cannot show it. `bench-agent` measures every turn on its own, against
any OpenAI-compatible server, so that fLLM, vLLM and SGLang can be run with the same command:

```sh
fllm-rs bench-agent --url http://HOST:8000 --users 32 --turns 8 --system-words 3000 --answer-tokens 128 \
    --passes 2 --label "fLLM mi300x tp8" --json agent-fllm.json --markdown agent-fllm.md
```

Read the table by turn and by pass. Turn 1 of the cold pass has only the shared system prompt to reuse; turn 2 and later reuse the history;
the warm pass repeats the same conversations and finds what the cold pass left in the caches.

For a comparison that holds up:

* **Answers.** `--answers fixed` (default) puts a text made from the seed into the history, the same for every server, so all of them see
  byte-identical prompts and a difference in the time of a turn is the server's. `--answers server` sends back what the server really wrote;
  then two servers see different prompts as soon as their answers differ, and the header of the run says so.
* **Caches on.** Start vLLM with prefix caching (`--enable-prefix-caching`; on by default in recent versions) and SGLang with its radix cache
  (default). A comparison against a server whose cache is off measures the cache, not the server. `cached` shows what each server reports;
  a server that does not report it shows 0, which means unknown there.
* **Same settings.** Same model and quantisation, same `--users`, `--turns`, `--seed`, the same context length and max-batch settings.
  Pass `--seed` unchanged between servers; a different seed is a different set of texts.
* **Cold means cold.** Restart the server before the cold pass. The warm pass is the second one of the same run.
* **Output length.** Every turn asks for `--answer-tokens` tokens and ignores the end-of-sequence token (`min_tokens`, `ignore_eos`), so the
  work per turn is the same on every server. A server that ignores these fields makes shorter turns; compare `e2e` with care then.

Measured on this repository's CPU host (Qwen2.5-0.5B Q8_0, 3 users x 4 turns, 800-word system prompt, 24 tokens): turn 1 cold 11.4 s TTFT with
38 % cached (the other users' system prompt), turn 2 1.8 s with 89 % cached, turns 3 and 4 1.3 s; the warm pass 0.3-0.5 s with 99-100 %
cached. That shows the tool reads the cache effect; it says nothing about a GPU server or about another engine, and no number from vLLM or
SGLang has been measured here.

