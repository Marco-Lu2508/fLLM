# Speculative decoding, sampling and scheduling

This page describes how fLLM speeds up decoding by proposing several tokens per step and verifying them with the target model,
which flags control it, when it helps, and how to measure it. It also covers the device sampler and the scheduler options that
were added with it. Test levels are stated per feature in [changes/spec.md](changes/spec.md).

## What is implemented

| `--speculative` | Draft source | Needs | Runs on |
|---|---|---|---|
| `off` | none | | |
| `ngram` (or a number, the older syntax) | longest earlier match of the conversation's last tokens (prompt lookup) | nothing | every backend |
| `suffix` | longest match across the whole conversation and a cache of recent generations; follows the continuation most matches agree on | nothing | every backend |
| `draft` | a small model of the same tokenizer | `--speculative-model PATH_OR_HF` (or `--draft-model`) | the same device as the target, own KV pool |
| `mtp` | the NextN / MTP head of the checkpoint (DeepSeek-V3 style), one candidate | a checkpoint with the head | CPU or CUDA (`--device cpu`, `auto` or `cuda`; the other backends are refused) |
| `eagle`, `eagle3`, `medusa` | | | not implemented: the flags are rejected with an explanation |

Other flags: `--num-speculative-tokens N` (1 to 32, the largest number of draft tokens per verification; default 5 for `ngram` and
`draft`, 6 for `suffix`), `--speculative-min-match N` and `--speculative-cache-tokens N` (suffix source), `--speculative-window W`
(n-gram index window). `FLLM_SPEC_ADAPTIVE=0` switches the adaptive draft length off (default on: every sequence shrinks and grows
its draft length between 1 and the configured maximum from its recent acceptance rate).

Draft and suffix speculation (and MTP) are refused for stateful models (Qwen3.5 / Qwen3-Next DeltaNet, Mamba, Nemotron-H, LFM2):
verifying rejected draft tokens would advance the recurrent state past the accepted prefix, and the state snapshot/rollback that
would fix this is not implemented. The n-gram source was already off for them.

## Exactness

* **Greedy requests** produce exactly the tokens of plain decoding; a draft token is accepted when it is the target's argmax.
* **Sampled requests** (temperature, top-k, top-p, min-p, typical-p, repetition / presence / frequency penalties, logit bias,
  `allowed_token_ids`, `bad_words`, `min_tokens`) are verified with the speculative sampling rule of Leviathan et al. and Chen et al.
  (accept token `x` with probability `min(1, p(x) / q(x))`, otherwise draw from `norm(max(0, p - q))`, plus a bonus token when all
  are accepted), where `p` is the filtered target distribution of the request at that position. The emitted tokens have the
  distribution of plain sampling. A draft source without a probability (n-gram, suffix, MTP argmax) is a point mass `q`; the draft
  model draws from its own distribution filtered with the request's parameters.
  Seeded requests are reproducible (the same seed gives the same text with the same engine configuration); the text differs from the
  one without speculation because the random numbers are consumed differently.
* **Grammar-constrained requests** speculate: the grammar mask is applied to the target distribution at each position and advances with
  each accepted token.
* **Logprobs** of the emitted tokens come from the target distribution at their position, never from the draft.
* A request opts out with `"speculative": false` in its body.
* Mirostat requests, requests with `n > 1` and requests with image embeddings do not speculate.

## Metrics

`/metrics` exports `fllm_spec_steps_total`, `fllm_spec_proposed_tokens_total`, `fllm_spec_accepted_tokens_total`,
`fllm_spec_acceptance_ratio`, `fllm_spec_mean_accepted_length`, `fllm_spec_mean_tokens_per_step` and `fllm_spec_speedup_estimate`
(tokens per verification step divided by the relative cost of the step; the extra rows of a verification step are charged at 3 %
per draft token and the draft work is measured, so the estimate is rough: use the measurement below).

## When it helps

Decoding is memory bound: a verification step that checks 5 draft tokens costs little more than a step for one token, so every
accepted token is nearly free. The gain is the mean number of tokens per step divided by the extra cost.

* `ngram` and `suffix` cost nothing when they find no match and shine on code editing, summarization, extraction, RAG with quoted
  context and agent loops that repeat tool output. They gain little on free-form chat.
* `draft` helps on every kind of text when the draft model agrees with the target often (same family, 1/10 of the size, for example a
  0.5B-1B model for an 8B-14B target). It costs the draft's memory and a draft step per token: with a draft that is slower than 10 %
  of the target it can lose. Check `fllm_spec_acceptance_ratio` and the measurement below.
* Large batches (many concurrent sequences) already use the GPU well; speculation helps most with 1 to 8 sequences.

## Measuring the speedup (Colab recipe)

Use an 8B model on a T4, L4 or the RTX PRO 6000 Blackwell of a G4 instance, for example a Q4_K_M Llama-3.1-8B-Instruct and a
Llama-3.2-1B-Instruct draft of the same tokenizer (a `llama-3.1-8b` 128k vocabulary and the 1B model share it).

```
# plain
fllm-rs serve --model <8B>.gguf --device cuda --port 8000 &
fllm-rs bench --url http://127.0.0.1:8000 --model <id> --repeats 3      # tokens/s without speculation
# per method (restart the server each time)
fllm-rs serve --model <8B>.gguf --device cuda --speculative ngram --num-speculative-tokens 5 --port 8000
fllm-rs serve --model <8B>.gguf --device cuda --speculative suffix --port 8000
fllm-rs serve --model <8B>.gguf --device cuda --speculative draft --speculative-model <1B>.gguf --num-speculative-tokens 5 --port 8000
fllm-rs bench --url http://127.0.0.1:8000 --model <id> --repeats 3
curl -s localhost:8000/metrics | grep fllm_spec_      # acceptance ratio, mean tokens per step
```

Run each with a repetitive workload (a code-editing or summarization prompt) and with an open chat prompt, once greedy
(`temperature 0`) and once sampled (`temperature 0.7`), and compare tokens/s against the plain run. The acceptance ratio and
`fllm_spec_mean_tokens_per_step` explain the result. The numbers depend on the workload; none are claimed here.

## Device sampling

`FLLM_DEVICE_SAMPLING=1` runs the sampler of decode steps on the CUDA device instead of copying the logits of every sampled row to the
host: logit bias, repetition / presence / frequency penalties, the `min_tokens`, `allowed_token_ids` and `bad_words` controls,
temperature, top-k, min-p, typical-p, top-p, a seeded draw and the log-probabilities (see `cuda/sampling.cu`). The draw is a
Gumbel-max over the surviving tokens with Philox-4x32-10 noise keyed by the request seed, the draw number and the token id, so it
depends on nothing but `(seed, draw number, logits)`. The host sampler uses the same generator for its uniform numbers
(`FLLM_SAMPLER_RNG=xorshift` restores the old generator). The device and the host sampler are separate implementations: they select
the same *distribution* (tested statistically), and `sampling::device_sample_reference` is the host function that reproduces the
device's choice token for token. Ties at a top-k, top-p or typical-p threshold are kept on the device. Grammar-constrained rows send
the mask as a bitmask; DRY and Mirostat rows keep sampling on the host. Default: off, until the device kernels have been run on a GPU
(`fllm-rs selftest-gpu --only sampling`).

## Scheduling

| Flag | Default | Meaning |
|---|---|---|
| `--scheduling-policy fcfs\|priority` | `fcfs` | `priority` serves the smaller `priority` value of the request first; a waiting request of better priority preempts the worst running sequence when all slots are taken or the KV pool is full (the preempted sequence is recomputed) |
| `--max-seqs N\|auto` | 16 | `auto`: on a GPU as many sequences as the KV pool holds at 512 tokens each, at most 256 (on the CPU 16) |
| `--kv-watermark-blocks N` | 1 % of the pool (at most 64) | a waiting sequence is admitted only if this many blocks stay free for the running ones |
| `--max-num-partial-prefills N` | 0 (no limit) | at most N sequences prefill in one step |
| `--long-prefill-token-threshold T`, `--max-long-partial-prefills N` | 0, 1 | prompts longer than T count as long; at most N of them prefill per step, shorter prompts interleave |
| `--chunked-prefill-policy prefill-first\|decode-first` | `prefill-first` | `decode-first` keeps one token of each step's budget per running decoder out of the prompt chunks' reach (lower inter-token latency under load) |
| `--preemption-mode recompute\|swap`, `--swap-space GiB` | `recompute`, 0 | `swap` copies the KV blocks of a preempted sequence to host memory and back instead of recomputing the sequence; it needs KV block operations on the backend (checked at start-up) |

`n > 1` and `best_of` (the `n` candidates with the highest cumulative log-probability of `best_of` samples; not with `stream`) fork the sequence after the
prompt by copying the tail KV block; on an accelerator the block copy is verified on the device at start-up.

Request fields: `priority` (integer, smaller first), `timeout` (seconds; the request is aborted with an error when it has not finished by
then, queue time included). `/metrics`: `fllm_queue_wait_seconds`, `fllm_preemptions_total`, `fllm_request_timeouts_total`, and the
existing running/waiting gauges. `FLLM_CUDA_GRAPH_MAX_ROWS` (default 64, at most 256) raises the largest batch the CUDA graphs
cover.

## Not implemented

EAGLE, EAGLE-3 and Medusa heads, tree verification of the draft (the verification, `speculative::verify_tree`, is implemented and
tested, but the decoder has no tree attention mask), recurrent-state snapshots for hybrid models (so no speculation and no `n > 1` for them),
the stateful prefix cache, prompt log-probabilities / `echo` log-probabilities, beam search, host-memory prefix-cache tiers.
See [changes/spec.md](changes/spec.md) for the full list.
