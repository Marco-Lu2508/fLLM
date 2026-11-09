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

## Models with recurrent layers

**The suffix cache survives a restart.** With `FLLM_SUFFIX_DIR=<dir>` the finished generations are written to `<dir>/suffix-<model digest>.bin`
(the digest of the model file, so the token ids of one tokenizer are never read by another model) and read back at the next start. The
namespace (`cache_salt`) of every generation is kept, so a restart does not let one caller's text draft for another. The index is rebuilt
from the tokens, not stored. Measured on Qwen2.5-0.5B Q8_0 with the same question asked in two processes: the first drafted 18 tokens and
had 7 accepted (2.17 tokens per verification step, 3.00 s); the second, which restored the first one's answer, drafted 67 and had all 67
accepted (6.58 tokens per step, 1.63 s); a third process without a directory behaved like the first (2.17, 2.88 s). The text was the same in
all three. That is the best case (an identical question); a question that only resembles the first gains less, and that was not measured.
A hard kill loses the generations since the last write; the answers stay correct, the cache is only faster.

Prompt lookup (`ngram`), conversation lookup (`suffix`) and a draft model work with models that keep a recurrent state besides (or
instead of) a KV cache: Gated DeltaNet (Qwen3.5, Qwen3-Next, Qwen3.8-Flash-Next), Mamba 1 and 2 (also inside Jamba, Falcon-H1,
Nemotron-H, Granite-hybrid), Zamba2, LFM2 short convolutions, Kimi Delta Attention and RWKV 6 / 7. A rejected draft token has already
been folded into such a state, and unlike a KV cache it cannot be cut back to a shorter length. So the engine copies the state of the
sequence before every verification pass (`RustDecoder::snapshot_state`); when draft tokens are rejected it puts the copy back
(`restore_state`) and runs the verified token and the accepted draft tokens through the layers once more. The sequence ends up exactly
where plain decoding would have it. A pass whose draft is accepted completely needs no restore. For Qwen3.8-Flash-Next the copy also
holds the n-gram ids and convolution histories and the token-indexer keys, which live next to the layer states.

The alternative, a checkpoint of the state after every token of the verification pass, would avoid the second run, but it needs a
state copy per draft token inside every kernel of every backend. The copy before the pass costs one state-sized copy per verified
sequence and per pass plus a short second run only after a rejection. It is the same code for every layer kind and every device.

A draft model with recurrent layers keeps its own copy of the state taken before its draft tokens and goes back to it, or starts over
when the history changed earlier.

What this costs: one state copy per verification pass, on the processor a memory copy, on a graphics card a download of the state
buffer with a synchronisation. For a big hybrid model the state is tens of megabytes, so lookup speculation can be slower than plain
decoding when the drafts are rarely right. Because that has not been measured on real models, conversation lookup stays **off by
default** for models with recurrent layers (`--speculative suffix` turns it on), and the Studio offers all choices except
multi-token prediction.

Not available for these models: **multi-token prediction** (`mtp`), whose heads are only implemented for DeepSeek and GLM
checkpoints, and `n > 1` / forked sequences.

What has been run: greedy output with scripted drafts that are wrong at different positions is token-for-token the output of plain
decoding on small random models of Gated DeltaNet + attention (the Qwen3.5 layout), RWKV 6, RWKV 7, Mamba 2 + attention (Falcon-H1),
Mamba 1 + MoE (Jamba) and Zamba2, on the processor; and with a draft model that is a Qwen3.5-layout model itself
(`rust/engine/src/spec_state_tests.rs`). A small random Qwen3.8-Flash-Next model (two residual streams, three delta layers, an n-gram
layer, an attention layer whose token indexer has more blocks than it selects, a MoE in every layer) is run the same way, in F32 and
as Q4_K_M (`rust/engine/src/spec_exact_tests.rs`). Not run: any real checkpoint, a graphics card (the device state buffers are copied
by the same call, but only the RWKV copy has a test, on the CPU backend standing in for a device), Kimi Delta Attention, DeepSeek-V4,
Inkling and MiniMax-M3.

## Exactness

* **Greedy requests** produce the tokens of plain decoding; a draft token is accepted when it is the target's argmax. How exact that
  is depends on the kernels, see [Greedy output and rounding](#greedy-output-and-rounding) below.
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

### Greedy output and rounding

A report of a Qwen3.8-Flash-Next `IQ3_XXS` file on the CPU gave 189 tokens with `--speculative suffix` and 176 without it for the same
greedy prompt. The state copy and rollback were not the cause (see the tests below). The cause was rounding. A verification pass runs
the target on several rows at once, and the CPU kernels that are fast for several rows add the terms of a dot product in another
order than the kernels of a single-token step: the 16-row float panels (F32, F16 and BF16 weights: the router, the gates of the
hyper-connections, small tensors that stay unquantized in every file), the predecoded and eight-row panels of Q2_K, Q3_K, Q4_K, Q5_K
and Q6_K (the 4-bit to 6-bit formats of nearly every mixed quantization), the batch-one AVX2 kernel of Q4_K to Q6_K, and the
expert kernels of a batch of tokens. A last-bit difference is harmless in a float model, but the quantized kernels round their input
to 8 bits, and a last-bit difference moves some of those roundings by a whole step. On small random models the logits of a six-row
pass differed from single-token decoding by up to 1.2e-2 (Q8_0), 1.5e-2 (Q4_K_M), 1.8e-2 (Q2_K) and 4.4e-2 (Q4_K_M, Qwen3.8-Flash-Next
layout) of the largest logit, and flipped a greedy token (`NOISE` lines of `spec_exact_tests`). Where two tokens are close, the text
then differs after that token, and the length with it.

What the engine does now: the passes that verify draft tokens, and the re-run of the accepted prefix after a rejection, run with
batch-invariant CPU kernels (`cpu_kernels::BatchInvariant`). With them every row of a pass gets exactly the bits it gets as the only
row of a call: no prompt panels, the batch-one AVX2 kernel per token, the experts of one token at a time. Plain decoding is unchanged;
it uses the single-token kernels already. `FLLM_SPEC_EXACT=0` switches this off (the prompt kernels are used, the text is then equal up
to rounding only), `FLLM_BATCH_INVARIANT=1` keeps the invariant kernels for every call (a request then gives the same bits whatever
else the server runs beside it, at the price of slower prompts).

Measured on a 10-thread AVX2 CPU (no AVX-VNNI, no AVX-512), tiny random models of 256 channels:

| Check | Prompt kernels (before) | Batch-invariant kernels (now) |
|---|---|---|
| Logits of a 6-row pass against single-token steps, 12 models (Qwen3.5 layout F32, Q8_0, Q4_K_M, Q2_K; Qwen3.8-Flash-Next layout F32, Q4_K_M; Llama F32, Q8_0, Q4_K_M; Qwen3 Q8_0; Qwen3-MoE Q8_0; RWKV 7), 47 positions each | bit-identical only for Llama and Qwen3 Q8_0 (47/47) and nearly for Qwen3-MoE Q8_0 (46/47); the other nine models 0 to 17 of 47 | bit-identical at all 47 positions in all 12 models |
| Requests (12 per model, each alone, scripted drafts wrong at every position in turn) whose text changed | Qwen3.8-Flash-Next layout Q4_K_M 4 of 12, Llama Q4_K_M 1 of 12, Qwen3.5 Q2_K / Q4_K_M and Qwen3.8 F32 0 of 12 | 0 of 12 everywhere |
| Projection kernels, token alone against the same token in a batch of 2 to 16 rows (22 formats) | F32, F16, BF16 from 4 rows, Q3_K from 4, Q2_K from 8, Q4_K / Q5_K / Q6_K from 2 differ; Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, the lattice IQ formats (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S), IQ4_NL, IQ4_XS, MXFP4 and Q2_0 already agree | all agree |

Cost of a 7-row verification pass in units of a 1-row step (one 4096-column projection of 320 MB, fastest of 25 runs, a shared build
host, so the spread is about 30 %): Q4_K 2.1 before and 3.5 after, Q6_K 3.2 and 3.4, Q3_K 2.9 and 3.0, Q2_K 3.4 and 3.7, IQ3_XXS 2.5 and
2.6, F32 1.7 and 1.3, BF16 2.2 and 2.1. Only Q4_K on a CPU without AVX-VNNI pays visibly (the batch-one kernel runs per token). On
CPUs with AVX-VNNI or AVX-512 the batch-one and the streaming kernels are the same code, so no cost is expected there; that was not measured.

An F32 verification pass now runs as one dispatch over four-row panels: every token still goes through the batch-one tile kernel, in
the same order of sums, but each loaded weight vector serves all tokens of the call instead of one, so the weights of a row are read
once per eight columns. This needs the four-row AVX2 tile to be the single-token path, so it applies to hosts without a 512-bit float
kernel; with AVX-512 BW (or BF16 dot) the single-token call takes the wide kernel and the pass keeps the row blocks. On an AVX-VNNI host, seven rows of the 320 MB projection: 34.0 -> 25.0 ms (3.86x -> 2.80x of a one-row step).
F16 and BF16 keep the row blocks: their time goes into converting every 16-bit value, so sharing the load bought nothing there
(BF16 40.7 -> 42.7 ms, measured twice). `batched_float_panels_give_each_token_the_single_token_bits` compares the raw `f32` bits of
every token against a single-token call, including a column count that is not a multiple of eight and a row tail, and its three
mutations (fma replaced by mul+add, another reduction lane, fma in the scalar tail) each fail it.

End to end with speculation, Qwen2.5-0.5B-Instruct converted to F32 GGUF (`fllm-rs convert --outtype f32`), CPU, 2 threads,
`--speculative suffix`, exact pass on, three alternating rounds of the old and the new binary over three prompts (plain decoding of
both binaries is identical code, and its spread of 7.8 to 10.2 tokens/s shows the host noise):

| Binary | Plain, tokens/s (median) | Speculative, tokens/s (median) | Speculative / plain |
|---|---:|---:|---:|
| before | 8.3 | 16.0 | 1.86 |
| with the F32 panels | 9.7 | 24.0 | 2.50 |

The greedy text of the two modes was equal in all 18 runs. The median ratio is the figure to quote (the absolute numbers moved with a
plain baseline that drifted by 17 % between the rounds); in the round whose plain baseline matched (9.5 to 9.8 tokens/s for both
binaries), speculative decoding went 18.3/16.9/14.5 -> 23.7/24.9/23.4 tokens/s.

Nothing was measured end to end with speculation on a real model in F16, BF16 or a quantized format; those keep the row blocks.


What this does not prove:

* The check ran on the AVX2 tier only. The AVX-VNNI, AVX-512 (also the `avx512` build feature with its AMX and 16-row prompt GEMMs)
  and NEON kernels are not tested for bit equality; for VNNI and AVX-512 the code reading says the streaming kernels do not depend on
  the batch size below four rows, and the prompt kernels are skipped under the guard, but the figures are not measured. The test
  `projection_kernels_give_a_token_the_same_bits_alone_and_in_a_batch` runs on whatever CPU runs the test suite and fails if a kernel
  breaks the rule there.
* Graphics-card backends (CUDA, HIP, Vulkan, Metal) keep their own verification kernels. Rounding differences between a one-row and
  a multi-row pass on a device are not removed, so on a device the text can differ from plain decoding on a near-tie.
* No real checkpoint was run, in particular not the 76 GB `IQ3_XXS` file of the report. Attention over long contexts (the tests use
  at most 64 positions) and grammar-constrained requests are not covered by a bit-equality test.
* A request that shares the server with others is only equal to its solo run when `FLLM_BATCH_INVARIANT=1` is set: plain
  decoding of several sequences in one pass uses the multi-row kernels too.

The tests are `rust/engine/src/spec_exact_tests.rs` (kernels, 6-row noise, scripted speculation on Qwen3.5, Qwen3.8-Flash-Next, Llama and
Qwen3-MoE layouts in F32, Q8_0, Q4_K_M and Q2_K) and `rust/engine/src/spec_state_tests.rs` (state rollback).

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

`FLLM_DEVICE_SAMPLING` (on by default when the model runs on CUDA, `0` keeps the host sampler, `1` asks for it on any backend) runs the sampler of decode steps on the CUDA device instead of copying the logits of every sampled row to the
host: logit bias, repetition / presence / frequency penalties, the `min_tokens`, `allowed_token_ids` and `bad_words` controls,
temperature, top-k, min-p, typical-p, top-p, a seeded draw and the log-probabilities (see `cuda/sampling.cu`). The draw is a
Gumbel-max over the surviving tokens with Philox-4x32-10 noise keyed by the request seed, the draw number and the token id, so it
depends on nothing but `(seed, draw number, logits)`. The host sampler uses the same generator for its uniform numbers
(`FLLM_SAMPLER_RNG=xorshift` restores the old generator). The device and the host sampler are separate implementations: they select
the same *distribution* (tested statistically), and `sampling::device_sample_reference` is the host function that reproduces the
device's choice token for token. Ties at a top-k, top-p or typical-p threshold are kept on the device. Grammar-constrained rows send
the mask as a bitmask; DRY and Mirostat rows keep sampling on the host. Default: on for CUDA since the device kernels ran on an A100 (`fllm-rs selftest-gpu --only sampling`: 16 checks ok; Qwen2.5-7B bf16, 16 users with
`temperature 0.7`, `top_p 0.95`: 336 -> 786 tokens/s; seeded repeats equal, other seeds differ, log-probabilities sorted, a JSON schema and a stop
sequence honoured, frequency penalty as on the host). Not measured: other cards, ROCm.

## Scheduling

| Flag | Default | Meaning |
|---|---|---|
| `--scheduling-policy fcfs\|priority` | `fcfs` | `priority` serves the smaller `priority` value of the request first; a waiting request of better priority preempts the worst running sequence when all slots are taken or the KV pool is full (the preempted sequence is recomputed) |
| `--max-seqs N\|auto` | 16 | `auto`: on a GPU as many sequences as the KV pool holds at 512 tokens each, at most 256 (on the CPU 16) |
| `--kv-watermark-blocks N` | 1 % of the pool (at most 64) | a waiting sequence is admitted only if this many blocks stay free for the running ones |
| `--max-num-partial-prefills N` | 0 (no limit) | at most N sequences prefill in one step |
| `--long-prefill-token-threshold T`, `--max-long-partial-prefills N` | 0, 1 | prompts longer than T count as long; at most N of them prefill per step, shorter prompts interleave |
| `--chunked-prefill-policy prefill-first|decode-first` | `decode-first` | `decode-first` keeps one token of each step's budget per running decoder out of the prompt chunks' reach: a user whose prompt is done gets the answer streamed while the prompts of others are still read (4 users with 2000-token prompts on Vulkan: first tokens after 9 to 32 s instead of all after 31 s); `prefill-first` reads every waiting prompt first |\|decode-first` | `prefill-first` | `decode-first` keeps one token of each step's budget per running decoder out of the prompt chunks' reach (lower inter-token latency under load) |
| `--multi-step N` (alias `--num-scheduler-steps`) | 1 (off) | up to N decode rounds per scheduler round while the batch is unchanged: no new request waits, nothing finished, no prompt is being read. It saves the scheduler's bookkeeping between rounds, not forward passes (the forward passes stay the same number), at the price that new requests and cancellations are noticed up to N rounds later and a stop string is noticed by the server after the extra tokens exist (they are discarded, never sent). Speculative decoding, grammar-constrained requests, media and beam search always run single steps. Accepted on the CPU backend only; on an accelerator it is ignored with a message because it has not been verified there (`FLLM_MULTI_STEP_UNVERIFIED=1` overrides) |
| `--preemption-mode recompute\|swap`, `--swap-space GiB` | `recompute`, 0 | `swap` copies the KV blocks of a preempted sequence to host memory and back instead of recomputing the sequence; it needs KV block operations on the backend (checked at start-up) |

`n > 1` and `best_of` (the `n` candidates with the highest cumulative log-probability of `best_of` samples; not with `stream`) fork the sequence after the
prompt by copying the tail KV block; on an accelerator the block copy is verified on the device at start-up.

Request fields: `priority` (integer, smaller first), `timeout` (seconds; the request is aborted with an error when it has not finished by
then, queue time included). `/metrics`: `fllm_queue_wait_seconds`, `fllm_preemptions_total`, `fllm_request_timeouts_total`, and the
existing running/waiting gauges. `FLLM_CUDA_GRAPH_MAX_ROWS` (default 64, at most 256) raises the largest batch the CUDA graphs
cover.

## Not implemented

EAGLE, EAGLE-3 and Medusa heads, tree verification of the draft (the verification, `speculative::verify_tree`, is implemented and
tested, but the decoder has no tree attention mask), `n > 1` for models with recurrent layers,
the stateful prefix cache, prompt log-probabilities / `echo` log-probabilities, host-memory prefix-cache tiers.
See [changes/spec.md](changes/spec.md) for the full list.
