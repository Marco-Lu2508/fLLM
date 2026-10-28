# Changes of the speculative decoding / sampling / scheduler work

User guide: [../speculative-decoding.md](../speculative-decoding.md). Test levels used below: L0 = compiles (and for CUDA: `nvcc` for the
architectures named), L1 = `selftest-gpu` ran on the device, L2 = text matches the CPU reference, L3 = benchmarked. **Nothing in this
change was run on a GPU**: every CUDA statement is L0. CPU statements say `cargo test`.

## 1. Features

### 1.1 Speculative decoding for sampled, constrained and logprob requests

`--speculative off|ngram|draft|mtp|suffix` (a number still selects the n-gram length), `--speculative-model PATH_OR_HF`,
`--num-speculative-tokens N`, `--speculative-min-match N`, `--speculative-cache-tokens N`, per-request `"speculative": false`.

* Verification is the speculative sampling rule (`speculative::verify_chain`): accept `x` with probability `min(1, p(x)/q(x))`, otherwise
  resample from `norm(max(0, p - q))`, bonus token when everything is accepted; `p` and `q` are the request's filtered distributions
  (`sampling::processed_distribution`). Greedy requests reduce to "accept iff argmax" and consume no random numbers. A tree version
  (`verify_tree`, SpecInfer's sequential rule for point-mass siblings) exists and is tested, but the decoder cannot run a token tree.
* Draft sources: n-gram (scheduler), suffix match across the conversation and a cache of recent generations (`speculative::SuffixCache`),
  draft model (the draft now draws from its filtered distribution and returns `q`; it is loaded on the same device by `attach_compute`, serve.rs
  takes its file size out of the GPU budget), MTP (unchanged, CPU only). EAGLE / EAGLE-3 / Medusa: rejected with an explanation.
* Adaptive draft length per sequence (`speculative::AdaptiveDraft`), `FLLM_SPEC_ADAPTIVE=0` disables it.
* Grammar masks, `min_tokens`, `allowed_token_ids`, `bad_words`, penalties and bias are applied to the target distribution at every verified
  position; the grammar advances with each accepted token. Log-probabilities come from the target distribution of the emitted position.
* Draft, suffix and MTP speculation are refused for stateful models (the recurrent state would run past the accepted prefix).
* Metrics: `fllm_spec_*` (see the user guide).

### 1.2 Sampling

* Counter-based Philox-4x32-10 generator: draw `n` of a sequence is a pure function of `(seed, n)`. Default for the host sampler;
  `FLLM_SAMPLER_RNG=xorshift` restores the previous generator.
* `sampling::processed_distribution` exposes the exact distribution a request samples from; top-k now selects instead of sorting the vocabulary.
* New controls: `allowed_token_ids`, `bad_words` (strings, blocked at the start of a text and after a space; multi-token words block their last token
  once the prefix was generated), `logprobs_mode` (`raw_logprobs` default, `processed_logprobs`), `priority`, `timeout`, `speculative`, `best_of`
  (non-streaming; the `n` candidates with the highest cumulative log-probability are returned, usage counts all candidates).
  (`use_beam_search` was rejected here at the time; it is implemented now, see `docs/api-compat.md`.)
* CUDA sampler `cuda/sampling.cu` + `Stream::sample_rows` + capability `sample_on_device` (true on CUDA and the CPU backend): bias, penalties, grammar bitmask,
  temperature, top-k, min-p, typical-p, top-p, Gumbel-max draw with Philox noise, raw or processed log-probabilities with top-N alternatives. Host reference
  `sampling::device_sample_reference` (sorting instead of bisection). `FLLM_DEVICE_SAMPLING=1` routes decode steps through it (default off).
  Not on the device: DRY, Mirostat, grammar-constrained rows (they stay on the host), logit scale / soft-cap models.
  `selftest-gpu --only sampling` has five new checks: controls (exact), filters (exact up to boundary rows), log-probabilities, batch independence, distribution.

### 1.3 Scheduler

`--scheduling-policy fcfs|priority`, `--kv-watermark-blocks N`, `--max-num-partial-prefills N`, `--long-prefill-token-threshold T`,
`--max-long-partial-prefills N`, `--chunked-prefill-policy prefill-first|decode-first`, `--max-seqs auto`, `--preemption-mode recompute|swap`,
`--swap-space GiB`, request fields `priority` and `timeout`. Queue wait / preemption / timeout metrics. `FLLM_CUDA_GRAPH_MAX_ROWS` (default 64, at most 256)
raises the CUDA graph buckets to 128 / 256 rows. Also fixes a scheduler error after cancelling a request that was still waiting.

### 1.4 KV block operations

`kv_blocks.rs`: copy, export and import of a whole KV block for every KV format, on the host arrays and (through slices of the pool buffers and
`Stream::copy_buffer` / `upload` / `download`) on device pools. Used by

* preemption by swap (blocks of the preempted sequence leave for host memory, the sequence resumes without recomputation),
* `n > 1` / `best_of` on accelerator backends: at start-up the copy is verified on the device itself (copy block 0 to 1, compare exports); a failure
  keeps `n > 1` off with a message.

## 2. Environment variables and flags

| Name | Default | Meaning |
|---|---|---|
| `--speculative METHOD` | off | `off`, `ngram`, `draft`, `mtp`, `suffix`, or a number (n-gram length) |
| `--speculative-model X` | | draft model (file, `hf://` reference, directory); same as `--draft-model` |
| `--num-speculative-tokens N` | 5 (suffix 6) | largest draft length |
| `--speculative-min-match N`, `--speculative-cache-tokens N` | 3, 1048576 | suffix source |
| `--speculative-tree` | | only `off` accepted (no tree attention yet) |
| `FLLM_SPEC_ADAPTIVE` | on | `0` fixes the draft length |
| `FLLM_SAMPLER_RNG` | `philox` | `xorshift` = previous generator |
| `FLLM_DEVICE_SAMPLING` | off | `1` samples decode steps on the CUDA device |
| `--scheduling-policy` | `fcfs` | `priority`: smaller `priority` first, with preemption |
| `--max-seqs` | 16 | a number or `auto` (GPU: pool tokens / 512, at most 256; CPU 16) |
| `--kv-watermark-blocks` | pool / 100, at most 64 | blocks kept free at admission |
| `--max-num-partial-prefills`, `--long-prefill-token-threshold`, `--max-long-partial-prefills` | 0, 0, 1 | prefill limits |
| `--chunked-prefill-policy` | `prefill-first` | `decode-first` reserves one budget token per decoder |
| `--preemption-mode`, `--swap-space` | `recompute`, 0 | swap needs GiB of host memory and KV block operations |
| `FLLM_CUDA_GRAPH_MAX_ROWS` | 64 | graph buckets up to 256 |
| request: `speculative`, `priority`, `timeout`, `allowed_token_ids`, `bad_words`, `logprobs_mode`, `best_of` | | see above |

## 3. THIRD_PARTY entries

* Philox-4x32-10 (Salmon, Moraes, Dror, Shaw: "Parallel random numbers: as easy as 1, 2, 3", SC11; Random123, D. E. Shaw Research, BSD-3-Clause): the
  round function and constants, re-implemented in `rust/engine/src/sampling.rs` and `rust/engine/cuda/sampling.cu`; known-answer tests from the Random123 vectors.
* vLLM v1 sampler (Apache-2.0): the order of operations (bias, penalties, temperature, top-k / min-p / top-p, exponential-race draw), the tie rule of
  the top-k threshold and the `priority` convention. Design reference only; no code copied.
* Leviathan, Kalman, Matias (ICML 2023) and Chen et al. (2023): speculative sampling; Miao et al. (SpecInfer, 2024): the multi-candidate rule used by `verify_tree`;
  Oliaro et al. (SuffixDecoding, 2024): longest suffix match across requests. Algorithms only; no code copied.

## 4. Test levels

CPU (`cargo test --release --locked -p fllm-engine -p fllm-checkpoint -p fllm-gguf-probe`, see REPORT.md for the counts):

* statistical (chi-square, 100k samples on a toy model) equality of the speculative output with the target distribution for sampling, top-k, top-p, min-p, point-mass drafts and trees; a control that the test detects a wrong verifier;
* exact equality of greedy output with n-gram, suffix and draft-model speculation on a tiny model, including logprobs; stochastic speculation reproducible per seed, acceptance rate 1 for a draft equal to the target;
* grammar-constrained speculation equals plain constrained decoding; `min_tokens`, `allowed_token_ids`, `bad_words` equal with and without speculation; opt-out;
* scheduler: priority, preemption, watermark, decode-first, prefill limits, deadlines; swap and recompute preemption reproduce the tokens of an unpressured run;
* KV blocks: copy / export / import for all formats on host arrays and through the Stream API on the CPU backend;
* sampler reference: Philox known answers, flatness, equality with `processed_distribution` statistically for six filter combinations, log-probabilities;
* `selftest-gpu` on the CPU backend runs the new sampling checks.

CUDA: L0 only. `cargo clippy --features cuda` (`FLLM_CUDA_ARCHS=75`) builds `sampling.cu`, the C API and the Rust glue; `nvcc` compiles `sampling.cu` for sm_75, 80, 89, 90 and 120.
The kernel logic was additionally executed on the CPU with a thread-per-lane emulation (shim outside the repository: one `std::thread` per CUDA thread, barrier for
`__syncthreads`, exchange for `__shfl_xor_sync`) on 360 generated cases (vocabularies 50, 1000, 4099; all filters, penalties, bias, masks, log-probabilities, 64/128/256 threads)
and agreed with `device_sample_reference` on every token, log-probability and alternative list. That is a check of the algorithm, not of the GPU build.

## 5. Known limits / not done

* EAGLE, EAGLE-3, Medusa, tree attention (`cuda/attn_tree.cu`) and KV compaction of tree branches: not implemented. `speculative::verify_tree` is ready for them.
* Speculation and `n > 1` for stateful (hybrid / recurrent) models, recurrent-state snapshots, prefix cache for stateful models, `--prefix-state-interval`: not implemented.
* MTP stays CPU-only with one candidate; multi-candidate MTP, MTP on the GPU: not implemented.
* Prompt logprobs, `echo` log-probabilities, `include_stop_str_in_output`, `skip_special_tokens`, `logits_processors`: not implemented (`best_of`,
  `bad_words`, `allowed_token_ids`, `logprobs_mode`, `min_tokens`, `ignore_eos`, `stop_token_ids` are).
* Host-memory prefix-cache tier (`--kv-offload-gib`): not implemented.
* The device sampler and block copy on CUDA, device swap and the device draft model are unproven on hardware.
* A seeded sampled request gives different text with speculation than without (same distribution, different random numbers).
* Philox is the default host RNG: seeded outputs of earlier versions are not reproduced unless `FLLM_SAMPLER_RNG=xorshift`.
