# Colab verification for fLLM releases

One script for a Google Colab cell that verifies a fresh fLLM build end to end on the
Colab GPU and prints a PASS/FAIL/SKIP table plus a `verify_result.json` for download.

```sh
python3 scripts/colab/verify_all.py               # full GPU run (L4, A100, H100, G4, T4)
python3 scripts/colab/verify_all.py --dry-run     # print the plan, download nothing
python3 scripts/colab/verify_all.py --cpu-only    # CPU checks only, any machine
```

`--build-from-source` builds `main` on the VM: with the CUDA kernels compiled in parallel (one nvcc per core) that took 7 min 23 s on a 12-core L4 VM, 5 min 22 s for the engine alone once the dependencies were built (measured 2026-11-07; before the change a build was still running after 44 minutes). It also clones CUTLASS and FlashInfer like the release build, which the attention checks need. The default (no flag) downloads the release binary (`fllm-linux-x86_64-cuda.tar.gz`, built for every architecture from Pascal to Blackwell) and starts checking after about a minute.

Useful options: `--binary PATH` (verify a binary you built), `--build-from-source`
(slow, builds `main` with the CUDA kernels for the detected card), `--version TAG`
(a specific release instead of `latest`), `--work DIR` (download/working directory,
default `/content/fllm_verify` on Colab), `--steps a,b,c` (subset of the steps).

The script needs only python3; model and binary downloads are the only network access.
On Colab the first full run downloads about 8 GB of small models (see below) and takes
roughly 20-40 minutes on an L4. Every step has a timeout and no failing step aborts the
run - the table shows what passed, failed or was skipped.

## What it checks

| Step | Check |
| --- | --- |
| Acquire | the binary: release asset (`fllm-linux-x86_64-cuda.tar.gz`, sha256-verified), `--binary` or `--build-from-source` |
| Hardware | `nvidia-smi`, compute capability, derived gates (FP8/FA3/NVFP4) |
| Binary | `--help` works, size/sha256 recorded |
| Devices | compiled backends (`cuda` must be in the list) |
| selftest-gpu | full run: ok/fail counts and the failing checks |
| selftest-cpu | full run (CPU-only mode) |
| Dense | greedy text GPU vs CPU, Llama-3.2-3B-Instruct Q4_K_M; a difference at a token whose two best candidates are within 0.25 nats on both devices passes and says so (`test_near_tie.py`), any other difference fails |
| MoE | greedy text GPU vs CPU, OLMoE-1B-7B-0924-Instruct Q4_K_M |
| GPTQ import | Qwen2.5-1.5B-Instruct-GPTQ-Int4 converted and compared GPU vs CPU |
| NVFP4 import | runs only on sm_120 and with `FLLM_VERIFY_NVFP4_MODEL` set (no small public NVFP4 checkpoint) |
| Speed | decode and prefill tok/s at 355/1863/8000 context vs `scripts/colab/reference.json` (warning only, see below) |
| CUDA graphs | decode with `FLLM_CUDA_GRAPH=0` and `1`: same text, engine reports the state |
| Spec decoding | prompt lookup (`--speculative-lookup`): same text as without |
| FP8 KV cache | `--kv-cache-type fp8_e4m3` from sm_89, text compared against the dense run |
| FP16 activations | `FLLM_ACT_DTYPE=f16`, skipped unless the binary knows the switch |
| Vision | SmolVLM-500M + projector: image arrives at `/v1/images/embeddings` |
| Embeddings | bge-small: two 384-dim vectors from `/v1/embeddings` |
| Multi-GPU | skipped on the single GPU of a Colab session |

Card gates: FP8 needs sm_89 (L4, G4), Hopper paths sm_90 (H100), NVFP4 sm_120
(G4 = RTX PRO 6000 Blackwell); T4 is sm_75 (fp16 only), A100 sm_80.

Steps are skipped - not failed - when the running binary does not have the option or
feature. That is detected from `--help` and a search in the binary itself, never from a
version number, so the script also works with older releases.

## Reading the result

* `verify_result.json` next to the run (in the work directory): full step data, hardware,
  gates, binary hash. `verify_result.md` is the same table as a file; download either one
  from the Colab file browser.
* Exit code 0 means no step failed.
* Speed comparison: reference numbers live in `scripts/colab/reference.json`, per card.
  Values are single-user greedy decode rates from `docs/gpu-backends.md`. A drop of more
  than 10 % is printed as a warning only - run-to-run noise is around 15 %. Cards without
  reference numbers just record the measurement.

## Limits

* Not a benchmark replacement: no multi-run medians, no cross-run comparison (that is
  `fllm-rs bench` with `bench-report`).
* Speculative decoding checks text equality, not a speedup claim.
* `--build-from-source` builds only the CUDA architectures of the detected card, so the
  resulting binary is not the release artifact.

## Comparison with vLLM, SGLang, llama.cpp, ik_llama.cpp and Ollama

First results on an L4, an A100, an RTX PRO 6000 Blackwell, a T4 and two CPUs (fLLM against llama.cpp and ik_llama.cpp, a quant matrix of eleven models) and the gaps they showed: [docs/colab-comparison-2026-11.md](../../docs/colab-comparison-2026-11.md). `--cpu --threads N` runs the three engines that read GGUF on the processor; `--models` takes the quant keys listed in `--help`.

`compare.py` serves the same models with fLLM and the other engines on the Colab GPU: GGUF files with fLLM, llama.cpp (built with
CUDA) and Ollama; AWQ 4-bit checkpoints with vLLM, SGLang and fLLM (converted from the same checkpoint). It measures approximate prompt
sizes of 100, 500 and 2000 with 1, 4 and 16 simultaneous users (aggregate and per-user end-to-end tokens/s, first output delta latency):

```sh
git clone -q https://github.com/Marco-Lu2508/fLLM && python3 fLLM/scripts/colab/compare.py                       # release binary
python3 fLLM/scripts/colab/compare.py --build-from-source --models llama8b-awq,qwen3b-awq,gemma4 --users 1,8,32   # newest main
```

Measurement contract (`compare_result.json`, `schema_version: 2`; per-user rates changed meaning from the older chunk-based files):

* Every stream requests `stream_options.include_usage=true`. fLLM supports a final usage-only event (`choices: []`) before `[DONE]`.
  `requests` retains each request's `prompt_tokens`, `completion_tokens`, `cached_tokens` (from
  `usage.prompt_tokens_details.cached_tokens`) and seed. Missing counts stay `null`, not zero.
  `prompt` is still the requested **approximate** size; `actual_prompt_tokens` lists server-tokenized lengths including the chat template.
* `total_tok_s` is summed completion usage divided by batch wall time. `each_tok_s` is now the mean per-request **end-to-end** rate
  (completion usage / request duration, including prefill and TTFT), identified by `each_tok_s_basis`. A post-first-token rate cannot
  be recovered reliably from SSE: a single delta may contain multiple tokens. `first_s` measures the first nonempty content/reasoning
  delta, not an empty role event, and is `null` if any request emits none.
* `token_source` is `usage`, `sse_chunks_unverified` or `mixed`. Without valid completion usage for **every** request, both tok/s fields
  are `null` (table: `-` plus a warning). `chunks` / `total_chunks_s` are explicitly unverified output-delta counts/rates, never token
  counts/rates. Stream errors, malformed JSON and missing `[DONE]` invalidate the request/batch rather than reporting partial success.
* The fLLM control always sets `FLLM_PREFIX_CACHE=off`, `FLLM_OUTPUT_CACHE=off`, `FLLM_SPECULATIVE_DEFAULT=off` and `--speculative off`;
  these settings are recorded under `fllm_control`. Speculation overrides in `--fllm-args` are rejected. Other engines retain their own
  cache defaults; inspect reported cached counts when comparing. This is not a cache/speculation speedup test.
* `--seed N` (default `0`) gives identical prompt nonces and sampler seeds across engines and A/B runs, independent of thread order.
  Nonces differ within each batch; warm-up and answer-probe requests are separate. Equal seeds do not guarantee cross-engine text equality.

Offline regression and in-memory mutation checks (no token server or Colab needed):

```sh
python3 -B scripts/colab/test_compare_measure.py -v
python3 -B scripts/colab/test_compare_measure.py --mutations
```
