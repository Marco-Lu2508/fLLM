# Colab verification for fLLM releases

One script for a Google Colab cell that verifies a fresh fLLM build end to end on the
Colab GPU and prints a PASS/FAIL/SKIP table plus a `verify_result.json` for download.

```sh
python3 scripts/colab/verify_all.py               # full GPU run (L4, A100, H100, G4, T4)
python3 scripts/colab/verify_all.py --dry-run     # print the plan, download nothing
python3 scripts/colab/verify_all.py --cpu-only    # CPU checks only, any machine
```

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
| Dense | greedy text GPU vs CPU, Llama-3.2-3B-Instruct Q4_K_M |
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

## Comparison with vLLM, SGLang, llama.cpp and Ollama

`compare.py` serves the same models with fLLM and the other engines on the Colab GPU: GGUF files with fLLM, llama.cpp (built with
CUDA) and Ollama; AWQ 4-bit checkpoints with vLLM, SGLang and fLLM (converted from the same checkpoint). It measures prompts of 100,
500 and 2000 tokens with 1, 4 and 16 simultaneous users (total tokens per second, time to the first token, speed per user):

```sh
git clone -q https://github.com/Marco-Lu2508/fLLM && python3 fLLM/scripts/colab/compare.py                       # release binary
python3 fLLM/scripts/colab/compare.py --build-from-source --models llama8b-awq,qwen3b-awq,gemma4 --users 1,8,32   # newest main
```
