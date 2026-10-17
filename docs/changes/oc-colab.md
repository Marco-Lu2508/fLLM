# Colab verification script (`scripts/colab/verify_all.py`)

## What it is

A single-file python3 script (standard library only) that the maintainer runs in a
Google Colab cell with a GPU runtime (T4, L4, A100, H100 or G4 = RTX PRO 6000 Blackwell).
It verifies one fLLM build end to end - self test, correctness against the CPU, speed and
the newer switches - and prints a PASS/FAIL/SKIP table plus `verify_result.json` for
download. Quick options: `--dry-run` (plan only), `--cpu-only` (CPU checks, any machine),
`--binary PATH`, `--build-from-source`, `--version TAG`, `--steps a,b,c`.

## How it gets fLLM

Default: the latest GitHub release asset `fllm-linux-x86_64-cuda.tar.gz` (the glibc CUDA
build from `release.yml`; the musl asset cannot load the NVIDIA driver, see
`docs/changes/oc-release.md`), with the `.sha256` file verified. `--cpu-only` uses the
musl asset, `--binary` takes a local build, `--build-from-source` builds the current
`main` with the CUDA kernels of the detected card (slow; not the release artifact).

## Checks

Hardware (`nvidia-smi`, compute capability, derived gates), binary (`--help`, hash),
devices (compiled backends), full `selftest-gpu` with ok/fail counts and the failing
checks, full `selftest-cpu` (in `--cpu-only` mode), greedy text GPU vs CPU for
Llama-3.2-3B-Instruct Q4_K_M, OLMoE-1B-7B-0924-Instruct Q4_K_M and a GPTQ import
(Qwen2.5-1.5B-Instruct-GPTQ-Int4, converted on the fly), NVFP4 on sm_120 with
`FLLM_VERIFY_NVFP4_MODEL` configured, decode/prefill tok/s at 355/1863/8000 context
against `scripts/colab/reference.json`, and the optional features below.

Card gates: FP8 from sm_89 (L4, G4), Hopper paths sm_90 (H100), NVFP4 sm_120 (G4);
T4 is sm_75 (fp16 only), A100 sm_80. The gates decide which steps apply, not which pass.

## Optional features: probes instead of version numbers

CUDA graphs (`FLLM_CUDA_GRAPH=0` vs `1`, same text, engine reports the state),
FP16 activations (`FLLM_ACT_DTYPE=f16`), speculative decoding (prompt lookup,
`--speculative-lookup`), vision (SmolVLM-500M + projector via `--mmproj`),
embeddings (bge-small via `--embeddings`) and the multi-GPU self test are all skipped
with a reason when the running binary or the card does not offer them. Support is
detected from `--help` plus a search in the binary itself, never from a version number:
the hand-written usage text does not print every option (`--mmproj` is accepted but not
listed), so a `--help` probe alone would produce false "not supported" results.

## Speed comparison

`scripts/colab/reference.json` holds per-card reference values (single-user greedy
decode rates from `docs/gpu-backends.md`; today only for the RTX PRO 6000 / Colab G4).
A drop of more than 10 % prints a warning only - run noise is around 15 %. Cards without
reference values just record the measurement. This is a smoke-level check, not the
`bench`/`bench-report` workflow.

## Environment variables used by the script

| Variable | Meaning |
| --- | --- |
| `FLLM_CUDA_GRAPH` | set to `0` and `1` by the CUDA-graph step, back to back |
| `FLLM_ACT_DTYPE` | set to `f16` by the activation step (when the binary knows the switch) |
| `FLLM_VERIFY_IMPORT_MODEL` | model for the import check (default `Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4`) |
| `FLLM_VERIFY_NVFP4_MODEL` | NVFP4 checkpoint; empty (default) skips the NVFP4 step |
| `HF_HOME` | set to the work directory so models are cached inside it |

## Test level

Tested on a 10-core Linux host without a GPU:

* `python3 -m py_compile` and `pyflakes` clean (pyflakes in a container, the host has no
  pip), `--dry-run` prints the full plan.
* Full `--cpu-only` run: 7 PASS, 0 FAIL, 11 SKIP (Acquire, Hardware, Binary, Devices,
  `selftest-cpu` - scalar 9 checks and avx2 2762 checks, 0 failed -, Vision with SmolVLM-500M,
  Embeddings with two 384-dim vectors). The GPU steps were skipped because the host has
  no NVIDIA GPU; no GPU result is claimed here. That GPU evidence is exactly what this
  script produces on Colab.

## Limits

* First full run downloads about 8 GB of small models (Llama-3.2-3B Q4_K_M,
  OLMoE-1B-7B q4_k_m, Qwen2.5-1.5B GPTQ-Int4, SmolVLM-500M, bge-small).
* Only x86_64 Linux binaries; a Colab session has one GPU, so the multi-GPU step always
  skips there.
* The NVFP4 check needs a configured checkpoint; there is no small public NVFP4 model,
  so the step is a skip by default.
* `--build-from-source` compiles only the detected architecture, so its binary differs
  from the release artifact.
* Speculative decoding checks text equality, not a speedup claim.
