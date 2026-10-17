# Proof scripts

These scripts check fLLM against independent references (`transformers`, PyTorch, `gguf`) and against the CPU path. They are how the
claims in the documentation are reproduced. Test levels (L0 to L3) are defined in [docs/test-levels.md](../docs/test-levels.md).

## Setup

```sh
cargo build --release --locked -p fllm-engine --features rust-gpu --bin fllm-rs
cargo build --release --locked -p fllm-engine --example verify_checkpoint --example tokenize --example verify_quant
python3 -m venv .venv && . .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r scripts/requirements.txt
```

`make venv` does the Python part. Disk: the CPU torch wheel and `transformers` need about 1 GB in the virtual environment. The
architecture checks use tiny random checkpoints (a few hundred kilobytes each) in a temporary directory; the model-level checks need a
small real GGUF (`fllm-rs pull hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf` is 680 MB). Nothing here downloads
a model unless you ask for it (`fetch_feature_models.py`, `pull`).

## Checks that need no model download

| Script | What it proves | Example | Runtime |
| --- | --- | --- | --- |
| `verify_architectures.py` | For each architecture: a random tiny HF checkpoint (`hf_golden.py`) is converted by `fllm-rs convert`, run by fLLM on a fixed token sequence, and every position's logits agree with `transformers` (largest difference relative to the logit range below 2e-3, no clear arg-max flip) | `python scripts/verify_architectures.py --fllm target/release/fllm-rs --example target/release/examples/verify_checkpoint llama qwen2 gemma2 qwen3 mixtral qwen3_moe` | about 3 s per architecture |
| `verify_generation.py` | Greedy generation through the HTTP server (four concurrent requests) equals `transformers.generate`; exercises incremental decoding and recurrent state | `python scripts/verify_generation.py --fllm target/release/fllm-rs llama qwen3 qwen3_moe` | about 5 s per architecture |
| `verify_tokenizer.py` | fLLM's tokenization of a GGUF equals the Hugging Face tokenizer on a set of hard strings | `python scripts/verify_tokenizer.py MODEL.gguf HF_DIR target/release/examples/tokenize` | seconds |
| `verify_rust_quants.py` | CPU (and with `--device`, GPU) quantized matmuls against `gguf.dequantize` on a generated fixture for every supported type | `python scripts/verify_rust_quants.py --create --fixture /tmp/q.gguf --binary target/release/examples/verify_quant` | seconds |
| `verify_mxfp4.py` | MXFP4 import and the resulting logits against the unquantized tiny checkpoint (informational error) | `python scripts/verify_mxfp4.py --fllm target/release/fllm-rs --example target/release/examples/verify_checkpoint` | seconds |
| `compare_gguf.py` | Tensor-by-tensor type and byte comparison of two GGUF files | `python scripts/compare_gguf.py ours.gguf reference.gguf` | seconds |

`hf_golden.py` itself builds one fixture: `python scripts/hf_golden.py ARCH OUTDIR`. The list of architectures is the key set of
`config_for` in that file. A few architectures use remote code that is not in `transformers` and expect a directory of that code in an
environment variable (`K2HZ_DIR`, `IL2_DIR`, `G9V3_DIR`); those cases are skipped unless you provide it.

## Checks that need a small real model

| Script | What it proves | Example |
| --- | --- | --- |
| `verify_features.py` | Authentication, tokenize/detokenize, JSON schema and GBNF constrained output, embeddings, required tool calls and streaming, seeded concurrent requests, invalid parameters, Ollama options; with `--draft-model` also draft/target equality, with `--lora` adapters | `make test-model MODEL=qwen2.5-0.5b-instruct-q8_0.gguf` |
| `verify_encoder.py` | BERT-style encoders (and ModernBERT, NomicBERT, XLM-R) against PyTorch with the same GGUF weights; pooling and normalization | `make test-encoder MODEL=bge-small-en-v1.5.gguf` |
| `verify_real_embedding.py` | Cosine of HTTP embeddings against `transformers` for a real encoder | `--binary ... --gguf M.gguf --hf-dir HF_DIR` |
| `verify_vision.py` | SigLIP/Idefics3 (SmolVLM) vision parity and multimodal chat | `--binary ... --model M.gguf --mmproj P.gguf` |
| `fetch_feature_models.py` | Downloads the small pinned test models (revision and SHA-256 verified) used by the checks above | `--output-dir DIR --asset vision` |
| `api-tests/run_all.sh` | The OpenAI, Anthropic and Ollama Python SDK clients against a live server (`openai_client.py`, `anthropic_client.py`, `ollama_client.py`) | `MODEL=M.gguf VENV=.venv scripts/api-tests/run_all.sh` |
| `api-tests/cli-tests.sh` | CLI acceptance: help, devices, formats, chat, bench smoke, `run` end to end, keyed server | see the header of the script |
| `bench-compare.sh` | Quick time-to-first-token check of one model on the CPU | `MODEL=M.gguf scripts/bench-compare.sh` |

Expected runtime of `verify_features.py` with the 0.5B model on a laptop CPU: about a minute.

## Backend verification scripts

`verify-<backend>.sh` (and `.ps1` for Windows) run the procedure of [docs/backends-verify.md](../docs/backends-verify.md) on the machine
that has the hardware: build with the feature, list devices, `selftest-gpu`, start a server, compare greedy text with the CPU, benchmark,
and write a JSON report. They share `verify-common.sh` / `verify-common.ps1`. Backends: `windows`, `vulkan`, `metal`, `hip`, `sycl`,
`npu-openvino`, `tpu`, plus the CPU checks `arm` and `avx512` (which call `check-arm-cpu.sh` and `check_x86_ext.sh`). Example:

```sh
scripts/verify-vulkan.sh --model-uri hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf
```

Without the hardware these scripts reach L0 only; their output on the device is the evidence for L1 and L2.

Static helpers: `check_metal_kernels.py` (every Metal kernel the Rust planners request is defined in the `.metal` sources),
`hip_mma_layout_check.py` and `hip_q4z_layout_check.py` (emulations of the AMD matrix-core and Q4Z layouts against references),
`cmpdump.py` (compare two layer dumps written with `FLLM_DUMP_LAYERS`), `gen-cli-reference.sh` (generate a CLI reference from `--help`),
`colab/stablehlo_verify.py` (parse the StableHLO programs of the TPU backend; no TPU needed).

## Colab scripts (`scripts/colab/`)

`ab_matrix.py` (A/B matrix of the CUDA switches and the measurements in the README), `cpu_confirm.py` (CPU tier checks and speed),
`nvfp4_quality.py` (perplexity of NVFP4 variants, optionally against vLLM), `tpu_check.py` and `tpu_check.md` (TPU runtime). They are
written for Google Colab runtimes; see [docs/benchmarking.md](../docs/benchmarking.md).

## Receipts

The first reproducible receipts of these scripts (commands, versions, results) are in `docs/changes/` until they are merged into the
documentation. Include the same information when you publish your own results.
