# Rust tensor formats and checkpoint import

The valid GGML storage table is pinned to `b1ff4ca23630ac7c5a275405353ddb5c486332c9` (35 valid IDs; slots 4, 5, 31–33 and 36–38 are removed). Tensor storage support is separate from model architecture support and optimized kernels.

## GGUF v2/v3

Pure Rust decoding supports F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, IQ1_M, TQ1_0, TQ2_0, MXFP4, NVFP4, Q1_0, Q2_0 and scalar I8/I16/I32/I64/F64. Removed type slots fail explicitly. Quantized row lengths and payload sizes are validated before decoding.

Optimized CPU kernels cover the classic, Q8, K and float formats, and on x86-64 also Q2_K, Q3_K, IQ4_NL, IQ4_XS and MXFP4 (docs/cpu-kernels.md). The other added formats retain packed mmap-backed weights and decode individual rows for projections; this is a functional path, not a claim of optimized throughput. GPU formats without a specialized WGSL kernel use an F32 upload. 

## Checkpoint containers

Rust `convert` imports local or Hub-backed safetensors (single file or HF weight-map shards) and restricted PyTorch ZIP state dictionaries (`.bin`, `.pt`, `.pth`, including HF shards). Config and tokenizer files must accompany the weights. It writes GGUF directly; neither Python nor a llama.cpp converter process runs.

Safetensors and uncompressed contiguous PyTorch storage are mmap-backed. GGUF output is streamed with a bounded conversion buffer and atomically published. The PyTorch reader implements an allowlisted pickle interpreter for tensor/storage reconstruction; legacy raw pickle, arbitrary Python constructors, TorchScript and unknown opcodes are rejected.

Dense HF mappings cover supported Llama, Mistral, Qwen2, Qwen3 and K2-Horizon layouts. Model family names alone do not imply variant support. MoE, unsupported scaled/partial RoPE and unknown forward-affecting weights fail rather than silently producing a different model. Llama/Mistral Q/K rows are permuted from HF split-half to GGUF interleaved rotary layout; Qwen rows retain their original order. The complete HF tokenizer JSON is retained and validated against GGUF token IDs.

Checkpoint output types are preserve, F16 and F32. This does not add a GGUF quantizer: loading quantized files and creating new quantized weights are separate operations.

### Quantized checkpoints

`convert` reads these quantization schemes (the declaration is `quantization_config` in `config.json`, or `hf_quant_config.json`):

| Scheme | Output with `--outtype preserve` | Notes |
| --- | --- | --- |
| AutoGPTQ 4-bit (also act-order), AutoAWQ 4-bit GEMM with zero points | `Q4Z32/64/128`, `Q4ZP32/64/128` (fLLM-specific), else F16 | unchanged values |
| FP8 (`quant_method` fp8, ModelOpt `quant_algo` FP8, compressed-tensors float 8 bit) | `FP8M` for one scale per tensor or row; F16 for block-wise `weight_scale_inv` (`weight_block_size` or `block_structure`) | `weight_scale_inv` multiplies the decoded value, as in the DeepSeek and transformers FP8 code |
| ModelOpt NVFP4 | GGUF type 40 plus `<name>.scale` and `<name>.input_scale` | details in [awq-gptq-marlin.md](awq-gptq-marlin.md); other `quant_algo` values are refused |
| compressed-tensors `pack-quantized` (symmetric int4, int8) and `int-quantized` (symmetric int8) | int4 with group 32/64/128: `Q4Z`; 8 bit with a scale per 32 or more inputs: `Q8_0`; else F16 | scales may be f16, bf16 or f32 (rounded to f16 in the blocks) |
| EXL2, EXL3, gpt-oss MXFP4 | see [exl-formats.md](exl-formats.md) | |

`--outtype f16` or `f32` expands every quantized weight. Refused with a message that says what to use instead: ModelOpt algorithms
other than NVFP4 and FP8 (`unsupported ModelOpt quant_algo "..."`), compressed-tensors NVFP4 (`compressed-tensors NVFP4 (format
"nvfp4-pack-quantized") is not supported; use an NVIDIA ModelOpt NVFP4 checkpoint ...`), asymmetric or act-order or sparse
compressed-tensors, mixed-precision `config_groups`, and other quantization methods (`unsupported quantization_config quant_method
"..." (supported: gptq, awq, fp8, compressed-tensors, modelopt NVFP4/FP8, exl2, exl3)`).

### Conversion on first use

`run`, `serve` and `pull` resolve a model argument in this order: a `.gguf` file; a directory with `.gguf` files; a Hub reference
with a file name or `:QUANT`; a Hub repository with GGUF files (the GGUF wins even when the repository also holds a checkpoint);
a Hub repository or local directory with a checkpoint (`config.json` plus `model.safetensors`, a `*.safetensors.index.json` or
`pytorch_model.bin`). A checkpoint is handled like this:

1. Hub: the revision listing is fetched, then `config.json` (and `hf_quant_config.json`). The architecture and the quantization are
   checked from them; an unsupported `model_type`, a missing `tokenizer.json` (the conversion builds the vocabulary from it) or a
   LoRA adapter repository is reported before any weight file is downloaded.
2. The output type is chosen: the checkpoint's own quantization for AWQ, GPTQ, FP8, NVFP4, compressed-tensors; for unquantized
   checkpoints the stored dtype (`--outtype f16|f32` changes it) or `--quant TYPE` (`q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`,
   `q4_k_s`, `q4_k_m`, `q5_k_s`, `q5_k_m`, `q6_k`; converted to F16 first, then re-encoded as `quantize` does). `--quant` on a
   quantized checkpoint is an error.
3. The free disk space is compared with the download plus the converted file; a 16-bit model of 7B parameters or more that exceeds
   the free memory of the device prints a hint with the `--quant` sizes.
4. Weights are downloaded (resumable, `HF_TOKEN` for gated or private repositories; a gated repository without a token fails with
   the variable name), converted to `<name>.fllm-part` and renamed, so the final name only ever points to a complete file.

Cache: a Hub model is written into the snapshot directory of its commit, `models--owner--repo/snapshots/<commit>/<repo>-<label>-fllm<key>.gguf`
with `<name>.gguf.json` (source, output type, converter version) beside it. `<key>` hashes the converter version, repository, commit,
output type and the weight file list, so a new commit, another `--quant` or a newer converter creates a new file and a second start
reuses the old one. A local directory is converted into `<cache>/fllm-local/`, keyed by its canonical path and the size and
modification time of its config and weight files. `fllm-rs list` shows converted files with their origin, `show` of a checkpoint
lists the GGUF made from it, `rm owner/repo` removes everything, `rm owner/repo --converted` only the converted files, `rm DIR`
the conversions of a local directory.

A conversion that fails removes its partial file. After a kill (Ctrl-C included) the partial files `<name>.gguf*.fllm-part*` stay
until the next start of the same conversion, which removes them while holding the conversion lock (a second process waits for the
first); they are never mistaken for models.

## Commands

```sh
./run.sh formats
./run.sh inspect model.gguf
./run.sh pull hf://owner/repo/model.gguf --revision main
./run.sh convert /models/checkpoint /models/model.gguf --outtype f16
./run.sh serve --model /models/model.gguf --compute rust
./run.sh chat --url http://127.0.0.1:8000
./run.sh bench --url http://127.0.0.1:8000 --requests 4 --concurrency 1
```

HF downloads resolve revisions to immutable commit snapshots in the standard HF cache, honor HF_HOME/HF_HUB_CACHE/HF_TOKEN/HF_ENDPOINT and support `--local-files-only`. Files are streamed and published without replacing an existing cached file. For GGUF choose a concrete filename instead of downloading every quantization in a repository.

## References

- [GGML tensor enum](https://github.com/ggml-org/llama.cpp/blob/b1ff4ca23630ac7c5a275405353ddb5c486332c9/ggml/include/ggml.h)
- [GGML quantization reference](https://github.com/ggml-org/llama.cpp/blob/b1ff4ca23630ac7c5a275405353ddb5c486332c9/ggml/src/ggml-quants.c)
- [Safetensors format](https://github.com/huggingface/safetensors#format)
- [PyTorch serialization layout](https://docs.pytorch.org/docs/stable/notes/serialization.html#serialized-file-format-for-torch-save)

No per-format throughput claim is made here.
