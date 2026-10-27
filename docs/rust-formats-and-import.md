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

### Direct loading (no GGUF file)

For the checkpoints listed below `run`, `serve` and `pull` do **not** convert to a GGUF file at all. The engine reads the
checkpoint through a view that looks like the GGUF the converter would have written (`fllm_checkpoint::load_checkpoint_quantized`,
`fllm_engine::import::open_model`): the metadata is made from `config.json` and the tokenizer files, the tensors the converter
copies unchanged (most weights of an unquantized bf16/f16/f32 model, FP8 and NVFP4 payloads, norms) are slices of the
memory-mapped safetensors files, and every other tensor (the rope-permuted `q`/`k` projections, packed AWQ/GPTQ/compressed-tensors
repacks, dtype changes, stacked experts) is produced once, by the converter's own writer, into one anonymous read-only map. No
conversion logic is duplicated: `convert` and the direct load share `prepare_checkpoint` and the tensor writer. The disk holds the
checkpoint only; the extra memory is the size of the tensors that cannot be slices.

- `--quant TYPE` (and `--outtype f16|f32`) work while loading. Each float tensor is converted to F16 and re-encoded with the same
  rows-in-parallel quantizer and the same per-tensor mix as `fllm-rs quantize`, one tensor at a time, straight into the result;
  the half-precision form of the whole model never exists, and no file is written.
- `--direct` (or `FLLM_DIRECT_SAFETENSORS=1`) forces the direct load for every checkpoint the converter understands, also those not
  listed below, and ignores a cached conversion. `--no-direct-load` (or `FLLM_NO_DIRECT_LOAD=1`) always takes the older route:
  convert once and cache the GGUF.
- Default (neither flag): direct for source quantizations `none`, `awq`, `gptq`, `nvfp4`, `fp8`, `int4`, `int8` (compressed-tensors)
  in safetensors files. EXL2, EXL3, MXFP4, bitsandbytes, MLX and PyTorch `.bin`/`.pt` weights keep converting (their direct load is
  not tested). A finished conversion of the same model in the cache is still used (it starts faster than rebuilding the tensors
  on every start). A checkpoint whose weights exceed three quarters of the free memory (a half of it for already quantized ones)
  is converted too, because the converted file would live in the evictable page cache while the built tensors are anonymous memory;
  if the built tensors exceed three quarters of the free memory at load time the load stops with a message instead.
- A Hub repository is downloaded into its snapshot directory and that directory is loaded; `pull` then only downloads.
- Each start of the server builds the non-slice tensors again (seconds for small models; not measured for large ones). A converted,
  cached GGUF starts without that work, so `--no-direct-load` can be better for a model you start often and have disk for.

What the tests prove (all on the CPU, tiny synthetic models, nothing from the network):

| Claim | Test |
| --- | --- |
| Every tensor byte, dimension, type and every metadata value of the direct load equals the converted file, for bf16/f16/f32 stored as one file or 2-3 shards, with `preserve`, `f16` and `f32` | `fllm-checkpoint` `direct_tests::unquantized_checkpoints_load_identically_to_the_converted_file` |
| The same for GPTQ 4-bit (group 32), GPTQ 8-bit in two shards, AWQ | `direct_tests::packed_gptq_and_awq_checkpoints_load_identically_to_the_converted_file` |
| Quantizing while loading equals `convert` to F16 followed by `quantize` (q8_0, q4_k_m, q4_0, q6_k, q5_k_s, q2_k, mxfp4) | `direct_tests::quantizing_while_loading_equals_convert_then_quantize` |
| Untouched tensors are slices of the mapped file, the rest share one anonymous map | `direct_tests::untouched_tensors_stay_slices_of_the_checkpoint` |
| The memory limit stops a load before anything is built | `direct_tests::a_memory_budget_stops_the_load_before_anything_is_built` |
| Tensor bytes and the decoder's logits are bit-identical to the converted file for NVFP4, compressed-tensors int4/int8 (both layouts), block-wise FP8, and the mixture-of-experts variants of NVFP4, int4 and FP8 | `fllm-engine` `import_tests::direct_load_equals_the_converted_file_for_every_quantized_layout` |
| The same logits from a dense checkpoint at `preserve`, `f16` and `f32` | `import_tests::direct_load_of_unquantized_checkpoints_gives_the_same_logits_as_the_converted_file` |
| `prepare` hands out the checkpoint by default, writes nothing to the cache, honours `--no-direct-load`, the environment switches, a cached conversion and `--direct`; `--quant` while loading equals the cached conversion for q8_0, q4_0, q4_k_m; Hub download then direct load | `import::tests::verified_checkpoints_load_directly_by_default_and_no_direct_load_converts`, `an_existing_conversion_is_used_unless_direct_is_forced`, `the_environment_switches_choose_the_path`, `quantizing_on_load_matches_the_cached_conversion`, `only_verified_checkpoints_load_directly_without_being_asked`, `hub_checkpoint_loads_directly_after_the_download_without_a_gguf_file` |
| The real `fllm-rs serve` binary, started on a checkpoint directory (NVFP4, bf16, bf16 with `--quant q8_0`), writes no GGUF file, names the model after the directory and returns the same greedy completion as with `--no-direct-load` | `import_tests::the_serve_command_loads_checkpoint_directories_directly_and_answers_like_the_converted_file` (needs the binary built next to the test executable, else it skips itself) |

Not proven: real multi-GB checkpoints (time to first token, peak memory, behaviour near the memory limit), GPU runs of a directly
loaded model (the tensors reach the GPU through the same `MappedGguf` calls, but only the CPU was run), tensors of a safetensors
file that are not 32-byte aligned (a GGUF file aligns them; the CPU engine reads them unaligned, no GPU back end was run on such
a slice), EXL2/EXL3/MXFP4/bitsandbytes/MLX/PyTorch checkpoints, and `fllm-rs run` interactively (it passes the same flags to the
`serve` it starts; that flag list is unit tested).

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
