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
| FP8 (`quant_method` fp8, ModelOpt `quant_algo` FP8, compressed-tensors float 8 bit) | `FP8M` for one scale per tensor or row; `FP8B` (fLLM-specific, below) for block-wise `weight_scale_inv` (`weight_block_size` or `block_structure`), F16 only for tiles that are not powers of two | `weight_scale_inv` multiplies the decoded value, as in the DeepSeek and transformers FP8 code |
| ModelOpt NVFP4 | GGUF type 40 plus `<name>.scale` and `<name>.input_scale` | details in [awq-gptq-marlin.md](awq-gptq-marlin.md); other `quant_algo` values are refused |
| compressed-tensors `pack-quantized` (symmetric int4, int8) and `int-quantized` (symmetric int8) | int4 with group 32/64/128: `Q4Z`; 8 bit with a scale per 32 or more inputs: `Q8_0`; else F16 | scales may be f16, bf16 or f32 (rounded to f16 in the blocks) |
| MLX affine (`quantization: {bits, group_size}`, 2/3/4/5/6/8 bits) | `MLXA<bits>G<group>[B]` (fLLM-specific, see below), else F16 | words, scales and biases are copied as they are |
| EXL2, EXL3, gpt-oss MXFP4 | see [exl-formats.md](exl-formats.md) | |

`--outtype f16` or `f32` expands every quantized weight. Refused with a message that says what to use instead: ModelOpt algorithms
other than NVFP4 and FP8 (`unsupported ModelOpt quant_algo "..."`), compressed-tensors NVFP4 (`compressed-tensors NVFP4 (format
"nvfp4-pack-quantized") is not supported; use an NVIDIA ModelOpt NVFP4 checkpoint ...`), asymmetric or act-order or sparse
compressed-tensors, mixed-precision `config_groups`, and other quantization methods (`unsupported quantization_config quant_method
"..." (supported: gptq, awq, fp8, compressed-tensors, modelopt NVFP4/FP8, exl2, exl3)`).

### Block-wise FP8 (`FP8B`)

DeepSeek-V3 / R1, Qwen3-FP8, GLM-4.5-FP8, Kimi-K2, MiniMax and similar checkpoints store E4M3 weights with one `weight_scale_inv` per tile of
128 x 128 weights (`weight_block_size` in `quantization_config`, or `block_structure` of compressed-tensors). They stay 8-bit:

* **Layout** (`fllm_gguf_probe::Fp8bLayout`, the one place that knows it): the E4M3 codes of the tensor exactly as in the checkpoint (for
  stacked experts one matrix after the other), then one little-endian **f32** scale per tile (row-major over the tile grid, matrix after
  matrix; `ceil(rows / tile_rows) * ceil(cols / tile_cols)` per matrix, edge tiles are partial), then a 16-byte trailer
  `"F8B1", cols, rows, matrices` (`u32` each). The trailer makes the tensor describe itself, so a generic reader that only has the bytes
  (`compute::dequantize`, every backend that expands at upload) needs no side information. A weight is `e4m3(code) * scale`. The cost is
  `1 + 4 / (tile_rows * tile_cols)` bytes per weight: 1.00024 for 128 x 128. Scales of the checkpoint may be f32, bf16, f16 or UE8M0
  (`F8_E8M0`, powers of two, DeepSeek-V3.1 and later); they are stored as f32, exactly.
* **Type id**: `0x8000_0021` for 128 x 128; `0x8000_0100 | log2(tile_rows) << 4 | log2(tile_cols)` for every other power-of-two tile up to
  32768 (the converter names them `FP8B32x32`, ...). The tile shape is in the id, so the tensor length follows from the id and the GGUF
  dimensions (the probe computes it; this was the bug of the first `FP8B`, which counted only the codes). Tiles that are not powers of
  two (not seen in real checkpoints) still expand to F16 in the converter.
* **Row permutations** (the RoPE reordering of some architectures) are applied to the code rows and, when the permutation moves whole tile
  rows only (head size a multiple of the tile height, as for 128 x 128 and head size 128), to the scales; otherwise the tensor is expanded.
  A scale tensor whose shape is not the tile grid of its weight is refused.
* **Slicing**: `Linear::load_rows` cuts one expert (or a tile-aligned range of rows) out of a stacked tensor and rebuilds codes, scales and
  trailer for it (`compute::fp8b_rows`); a cut inside a tile row is refused with a message (it cannot keep its scales). A whole single
  matrix is a view. Tensor-parallel splitting of `FP8B` is refused (`parallel::shard::layout_for`); pipeline and expert parallelism work.
* **CPU** (`cpu_fp8.rs`): a row is expanded to f32 in a scratch buffer tile by tile with an integer-only E4M3 decoder (no table), then
  multiplied with every activation row; AVX2+FMA instance chosen at run time (`cpu_kernels::isa`), NEON with its native FMA, portable code
  elsewhere; threads split the output rows, so the result does not depend on the thread count. It backs the CPU backend and is the
  reference of the device checks.
* **CUDA**: [gpu-backends.md](gpu-backends.md#block-wise-fp8-fp8b) (the weight stays 8-bit on every NVIDIA GPU).
* **Other backends** (Vulkan, wgpu, Metal, SYCL, TPU, AMD HIP): no kernel; the codes are expanded with their scales to binary16 at upload
  (exact; 2 bytes per weight; one line on stderr names the first tensor). Correct, not compact.
* **Direct load**: the codes are copied once into the anonymous map next to their scales (the scales have to follow the codes in the same
  tensor, so the tensor cannot be a slice of the checkpoint file). Dense FP8B tensors therefore cost their size in anonymous memory, like
  FP8M, Q4Z and NVFP4 today; stacked experts have to be built anyway. The checks of the memory limit apply.

### MLX affine quantization (`MLXA`)

mlx-lm and mlx-community exports quantize every linear (and, with `nn.quantize`, the embedding and the head) per group of 32, 64 or 128 inputs:
`weight` holds `bits`-bit codes in `uint32` words (code `i` of a row at bit `i * bits` of a little-endian bit stream, so the 3, 5 and 6 bit
codes straddle word boundaries), `scales` and `biases` one value per group, `w = scale * code + bias`. The bias is a float, not a multiple of
the scale, so Q4Z (an integer zero point) cannot hold it exactly; the weights stay native as their own tensor types instead:

* **Type id** `0x8001_0000 | bf16 << 12 | (group / 32) << 4 | bits` (`fllm_gguf_probe::MlxAffine`; the converter names them
  `MLXA4G64`, `MLXA4G64B` for bfloat16 scales). Bits 2, 3, 4, 5, 6 and 8; any group that is a multiple of 32.
* **Block** (one group): scale and bias as two 16-bit floats (f16, or bfloat16 when the checkpoint\'s values need it), then
  `group * bits / 8` bytes of codes exactly as the MLX words hold them. A row is `cols / group` blocks, so the file costs what the
  checkpoint costs (4.5 bits per weight for 4-bit with group 64). Tensors that are rows of blocks need no special treatment in the
  loader: row slices, tensor-parallel splits along the inputs (by groups), memory mapping, and embedding lookups work as for the other
  block formats.
* **Exact or not at all.** The converter checks that every scale and bias of a tensor survives the round trip through f16 (preferred) or
  bfloat16 bit for bit. If not (float32 scales with a full mantissa, as `mx.quantize` makes them in float32), the width is 7 bits, or
  the group is not a multiple of 32, the tensor is expanded to F16 as before and the converter prints a warning that says why. Nothing is
  re-quantized. mlx-community checkpoints store bfloat16 or f16 scales, so they are native.
* **Refused**: the `mxfp4` and `nvfp4` modes of MLX (`"mode"` in the `quantization` dictionary; their E8M0 / E4M3 scales and the lack of
  biases need their own reader) and stacked expert weights (3-dimensional `weight`), with a message.
* **CPU** (`cpu_mlx.rs`): f32 activations against the packed codes; 32 codes are unpacked at a time with the bit width as a constant,
  tiles of 4 rows are decoded once for all tokens, AVX2 + FMA where the CPU has them. It is also the reference of the device tests.
* **CUDA** (`cuda/mlx_affine.cu`, device image in `backend/mlx.rs`): the codes, scales and biases become three planes; a GEMV (one warp per
  row, up to 4 tokens per pass), an FP32 tile GEMM and a tensor-core GEMM (WMMA, weights dequantized into FP16 fragments in shared memory,
  from sm_75) read the packed words directly. Group sizes and bit widths are not templated beyond the bit width. AMD GPUs expand to F16.
* **Metal** (`metal/common.metal` `q_fmt_mlx`, `q_*_mlx.metal`, `backend/metal/weights`): the kernels of the quantized linear family with
  a chunk decoder for the MLX codes; the codes plane and one `scale | bias << 16` word per group; bit 63 of the plane-1 offset marks bfloat16.
  `FLLM_METAL_NO_MLX=1` switches them off (the decoder then expands to F16 on upload).
* **Other backends** (Vulkan, wgpu, SYCL, NPU, TPU, HIP): no kernel; `RustDecoder::init_backend` checks the capability list and uploads the
  weight dequantized as F16 (one line on stderr names the first tensor). Correct, 16 bits per weight on the device.

What is proven (CPU tests) and what is not: see the table below and the end of this section.

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
- Default (neither flag): direct for every source quantization the converter reads: `none`, `awq`, `gptq`, `nvfp4`, `fp8`, `int4`,
  `int8` (compressed-tensors), `mxfp4` (gpt-oss and compressed-tensors), `bitsandbytes` (NF4, FP4, double quantization, int8),
  `mlx` (affine, 2 to 8 bits), `exl2` and `exl3`, and for PyTorch ZIP weights (`.bin`/`.pt`/`.pth`) of any of these. Each of these
  was moved from "converts and caches" to this list only after a parity test passed (below). The PyTorch reader stays restricted: a
  pickle can only name `collections.OrderedDict` and the torch tensor-rebuild functions, every other global or opcode is refused, and
  nothing in `data.pkl` is ever executed (test `malicious_pickles_are_refused_and_never_run`); a legacy non-ZIP pickle file is
  refused before it is parsed. A finished conversion of the same model in the cache is still used (it starts faster than rebuilding
  the tensors on every start). A checkpoint whose weights exceed three quarters of the free memory (a half of it for already
  quantized ones) is converted too, because the converted file would live in the evictable page cache while the built tensors are
  anonymous memory. bitsandbytes weights count four times their size in that comparison, since the converter expands them to
  16-bit weights; MLX weights and EXL2/EXL3 matrices stay packed (a compact blob is built once into the anonymous map, about a quarter of the former
  F16 size at 4 bits per weight), only stacked expert tensors are still expanded (MLX weights whose scales have no exact 16-bit form too). If the built tensors exceed three quarters of the free memory at load time the load
  stops with a message instead.
- What stays a slice: unquantized tensors that need no change (also those of a PyTorch archive whose storages are stored without
  compression, which is what `torch.save` writes). Everything the converter expands (bitsandbytes, EXL2/EXL3 stacked experts to F16) or builds as a compact blob (EXL2/EXL3, the `MLXA` blocks of MLX) or repacks
  (MXFP4 blocks, gpt-oss experts) is built once into the anonymous map, so these formats need memory for the 16-bit weights (bitsandbytes) or the packed blob (EXL, MLX) where the converted file would have lived in the page cache.
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
| Block-wise FP8 stays `FP8B` for 128 x 128, 32 x 32 and 16 x 16 tiles (dense and as stacked experts) and the engine's logits equal those of the dequantized F32 model (1e-3) and, within 4e-3, those of the F16 expansion of the same checkpoint; UE8M0 scales; a tile without a type id expands; a tensor costs `codes + 4 * tiles + 16` bytes | `import_tests::block_wise_fp8_stays_fp8_and_matches_the_reference`, `block_wise_fp8_experts_stay_fp8_with_the_scales_of_every_expert`, `native_block_wise_fp8_and_its_f16_expansion_give_the_same_logits`, `block_wise_fp8_with_ue8m0_scales_stays_fp8_and_matches_the_reference`, `block_wise_fp8_with_a_tile_without_a_type_is_expanded_to_f16`, `block_wise_fp8_tensors_take_one_byte_per_weight_plus_the_scales` |
| The CPU kernel equals an f64 evaluation of the dequantized weights (partial tiles on both edges, 1 x N and N x 1 tiles, bias, batches 1/2/5, any thread count, portable and FMA loops); scale indexing of the tile grid and of stacked matrices; trailer checks; expert and row-range slicing, with cuts inside a tile row refused | `cpu_fp8::tests::*` |
| The arithmetic of the CUDA kernels run on the host: the decoder formula for every finite code, the GEMV lane/run/scale arithmetic and warp reduction against the reference (odd shapes, tiles from 1 to 128, per-element scales), tile coverage of the tensor-core kernel (constants read from `fp8_block.cu`), scale addressing of the grouped MoE kernel and of the W8A8 kernel | `cpu_fp8::mirror_tests::*` |
| All `fp8b` checks of the GPU self-test (every tile shape, odd shapes, grouped experts) pass on the CPU backend, so the f64 reference and the random weights are right before they run on a GPU | `selftest::extra_group_tests::block_wise_fp8_checks_pass_on_the_cpu_backend` |
| The same logits from a dense checkpoint at `preserve`, `f16` and `f32` | `import_tests::direct_load_of_unquantized_checkpoints_gives_the_same_logits_as_the_converted_file` |
| `prepare` hands out the checkpoint by default, writes nothing to the cache, honours `--no-direct-load`, the environment switches, a cached conversion and `--direct`; `--quant` while loading equals the cached conversion for q8_0, q4_0, q4_k_m; Hub download then direct load | `import::tests::verified_checkpoints_load_directly_by_default_and_no_direct_load_converts`, `an_existing_conversion_is_used_unless_direct_is_forced`, `the_environment_switches_choose_the_path`, `quantizing_on_load_matches_the_cached_conversion`, `every_supported_checkpoint_format_loads_directly_without_being_asked`, `hub_checkpoint_loads_directly_after_the_download_without_a_gguf_file` |
| Every tensor byte, dimension, type and metadata value of the direct load equals the converted file for MXFP4 (compressed-tensors, 1-2 shards, `preserve` and `f16`; the gpt-oss checkpoint of the converter's tests, experts stay MXFP4) | `fllm-checkpoint` `direct_format_tests::mxfp4_compressed_tensors_loads_identically_to_the_converted_file`, `gpt_oss_mxfp4_checkpoint_loads_identically_to_the_converted_file` |
| The same for bitsandbytes NF4 and FP4 (block 64/128, with and without double quantization, 1-2 shards), int8 (1 and 3 shards), NF4 as F32, NF4 with `q8_0` on load | `direct_format_tests::bitsandbytes_checkpoints_load_identically_to_the_converted_file` |
| The same for MLX affine with 2, 3, 4, 5, 6 and 8 bits, group 32 and 64, linears only or also embedding and head, 1-2 shards, bits given as `quantization` or as `quantization_config`; the file holds native `MLXA` tensors of the right type, bfloat16 scales are picked when half does not hold them, float32 scales and group 16 fall back to F16 | `direct_format_tests::mlx_quantized_checkpoints_load_identically_to_the_converted_file`, `mlx_bfloat16_scales_are_native_and_scales_without_an_exact_16_bit_form_are_expanded` |
| Native MLX blocks decode bit for bit like the expanding reader (real `mx.quantize` fixtures for 2-8 bits, groups 32/64, with scales rounded to f16 and bfloat16; an independent word-and-shift decoder) and every bit width unpacks, including the codes that straddle two words | `dequant_bits::tests::mlx_fixtures_become_native_blocks_that_decode_identically`, `native_blocks_unpack_every_bit_width_including_codes_that_straddle_words`, `scales_pick_half_then_bfloat16_and_anything_else_is_not_native`, `mlx_dtype_names_round_trip_to_type_ids` |
| An MLX checkpoint (2-8 bits, groups 32 and 64, embeddings and head quantized too) runs on the CPU engine against its dequantized float32 twin to 1e-4 of the largest logit, and as an F16 expansion to 2e-2; the tensors are native `MLXA` types | `fllm-engine` `import_tests::mlx_affine_checkpoints_stay_native_and_match_the_reference` |
| The CPU kernel equals the dequantized weights to the f32 summation bound for every width, group 32-128, f16 and bfloat16 scales, 1-17 tokens and rows that fill no tile; codes at the word boundaries are read from both words | `cpu_mlx::tests::linear_matches_the_dequantized_weights_within_f32_summation_error`, `straddling_codes_are_read_from_both_words`, `every_code_of_every_bit_width_unpacks_including_the_ones_that_straddle_words` |
| The device image (three planes) and a host model of the CUDA GEMV (lane striding over chunks, 32-lane butterfly sum, FMA order) equal the CPU reference | `backend::mlx::tests::planes_hold_the_words_scales_and_biases_of_every_row`, `the_kernel_model_matches_the_cpu_reference_for_all_widths_and_odd_shapes` |
| Other MLX quantization modes are refused with the reason | `convert::tests::mlx_modes_other_than_affine_are_refused_with_the_reason` |
| Metal: file blocks become the codes plane and the scale words, the kernel names and the bfloat16 flag are planned, a Rust mirror of the MSL chunk decoder equals the reference for all 36 formats; the MSL decoder text itself, compiled as C++, decodes 107520 values of every width, group size and scale format bit for bit | `backend::metal::weights::tests::mlx_planes_hold_every_code_scale_and_bias_of_the_file_blocks`, `mlx_formats_agree_with_the_type_ids_of_the_checkpoint_and_plan_their_kernels`, `repacked_planes_decode_to_the_reference`; `metal/emu/run_dec8_mlx.sh` |
| The same for EXL2 (all group widths 2-8 bits, permuted inputs) and EXL3 (2-4 bits, all three codebooks, dense and bit-packed sign vectors, 1-2 shards) | `direct_format_tests::exl2_checkpoints_load_identically_to_the_converted_file`, `exl3_checkpoints_load_identically_to_the_converted_file` |
| The same for EXL2 (all group widths 2-8 bits, permuted inputs) and EXL3 (2-4 bits, all three codebooks, dense and bit-packed sign vectors, 1-2 shards), as native `EXL2` / `EXL3` tensors and (`--dtype f16`) expanded | `direct_format_tests::exl2_checkpoints_load_identically_to_the_converted_file`, `exl3_checkpoints_load_identically_to_the_converted_file` |
| EXL2 / EXL3 stay packed: the native blob dequantizes to the importer's expanded matrix (EXL2 within 2e-3 per weight, EXL3 within 1e-2 of the RMS; all widths, codebooks, row reordering), the CPU kernels equal the dequantized matrix, and a synthetic EXL2 model's logits equal those of an independent dequantization (5e-3) | `exl2::tests::native_blob_matches_the_expanded_importer_path`, `exl3::tests::native_*`, `cpu_exl::tests::*`, `import_tests::exl2_checkpoint_stays_packed_and_matches_the_dequantized_reference` |
| The same for PyTorch ZIP weights: one stored bf16 archive, one compressed archive, two f16 shards with an index, f32 converted to f16; stored storages stay slices of the archive | `direct_format_tests::pytorch_zip_weights_load_identically_to_the_converted_file`, `stored_pytorch_storages_stay_slices_of_the_archive` |
| A pickle that would run `os.system`, `eval`, `exec`, `subprocess` calls (GLOBAL, STACK_GLOBAL, INST, OBJ, NEWOBJ, EXT, BUILD) is refused and runs nothing, also through the direct load; a legacy non-ZIP pickle is refused | `direct_format_tests::malicious_pickles_are_refused_and_never_run` |
| The memory limit stops the load of MLX, bitsandbytes, EXL2 and compressed PyTorch weights before anything is built | `direct_format_tests::the_memory_budget_applies_to_every_expanded_format` |
| Through `open_model`, tensors and the decoder's logits (finite, bit-identical) equal the converted file for MLX (4 bit g32, 3 bit g64), bitsandbytes NF4 and int8, MXFP4, EXL2 and a PyTorch `.bin` | `fllm-engine` `import_tests::direct_load_equals_the_converted_file_for_mxfp4_bitsandbytes_mlx_exl2_and_pytorch_weights` |
| `prepare` hands these out by default without touching the cache, `--no-direct-load` converts to the same model (used afterwards by default), `--direct` ignores the conversion; a Hub repository with PyTorch weights loads directly after the download | `import::tests::mxfp4_bitsandbytes_mlx_exl2_and_pytorch_checkpoints_load_directly_by_default`, `hub_repository_with_pytorch_weights_loads_directly_after_the_download` |
| The real `fllm-rs serve` binary, started on a checkpoint directory (NVFP4, bf16, bf16 with `--quant q8_0`), writes no GGUF file, names the model after the directory and returns the same greedy completion as with `--no-direct-load` | `import_tests::the_serve_command_loads_checkpoint_directories_directly_and_answers_like_the_converted_file` (needs the binary built next to the test executable, else it skips itself) |

Not proven for MLX affine: any run on a GPU (CUDA `mlx_affine.cu` compiles with nvcc 12.8 for sm_80/89/90/120 and the Rust side builds with `--features cuda`; the self test group `mlx` of `fllm-rs selftest-gpu --only mlx` checks every kernel on a device and has not run on one; the Metal kernels have only been compiled as C++ on the CPU and are awaiting the Apple-silicon CI run), checkpoints from the real mlx-community (the fixtures come from `mx.quantize`, the engine tests from a writer in the test), `mxfp4` / `nvfp4` modes.

Not proven: real multi-GB checkpoints (time to first token, peak memory, behaviour near the memory limit), GPU runs of a directly
loaded model (the tensors reach the GPU through the same `MappedGguf` calls, but only the CPU was run), tensors of a safetensors
file that are not 32-byte aligned (a GGUF file aligns them; the CPU engine reads them unaligned, no GPU back end was run on such
a slice), real EXL2/EXL3/MXFP4/bitsandbytes/MLX/PyTorch checkpoints (the fixtures hold random codes in each layout; the decoders
themselves are checked against the reference libraries by their own tests, not by these), logits of EXL3 through the engine (its
tiles need dimensions of 128, larger than the engine test model; the byte-for-byte comparison of EXL3 is done), the PyTorch `bin`
fixtures are written by test code, not by `torch.save`, and `fllm-rs run` interactively (it passes the same flags to the
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
