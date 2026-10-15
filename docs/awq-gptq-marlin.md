# 4-bit AWQ and GPTQ weights (Q4Z, Marlin)

`fllm-rs convert` keeps 4-bit AWQ (AutoAWQ GEMM) and GPTQ (AutoGPTQ, no act-order) checkpoints 4-bit. Every weight matrix becomes a `Q4Z`
tensor, a fLLM-specific GGUF tensor type (ids `0x8000_0010` / `0x8000_0011` / `0x8000_0012` for groups of 32 / 64 / 128): per group of
input values `group / 2` bytes of 4-bit codes (value `2j` in the low nibble), the fp16 scale and one zero-point byte, row after row.
The dequantized weight is `(code - zero) * scale`, exactly what the checkpoint says; nothing is re-quantized. A Llama-3.1-8B AWQ
checkpoint becomes a 5.8 GB file (the embeddings and the output matrix stay fp16).

GPTQ checkpoints with act-order (`desc_act = true`, where `g_idx` is permuted) are converted into `Q4ZP` tensors (GGUF
type ids `0x8000_0014` / `0x8000_0015` / `0x8000_0016` for group sizes 32 / 64 / 128): the columns are grouped by `g_idx` into
contiguous quantization groups with their scales and zero points, and the permutation table (`argsort(g_idx)`) is appended as `cols * 4`
bytes to the tensor payload.

`--outtype f16` / `f32` expands such tensors as before (needed for the formats below, which fall back to it automatically):

- group sizes other than 32, 64 and 128, fp32 scales, stacked experts.

## Execution

On NVIDIA GPUs from Turing (sm_75) on, the CUDA backend repacks a `Q4Z` matrix once while it uploads it, with the repack kernel of vLLM,
and runs every linear layer, decode and prompt, with vLLM's Marlin GEMM (`rust/engine/cuda/marlin_vllm`, see `THIRD_PARTY.md`). Shapes
must be multiples of 128; others, older GPUs, the CPU and Vulkan dequantize (`compute::dequantize`).

The activations of fLLM are FP32, Marlin takes FP16: `marlin_vllm_glue.cu` converts around the call, folds a pending SwiGLU / GeGLU into
the conversion of the next GEMM and reuses the converted activations of q/k/v and gate/up. For `Q4ZP` tensors, the activation conversion
fuses the permutation gather `x16[row, col] = x[row, perm[col]]` directly with the FP16 conversion (and pending SwiGLU folding), vectorized
with 128-bit loads (`int4`). A device permutation cache in `CudaBackend` deduplicates permutation tables across projections (`q_proj`, `k_proj`,
`v_proj` and `gate_proj`, `up_proj`), enabling `reuse_x16` across all projections that share activation inputs.
`FLLM_CUDA_NO_GLU_FUSE=1` and `FLLM_CUDA_NO_X16_REUSE=1` switch those off.

## Measured

Llama-3.1-8B-Instruct AWQ (group 128), RTX PRO 3000 Blackwell Laptop (sm_120), `sb.sh`, greedy text checked:

| | fLLM | vLLM 0.31 (`awq_marlin`) |
|---|---|---|
| prompt, 356 tokens | 3930 tok/s | 4430 tok/s |
| prompt, 1863 tokens | 4040 tok/s | 4310 tok/s |
| decode | 101 / 97 tok/s | 96 tok/s |

Qwen2.5-1.5B-Instruct GPTQ-Int4 (group 128, `g_idx` sequential): 1.17 GB file, 17 400 tok/s prompt, 250 tok/s decode, greedy text equal to
the expanded fp16 conversion of the same checkpoint.

Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 (`desc_act = true`, group 128): 5.78 GB file (`Q4ZP128`), RTX PRO 3000 Blackwell Laptop (sm_120):
- Prompt (356 tok): 3646 tok/s (92.7% of AWQ)
- Prompt (1863 tok): 3762 tok/s (93.2% of AWQ)
- Decode: 100.2 tok/s (98.6% of AWQ) / 95.6 tok/s (98.4% of AWQ)
Greedy text is identical/consistent with the reference. `selftest-gpu` covers all group sizes (32, 64, 128) for both `Q4Z` and `Q4ZP`
across all batch sizes (1 to 356), totaling 330 checks.

## Importing NVFP4 checkpoints

ModelOpt NVFP4 safetensors (`quant_algo: NVFP4`; `hf_quant_config.json` or the `quantization_config` of `config.json`) are read
by `fllm-rs convert` and converted automatically by `run`, `serve` and `pull`; llama.cpp's converter is not needed. Per quantized
linear the importer reads `weight` (U8, two E2M1 codes per byte, the even input in the low nibble), `weight_scale` (E4M3, one per
16 inputs), `weight_scale_2` and `input_scale`, and writes GGUF type 40: the block scale bytes are copied unchanged, the nibbles are
regrouped into the layout of the type (value `j` of a group of 16 in the low nibble of byte `j`, value `j + 8` in the high nibble),
`weight_scale_2` becomes the F32 tensor `<name>.scale` and `input_scale` the F32 tensor `<name>.input_scale` (the engine quantizes
activations itself and does not read it). The Llama Q/K row permutation moves whole rows, so codes and block scales stay together.
Excluded modules (`lm_head`, anything without `weight_scale`) keep their bf16/f16 weights; the FP8 KV-cache scales are dropped. A
weight whose input dimension is not a multiple of 64 is expanded to F16. Verified: tiny random Llama-like checkpoints in ModelOpt
layout run on the CPU with logits identical to a dense F32 model holding the weights dequantized with
`code * block scale * weight_scale_2` ([docs/rust-formats-and-import.md](rust-formats-and-import.md)). Not verified: a real ModelOpt
checkpoint, and any GPU run of a converted file.

## NVFP4 on GPUs without FP4 tensor cores

NVFP4 checkpoints (GGUF type 40) run through the same Marlin kernel as weight-only FP4 (W4A16, group 16, E4M3 scales) on every GPU from
Turing on that has no FP4 tensor cores (everything before Blackwell GeForce, sm_12x); `FLLM_NVFP4_MARLIN=1` forces it on Blackwell, which
is useful to compare. The upload converts the nibbles and scales like vLLM's `prepare_fp4_layer_for_marlin` (GPTQ repack kernel, scales in
the S0E5M3 form, per-tensor factor times 2^7). Before this change such GPUs expanded NVFP4 weights to FP16 (four times the memory).
Llama-3.1-8B NVFP4 on the RTX PRO 3000 (which has the FP4 cores): Marlin 4190 / 4260 tok/s prompt, 97 / 93 tok/s decode; the FP4 tensor-core
path 6400 / 7790 and 101 / 97. `selftest-gpu` checks the Marlin path with `FLLM_NVFP4_MARLIN=1` (error 1e-4 against the CPU reference).
