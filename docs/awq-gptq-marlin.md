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

## Thread configuration for decode (`FLLM_MARLIN_CFG`)

`marlin_vllm.cu` is the host code of vLLM's `marlin.cu` (commit 9367d8b9, see THIRD_PARTY.md) and chooses the thread-block configuration
like vLLM does: for up to 16 rows (`thread_m_blocks == 1`) the list `{thread_k, thread_n, threads}` = `{128,128,256}`, `{64,128,128}`,
`{128,64,128}` in this order of priority (the first one that divides the shape, fits the shared memory and has a generated kernel), and
`{128,64,128}` instead whenever the grid would cover at most a quarter of the SMs; larger batches use `{64,256,256}`, `{64,128,128}`,
`{128,64,128}`. This was compared line by line with the vLLM source and with the call vLLM makes (`gptq_marlin_gemm`: thread_k = thread_n =
-1, FP32 reduce on, atomic add off): there is no difference to port, for sm_80, sm_89 and sm_120 alike (the 99 KB shared-memory limit
of sm_120 is read from the device like vLLM does). `FLLM_MARLIN_CFG=vllm|fllm|auto` therefore all mean the same thing, the default.

What decode looks like: a Qwen2.5-3B AWQ layer streams 38.5 MB of int4 weights in seven GEMMs (5.5 MB on average, 3.1 us at 1.79 TB/s);
the measured average is 12.5 us per call, so Marlin runs at about a quarter of the bandwidth for these small matrices and is 76 % of the
kernel time of a token. To find out whether another configuration is better for the shapes of a given model, `FLLM_MARLIN_CFG=K,N,T`
(for example `128,64,128`, `64,128,128`, `128,128,256`, `64,256,256`) makes every call try that configuration first and use it where it is valid
for the shape and a kernel for it was generated; otherwise vLLM's choice stays. The server prints one line when the variable is
accepted. Configurations split the sum over K differently, so the results can differ in the last digits; the selftest group `linear` (`fast linear` of
the q4z formats) is the check. Nothing is changed by default.

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

## NVFP4 on GPUs without FP4 tensor cores

NVFP4 checkpoints (GGUF type 40) run through the same Marlin kernel as weight-only FP4 (W4A16, group 16, E4M3 scales) on every GPU from
Turing on that has no FP4 tensor cores (everything before Blackwell GeForce, sm_12x); `FLLM_NVFP4_MARLIN=1` forces it on Blackwell, which
is useful to compare. The upload converts the nibbles and scales like vLLM's `prepare_fp4_layer_for_marlin` (GPTQ repack kernel, scales in
the S0E5M3 form, per-tensor factor times 2^7). Before this change such GPUs expanded NVFP4 weights to FP16 (four times the memory).
Llama-3.1-8B NVFP4 on the RTX PRO 3000 (which has the FP4 cores): Marlin 4190 / 4260 tok/s prompt, 97 / 93 tok/s decode; the FP4 tensor-core
path 6400 / 7790 and 101 / 97. `selftest-gpu` checks the Marlin path with `FLLM_NVFP4_MARLIN=1` (error 1e-4 against the CPU reference).
