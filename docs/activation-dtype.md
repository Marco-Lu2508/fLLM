# Activation type of the CUDA pipeline (`FLLM_ACT_DTYPE`)

Where the CUDA backend keeps hidden states and intermediates, what is converted where, what a narrow pipeline (FP16 or BF16, like vLLM) saves, and the
policy that keeps models with large residual values safe. Test levels: everything described as "implemented" below compiles for sm_75 to sm_120 and has
`selftest-gpu --only actdtype` cases; nothing was run on a GPU by the author (see "Status").

## 1. Audit: the FP32 pipeline as it was

Shapes of the byte counts: Llama-3-8B (hidden 4096, 32 query heads and 8 key/value heads of 128, feed-forward 14336), 4-bit AWQ/GPTQ weights through
Marlin, f16 KV cache, FlashInfer for prompt attention. `M` rows. "Bytes" are read + write of the activation tensors by the kernel, per layer and row;
weights and the KV cache are not counted, cache hits are not modelled. Decode = fusions of `fused.cu` on (M < 16), prompt = the `FLLM_FUSE_PREFILL_*` paths.

| tensor | dtype today | producer | consumer(s) | conversion kernels around it |
|---|---|---|---|---|
| embeddings of the step | FP32 (host) | host (`embed_rows`), upload | residual stream `x` | none |
| residual stream `x` | FP32 | upload; `f16_to_f32_acc_kernel` or the add inside `add_rms_norm_kernel` / `rms_norm_prompt_kernel` (`x += scale * (y16 + bias)`, FP16 result of the output/down projection) | norm kernels; the add of the next projection | FP16 -> FP32 at every add (inside the norm in the fused paths) |
| norm output | FP32 always, plus an FP16 copy when the next GEMM is Marlin (`QUANT=1` / `OUT16`) | `add_rms_norm_kernel`, `rms_norm_vec_kernel`, `rms_norm_prompt_kernel` | q/k/v and gate/up GEMMs (read the FP16 copy), everything else reads FP32 | FP32 -> FP16 inside the norm; `f32_to_f16_kernel` when the copy is not made |
| q, k, v (Marlin) | FP16 (`q4z_f16out`) | Marlin | `rope_kv_f16_kernel` | none; q leaves as FP32 |
| q after rotation | FP32 | `rope_kv_f16_kernel` / `rope_kv_f32_kernel` / `rope_kernel` | attention | FP32 -> FP16 `float_to_half_kernel` inside the FlashInfer wrapper (prompt) |
| k, v into the KV cache | FP16 words (f16 cache) | the same kernel | attention | FP32 -> FP16 rounding in the write |
| attention output | FP32 | `attn4_kernel` (decode), FlashInfer through `half_to_float_(copy_)kernel` (prompt) | o_proj | prompt: `half_to_float_copy_kernel` also writes the FP16 copy; decode: `f32_to_f16_kernel` |
| o_proj / down result | FP16 | Marlin | residual add in the next norm | FP16 -> FP32 in the add |
| gate, up | FP16 | Marlin (`q4z_f16out`) | `glu_f16_to_f16_kernel` | none |
| gated activation | FP16 (folded into the conversion for the down projection) | `glu_f16_to_f16_kernel` | down projection | none (FP32 `fllm_glu` math inside) |
| final norm output | FP32 | norm | head GEMM | head: its own conversion |
| logits | FP32 | head GEMM | argmax / download | none (stay FP32 in every mode) |
| NVFP4 paths | FP32 in, quantized to NVFP4 inside the norm (`QUANT=2`) or by `nvfp4_quantize_kernel`; FP16 results on prompts (`nvfp4_f16out`) | | | `FLLM_NO_F16_NVFP4`, `FLLM_FUSE_NVFP4_ATTN_OUT`, `fllm_x_half` |
| K-quant paths | FP32 in, int8 per 32 values (`quantize_x8_kernel`, `QUANT=3`) | | | `FLLM_FUSE_GLU_Q8` |

Switches that exist for this today: `FLLM_NO_F16_FFN`, `FLLM_NO_F16_NVFP4`, `FLLM_NO_FUSED_QKV`, `FLLM_NO_FUSED_ADD`, `FLLM_FUSE`,
`FLLM_FUSE_ADD_NORM`, `FLLM_FUSE_NORM_QUANT`, `FLLM_FUSE_ROPE_KV`, `FLLM_FUSE_GLU_Q8`, `FLLM_FUSE_PREFILL_ADD_NORM`, `FLLM_FUSE_PREFILL_NORM_F16`,
`FLLM_FUSE_PREFILL_ATTN_F16` (docs/performance-notes.md), `fllm_attn_x16` (the thread-local that hands the FlashInfer FP16 result to the output projection).

### Bytes per layer and row (activation tensors only)

| path | per row | M = 1 | M = 2048 | x 32 layers, M = 2048 |
|---|---|---|---|---|
| FP32 pipeline, decode kernels | 466 944 B | 0.47 MB | 956 MB | 30.6 GB |
| FP32 pipeline, prompt kernels (FlashInfer) | 483 328 B | 0.48 MB | 990 MB | 31.7 GB |
| FP16/BF16 pipeline, decode, FP32 attention bridged (first version) | 417 792 B | 0.42 MB | 856 MB | 27.4 GB |
| FP16 pipeline, prompt, FlashInfer reads and writes FP16 | 352 256 B | 0.35 MB | 721 MB | 23.1 GB |
| BF16 pipeline, prompt (FlashInfer through two FP16 <-> BF16 conversions) | 385 024 B | 0.39 MB | 789 MB | 25.2 GB |

(The decode attention of the narrow pipeline runs `attn4_kernel` on the narrow arrays itself when the cache and the head shape allow it; the bridged
variant, widening q and narrowing the result around the FP32 kernels, is the fallback.) The line items behind the table:

| kernel | FP32 pipeline (B / row) | narrow pipeline (B / row) |
|---|---|---|
| residual add + norm for the attention block | 65 536 | 32 768 |
| q/k/v Marlin | 36 864 | 36 864 |
| rotation + cache write | 28 672 | 20 480 |
| attention (decode) | 32 768 + 24 576 (conversion for o_proj) | 16 384 |
| attention (prompt, FlashInfer) | 24 576 + 16 384 + 32 768 | 16 384 |
| o_proj Marlin | 16 384 | 16 384 |
| residual add + norm for the feed-forward block | 65 536 | 32 768 |
| gate/up Marlin | 73 728 | 73 728 |
| gated activation | 86 016 | 86 016 |
| down Marlin | 36 864 | 36 864 |

Reading of the numbers:
* At M = 1 the activations are 0.4 % of the 109 MB of weights a layer streams. Bandwidth is not what a narrow pipeline changes in decode; the launch
  count is. The FP32 decode path needs about 14 launches per layer after the fusions; the narrow decode path needs 13 to 14 when the attention kernel
  takes the narrow arrays and 16 with the bridge, so a narrow decode is not faster by itself (CUDA graphs already remove the launch gaps).
* At M = 2048 the activation traffic of a layer is about 1 GB; at 1.5 to 1.8 TB/s that is 0.55 to 0.65 ms per layer, 18 to 21 ms per 2048-token chunk
  for 32 layers where a chunk costs about 55 ms in total. The narrow pipeline removes 25 % of those bytes (the feed-forward intermediates, 196 of
  352 kB per row, are the same size in both and dominate; they are the target of the SwiGLU-into-quantization fusions of docs/changes/cuda-act.md).
  Expected effect: 3 to 5 ms of 55 ms per chunk (5 to 9 %); the measurement decides.

## 2. The narrow pipeline (implemented, behind `FLLM_ACT_DTYPE`)

`FLLM_ACT_DTYPE=f32|f16|bf16|auto`, default `f32`: the pipeline above, bit-identical to earlier versions (the narrow code is not reached).

What `f16` / `bf16` change (kernels in `cuda/act_narrow.cu`, entry points in `cuda/act_api.inc`, Rust side `backend/cuda.rs`, `backend/act_dtype.rs`):

* The device activation buffers (residual stream, norm outputs, q/k/v, attention output, gate/up/activation, GEMM results) hold FP16 or BF16. Buffers are
  still allocated with the FP32 size (the narrow values fill the first half); sizes are inferred from the buffer size in many places, and this keeps
  every shape computation of the decoder unchanged.
* Sums of squares, the residual add (`x = round(x + scale * branch)` is computed in FP32 registers, rounded once, stored), rotations, SiLU/GELU and softmax
  run in FP32 registers. The norm reads the *rounded* residual, so the fused add + norm equals the unfused kernels bit for bit.
* RMS norm, add + RMS norm, residual add, gated activation, rotary embedding, f16 KV-cache write and the fused rotation + cache write have narrow
  kernels. Marlin 4-bit GEMMs read and write the narrow arrays directly (no conversion kernels; the FP16 variant is the existing kernel, the BF16 variant
  is generated by `generate_kernels.py` and needs BF16 group scales, which `CudaBackend::upload_q4z` writes when BF16 is selected). The output projection
  and the down projection add into the residual stream inside the next norm (`quantized_linear_add_norm` runs Marlin into scratch and then the
  add + norm kernel).
* Attention: the decode kernel `attn4_kernel` loads q and stores the result in the narrow type (a runtime type code in its arguments); FlashInfer
  prompt attention is given the FP16 arrays directly (BF16: converted to FP16 and back; FlashInfer is not instantiated for BF16 and the KV cache is
  FP16 anyway). Cases the kernels do not cover (ALiBi, other head widths, quantized caches) widen q and narrow the result around the existing FP32
  kernels.
* The logits head, the hidden-state capture of the embeddings API and `argmax` stay FP32: after the final norm one conversion kernel widens the rows
  (`widen_activations`), `quantized_linear_wide` runs the head.
* Every operation without a narrow kernel (K-quant, NVFP4, FP8, Q4Z with activation permutation, other activation functions) runs on FP32 copies kept
  in persistent per-stream scratch (no allocation per call after the first pass): correct, but it keeps the conversions of today. NVFP4 and K-quant
  GEMMs therefore gain nothing yet.
* Models the narrow pipeline cannot run (recurrent layers, MoE, MLA, gated attention, LayerNorm, q/k/v clipping, parallel residual, layers that run on the
  host, MTP) stay FP32 with a line in the log, whatever was asked.
* CUDA graphs of decode steps work with it: the scratch is sized by the ordinary pass that precedes every capture and cannot grow while recording
  (an attempt returns an error and the graph is not used).

### Overflow policy

FP16 overflows above 65504; BF16 has the range of FP32 and 8 significant bits (unit roundoff 2^-8 instead of 2^-11).

* `FLLM_ACT_DTYPE=auto` picks BF16 when the embedding/output tensors of the file are BF16 (the GGUF does not record the `torch_dtype` of config.json,
  this stands in for it) and for the Gemma family; FP16 otherwise. A quantized embedding table hides the source dtype: such a model gets FP16 from
  `auto`, which is wrong for a BF16-trained model whose residual stream is large; use `FLLM_ACT_CHECK=1` once, or `bf16` explicitly.
* `FLLM_ACT_DTYPE=f16` on a BF16 file or a Gemma model is honoured with a warning.
* `FLLM_ACT_CHECK=1` inspects the residual stream after every layer of the first two forward passes and prints the largest value, the layer where it
  occurred, the headroom to the FP16 limit and the first layer with NaN/Inf. It works for every dtype (run it once with `f32` to learn the headroom of a
  model without risking anything) and synchronizes after each layer, so it is for diagnosis.
* The selftest group `actdtype` includes the NaN/Inf check kernel (`check_activations`) with injected values.
* Not done: automatic fall back to BF16 or FP32 in the middle of a run when an overflow appears. A hook for it would have to re-run the step; the policy
  above is meant to make it unnecessary.

## 3. Status

| level | what |
|---|---|
| L0 (compiles) | all kernels for sm_75 .. sm_120 (BF16 Marlin for sm_80 and newer), Rust for `--features cuda`; unit tests of the policy |
| L1 (selftest ran on a device) | none: written only, see docs/changes/cuda-act.md for the exact Colab run |

Tolerances of the `actdtype` cases (derived from the unit roundoff `u`: 2^-11 FP16, 2^-8 BF16): single rounded kernels 1.5 u of the error scale of
the element (value for products, sum of the magnitudes of the terms for rotation and add), add + norm and the rotated cache words 3 u, the Marlin GEMM
4e-3 (FP16) / 3 u (BF16, which includes the BF16 rounding of the group scales), a chain of two feed-forward blocks 12 u of the RMS of the result, KV
writes bit-identical. These are estimates; the first Colab run may show that one of them is too tight, which is then a finding about the kernel or
about the bound, to be decided from the printed worst element.

## 4. CUDA graphs for prompt chunks (`FLLM_CUDA_GRAPH_PREFILL`): not implemented

Asked for: piecewise capture per layer group at chunk buckets 128/256/512/1024/2048, default off. Not done, for these reasons; the first two decide whether it is worth doing.

* **What it could save.** The measured idle time of a prompt pass (27 % of the wall time before the host-side fixes of docs/performance-notes.md) was host work per layer
  (block-table compares, run detection, driver attribute calls), which the step cache removed. What is left is the gap between two consecutive small kernels: 1 to 2 us
  inside a graph, 3 to 5 us as stream launches, about 15 launches per layer, 32 layers: 0.5 to 1.5 ms per chunk against a chunk of about 55 ms. `FLLM_PDL=1` removes a part of
  that without a graph (sm_90+). Take an nsys timeline of a 2048-token chunk with `FLLM_PDL=0/1` first: if the GPU is busy more than 95 % of the chunk, a graph cannot help.
* **What it needs.** Attention (FlashInfer plan with two stream synchronisations, a per-chunk KV length) stays outside, so a chunk is 33 graph segments (one per layer
  boundary). Rows must be padded to the bucket; the padded rows need a null KV block so that the cache write is harmless and attention must be called with the real
  rows only (a prefix view of q and the output, the real block tables and lengths). Every segment needs its own staged position/slot arrays (the capture slots
  of `stage()` are the four arrays of a decode graph). The decoder loop must run its Rust side (scratch acquire/recycle sequence) once to capture and then skip
  to the next attention call on replay, handing the buffers the graph points to (q, k, v, the residual stream) to the attention call: `forward_backend_inner` is one
  function of about 800 lines in which these buffers are locals, so it has to be restructured into per-layer segments first.
* **What the narrow pipeline changes for it.** Nothing in `act_api.inc` allocates or synchronizes inside a recording (scratch growth is refused while recording, the
  Marlin partial/lock buffers are sized by the warm-up pass like for decode graphs), so the C side is graph-safe for segments too; the missing piece is the decoder.
