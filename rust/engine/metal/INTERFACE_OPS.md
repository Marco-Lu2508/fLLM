# Non-matmul Metal kernels — interface contract

Verification level: **L0**. The sources pass `mslcheck.sh` (a C++ syntax shim, not Apple's compiler), and — more than a syntax check — they were
**executed on the CPU** by the emulator in `emu_ops/` (below) against the engine's CPU backend: 139 cases, all within 3e-5 absolute / 3e-4 relative
(f32), byte-exact for the KV pools, the embedding tables and argmax. **No kernel has ever been compiled by Apple's Metal compiler or run on a GPU.**
The emulator replaces the GPU only for the *logic*: it cannot see address-space errors, register or threadgroup-memory limits, Metal API
rules that C++ does not have, fast-math effects, or performance.

The Rust side (`src/backend/metal/ops.rs`, `ops/`) plans the dispatches without touching Metal: every `plan_*` function turns the arguments of one
`Stream` operation into `OpPlan`s (kernel name, parameter bytes, grid, buffer slots). The stream resolves slot names to buffers,
uploads the host arrays, binds, dispatches and declares the accesses to the hazard tracker. The tests of `ops/tests.rs` compare every plan with the
MSL source (kernel exists, `[[buffer(n)]]` indices, fields and size of the parameter struct), so the two sides cannot drift apart silently.

## Conventions

- Files: `common_ops.metal` (no kernel, prepended to every unit by the backend: parameter structs, helpers, quantized-block decoders, KV accessors;
  **every symbol starts with `o_`**), one file per family: `norm_*`, `rope_*`, `act_*`, `misc_*`, `kv_*`, `embed_*`, `attn_*`. Kernel names are global:
  `fllm_<op>_<variant>`. Variants are separate kernel names (template instantiations with `[[host_name]]`), **no function constants**.
- A kernel binds the `constant` parameter struct at buffer **0** (`OpPlan::constants`, `CONSTANTS_BUFFER_INDEX`) and its buffers at `1..n` in the order of
  the tables below (`OpPlan::slots`). Parameter structs are made of 4-byte scalars, optionally led by 8-byte `ulong` fields; `Params` (ops.rs) writes the same layout, with `_pad` words
  at the end of structs that contain a `ulong`. Parameters travel with `setBytes` (< 4 KiB).
- Activations and outputs are f32, one row per token. `grid` below is `threadgroups`, `threads` is the threadgroup size (`dispatchThreadgroups`;
  Metal has no 65535 limit per dimension, so there is no x/y splitting of long grids).
- `SlotSource::Host` arrays (little endian) are small per-dispatch arrays: positions / slots / token ids `uint32[]`, `meta` = `int32` lengths
  followed by the block tables (`rows x stride`, `int32`), `bias` = `float` sinks[heads] then ALiBi slopes[heads] (zeros when absent). `meta` can exceed
  4 KiB, so the stream must upload it into a small arena buffer rather than `setBytes` it. `SlotSource::Dummy`: an absent optional input
  (weight, bias, up, ...): bind any buffer of 16 bytes. `SlotSource::Scratch { id, bytes }`: same id within the plans of one operation = one buffer.
- No kernel assumes anything about alignment beyond the element size, except where noted (the attention kernels need SIMD width 32; see Limits).
- The Rust types: `OpPlan`, `BufferSlot`, `Access`, `SlotSource`, `PlanConstant`, `Params`, `Slots`, `plan(..)` in `ops.rs` are shared with the recurrent-kernel planner.

## Kernels

| Stream operation | planner (`ops::`) | kernels |
|---|---|---|
| `rms_norm` | `plan_rms_norm`, `plan_rms_norm_strided` | `fllm_rms_norm_f32`, `fllm_rms_norm_f32x4` |
| `layer_norm` | `plan_layer_norm` | `fllm_layer_norm_f32`, `fllm_layer_norm_f32x4` |
| `rope` | `plan_rope` | `fllm_rope_f32` |
| `embedding_lookup` | `embedding::plan_embedding` | `fllm_embed_{f32,f16,bf16}`, `fllm_embed_{q4_0,q4_1,q5_0,q5_1,q8_0,q2_k,q3_k,q4_k,q5_k,q6_k,iq4_nl,mxfp4}` |
| `swiglu`, `geglu`, `swiglu_oai` (+ plain activations) | `plan_glu`, `plan_glu_strided` | `fllm_act_{swiglu,geglu,swiglu_oai,silu,gelu,relu2,relu}` |
| `residual_add` | `plan_residual_add` | `fllm_residual_add_f32` |
| `split_q_gate` | `plan_split_q_gate` | `fllm_split_q_gate_f32` |
| `sigmoid_gate_mul` | `plan_sigmoid_gate_mul` | `fllm_sigmoid_mul_f32` |
| `argmax_rows` | `plan_argmax` | `fllm_argmax_f32` |
| `kv_cache_write` | `attention::plan_kv_write` (2 dispatches: keys, values) | `fllm_kv_write_f16`, `fllm_kv_write_q8` |
| `attention_paged` | `attention::plan_attention` | `fllm_attn_decode_{f16,q8}_e{1,2,3,4,5,6,8,12,16,18}` (+ `fllm_attn_merge_f32`), `fllm_attn_prefill_{f16,q8}_k{DKF}v{DVF}` |
| `zero_buffer`, `copy_buffer` | — | blits of the command recorder, not kernels |

Not covered here: `quantized_linear` and the other matrix products (see `INTERFACE_QUANT.md`), `moe_*`, `gather_rows`, `scatter_add_rows`, `delta_layer`, Mamba (see `INTERFACE_RECURRENT.md`),
`sampling_logits` (the CPU path is used; `argmax_rows` covers greedy decoding), `rope_kv_write_f16` and the other fused FP16 paths (the stream returns
"not done" and the caller takes the unfused path). `split_q_gate`/`sigmoid_gate_mul` need `supports_attention_gate() == true`.

### Buffers, parameters, grids

| kernel | buffers (index order) | parameter struct | grid x threads |
|---|---|---|---|
| `fllm_rms_norm_f32[x4]` | `x`, `weight`, `out` | `o_norm_params` {cols, rows, x_stride, x_off, o_stride, o_off, eps, flags (bit 0: weight)} | `[rows,1,1]` x `[reduce_threads,1,1]` |
| `fllm_layer_norm_f32[x4]` | `x`, `weight`, `bias`, `out` | `o_norm_params` (bit 0: bias) | same |
| `fllm_rope_f32` | `q`, `k`, `positions`(host), `cos`, `sin` | `o_rope_params` {rows, heads, kv_heads, dim, half_dim, neox} | `[ceil(half/tx), heads+kv_heads, rows]` x `[tx,1,1]` |
| `fllm_act_*` | `gate`, `up`, `out` | `o_act_params` {n_rows, width, gate_stride, gate_off, up_stride, up_off, out_stride, alpha, limit} | `[ceil(width/tx), n_rows, 1]` x `[tx,1,1]` |
| `fllm_residual_add_f32`, `fllm_sigmoid_mul_f32` | `x`, `residual`/`gate` | `o_count_params` {count, scale} | `[ceil(count/256),1,1]` x `[256,1,1]` |
| `fllm_split_q_gate_f32` | `projected`, `q`, `gate` | `o_split_params` {rows, heads, dk, per_head} | `[ceil(dk/tx), heads, rows]` x `[tx,1,1]` |
| `fllm_argmax_f32` | `logits`, `out`(int32) | `o_argmax_params` {vocab, rows} | `[rows,1,1]` x `[pow2 32..512,1,1]` |
| `fllm_embed_*` | `weights`, `ids`(host), `out` | `o_embed_params` {n_ids, cols, row_bytes} | `[ceil(cols/tx), n_ids, 1]` x `[tx,1,1]` |
| `fllm_kv_write_f16` | `src`, `slots`(host), `pool` | `o_kvw_params` {base, scale_base, rows, kv, group, _pad} | `[ceil(kv/tx), rows, 1]` x `[tx,1,1]` |
| `fllm_kv_write_q8` | same | same | `[kv/group, rows, 1]` x `[32,1,1]` |
| `fllm_attn_decode_*` | `q`, `kpool`, `vpool`, `meta`(host), `bias`(host), `out`, `part`(scratch) | `o_attn_params` | `[nsplit, heads, rows]` x `[256,1,1]` |
| `fllm_attn_merge_f32` | `bias`(host), `out`, `part` | `o_merge_params` {heads, dv, nsplit, flags, row0} | `[heads, rows, 1]` x `[128,1,1]` |
| `fllm_attn_prefill_*` | `q`, `kpool`, `vpool`, `meta`(host), `bias`(host), `out` | `o_attn_params` | `[ceil(rows/32), heads, 1]` x `[128,1,1]` |

`tx` = `round_up(width, 32)` capped at 256; `reduce_threads(items)` = `round_up(items, 32)` in 32..=512. The exact definitions of all structs are in
`common_ops.metal`; the headers of the `.metal` files (`FLLM-KERNEL:` line) repeat the buffer lists.

Semantics (equal to the CPU backend unless noted):

- RMSNorm: `out = x * (1/sqrt(mean(x^2) + eps) * weight)`; without weight `weight = 1`. LayerNorm: biased variance, two passes. `out` may alias `x`
  (elementwise). The `x4` variants (cols % 4 == 0) read `packed_float4`, so any `x_off`/`x_stride` works on the GPU.
- RoPE: pairs `(d, d + half)` (neox) or `(2d, 2d + 1)`; `a' = a cos - b sin`, `b' = b cos + a sin`; dimensions from `rope_dim` on are untouched.
- Activations: SwiGLU `silu(g) u`; GeGLU `gelu_tanh(g) u` (tanh argument clamped to +-15); `swiglu_oai`: `min(g, limit) / (1 + exp(-alpha min(g, limit)))
  * (clamp(u, -limit, limit) + 1)`; plain `silu`, `gelu`, `relu`, `relu2` do not read `up`. Fused gate/up projections bind the same buffer to `gate` and `up`.
- Argmax: lowest index wins a tie, NaN never wins, all -inf/NaN row gives 0. Compares integer sort keys, so fast math cannot change it.
- Embedding: the table is in the **GGUF layout of the file** (float rows or rows of the block formats of `QuantType`); one thread decodes one element
  (`o_deq_*`, a transcription of `compute::dequantize`). If the quantized-linear path repacks its weights for the matrix kernels, the embedding table (which doubles as
  the tied LM head in many models) must stay available in the file layout, or the stream dequantizes rows on the host. Formats without a kernel
  (IQ1/IQ2/IQ3, IQ4_XS, Q8_K, NVFP4, Q4Z, FP8) return `UnsupportedQuant`.
- KV write and the layout of the pools: see below. q8 encoding is `d = amax/127`, `scale = half(d)`, `q = round_half_away(x / d)` clamped to +-127
  (`precise::divide`, like the CPU encoder, which divides by the f32 `d`, not by the rounded scale).

## KV pools (`kv_write_*`, `attn_*`)

The pools are bound as `device char *` (the whole `DeviceKvCache::key_pool` / `value_pool`) and addressed in 16-bit units; `base` is the layer start in
16-bit words (`layer * words_per_layer_*`, a `ulong` — pools larger than 4 GiB are fine, only single slot ids must fit 32 bits). With
`kv = kv_heads * dim`, `group = kv_quant_group(dim)` (32 if `dim % 32 == 0`, else `dim`), `gps = kv / group`:

- f16: value `(slot, c)` is the binary16 at element `base + slot * kv + c` of the `half` view.
- q8: value `(slot, c)` is the int8 at byte `2 * base + slot * kv + c` times the binary16 scale at element `base + scale_words + slot * gps + c / group`,
  `scale_words = ceil(slots * kv / 2)`. `k_gshift`/`v_gshift` = 5 when the group is 32, 0 when the group is the whole head (then the scale index is the kv head).

A layer may start at an odd word; a half pointer handles that, and byte/half stores to neighbouring locations by different threads need no atomics on
Metal (the Vulkan shaders need `atomicAnd/Or`). Slots of one `kv_cache_write` call are distinct (Stream contract).

## Attention planning

`plan_attention` validates the batch like `AttentionParams::validate` and splits the rows into **segments**:

- **prompt tiles** (`fllm_attn_prefill_*`): runs of up to 32 consecutive rows whose block tables are prefixes of one table (prefill passes one row per
  token of one sequence) and at least `prefill_min_rows` (8) rows; consecutive full tiles merge into one dispatch of `ceil(rows/32)` x `heads`
  threadgroups. Needs `simdgroup_matrix` (`AttentionTuning::simdgroup_matrix`) and head widths up to 256/256 (`PREFILL_FRAGMENTS`); otherwise the rows go
  to the decode kernel (correct, slower).
- **decode** (`fllm_attn_decode_*`): everything else, including continuous batches of different sequences. The key range of a (row, head) is split over
  `nsplit` threadgroups so that rows x heads x splits reaches `target_groups` (256), but not below `min_split_keys` (128) keys per split, at most 64
  (`MAX_SPLITS` = 256); `nsplit > 1` adds `fllm_attn_merge_f32`. `part` is scratch of `(2 + dv) * 4` bytes per (row, head, split).

A row attends to `start .. lengths[r] - 1`; `start = (n - 1) / chunk * chunk` with a chunk, `n - min(n, window)` with a window, else 0. Score =
`dot(q, k) * scale`, plus `alibi[h] * (pos - (n - 1))`, then `cap * tanh(score / cap)`; a sink adds `exp(sink)` to the denominator. Head widths up to 576
for keys and values (different widths allowed, GQA with any `heads % kv_heads == 0`).

Decode kernel (structure after MLX `sdpa_vector`, MIT): 8 simdgroups of 32 lanes per (row, head, split); simdgroup `g` handles the keys `lo + g, lo + g + 8, ...`;
a lane holds `ceil(dk/32)` consecutive channels of `q` and `ceil(dv/32)` of the accumulator; scores by `simd_sum`; online softmax per simdgroup; the eight
simdgroups are combined in threadgroup memory (`E` x 32 x 8 x 4 bytes = 18 KiB at most). Instantiated for `E = ceil(max(dk, dv)/32)` in {1,2,3,4,5,6,8,12,16,18}.

Prompt kernel: 4 simdgroups x 8 rows = 32 rows per threadgroup, keys in tiles of 32, `S = Q K^T` and `O += P V` with `simdgroup_matrix<float,8,8>` (Q fragments
in registers, K and V staged through threadgroup memory in 64-channel chunks), softmax on the lanes (4 lanes per row), per-row rescale of the accumulators by
multiplication with a diagonal 8x8 matrix (documented API only, no use of `thread_elements()`), the same trick for the final `1/l`. 15.5 KiB threadgroup memory,
`(DKF + DVF) * 2` registers for the fragments. Instantiated for (DKF, DVF) in (8,8) (10,10) (12,12) (14,14) (16,16) (24,16) (24,24) (32,32): widths 64, 80, 96, 112,
128, 192/128, 192, 256 (and everything in between by zero padding).

## Limits and assumptions the stream must respect

1. **SIMD width 32** for all attention kernels (`AttentionTuning::simd_width` must be `Pipeline::simd_width()`; the planner refuses anything else). The reduction
   kernels (norms, kv_write_q8) work for any width; they keep one partial per simdgroup in 32 entries (threads <= 512).
2. **Threadgroup sizes**: the plans assume 256 (decode), 128 (prompt), 128 (merge), up to 512 (norms, argmax) threads. A pipeline may report a smaller
   `maxTotalThreadsPerThreadgroup` when it uses many registers (the largest prompt instantiations are the candidates); the stream must check
   `Pipeline::max_threads()` and fall back to the decode kernel (`AttentionTuning { use_prefill_kernel: false, .. }`) if it is lower than `threads_per_group`.
3. `simdgroup_matrix` (prompt kernel): Apple7+/M1+ (`FLLM_HAS_SIMDGROUP_MATRIX`). Without it the source of `attn_prefill.metal` contains no kernel.
4. Fast math is on by default in the backend; the kernels were written to be insensitive (finite `o_neg = -1e30` instead of -inf, integer keys for argmax,
   clamped `tanh`/`exp` arguments). `FLLM_METAL_FAST_MATH=0` is the first thing to try if a number is off.
5. Bind `meta` with at least 4-byte alignment; `positions`, `slots`, `ids` are `uint32`, `out` of argmax is `int32`.

## The emulator (`emu_ops/`)

`emu_ops/run_emu.sh` compiles the real `.metal` sources as C++ against the syntax shim of `mslcheck.sh` plus `emu_ops/msl_emu.h`, which runs a threadgroup as one OS
thread per Metal thread with real barriers, `simd_sum/max/min/shuffle*/any/all` between 32 lanes and `simdgroup_matrix` (held replicated in every lane), and
executes the **plans of the Rust planner** (grid, threads, parameter bytes) over fixtures that the harness test `emu_fixtures` writes with the engine's CPU
backend as reference (`ops/mirror.rs`). Usage and what it can and cannot find: see the header of `emu_ops/msl_emu.h`.

## Known weaknesses (L0)

- Never compiled by Apple's compiler. Likely first findings of a hardware run: a Metal-only restriction the C++ shim does not know (address spaces in templates,
  attribute combinations), register spills in the largest prompt instantiations, a pipeline limit lower than the planned threadgroup size.
- Tuning constants (256/128 threads, 8 simdgroups, tile 32x32, 64-channel chunks, 256 groups per decode launch) are guesses; nothing was measured.
- The decode kernel does not share K/V reads between the query heads of a GQA group (every head re-reads its kv head; served by the caches); a kernel that scores
  all heads of a group per key row would cut the traffic by the group size.
- The prompt kernel reads Q fragments from threadgroup staging once and K/V as f32 tiles in threadgroup memory (half would double the tile size at the price of
  conversions); `simdgroup_matrix<half>` or the `mpp` tensor API of Metal 4 (M5) are not used.
- `sigmoid_gate_mul` etc. are one thread per element, no vectorization; for decode (one row) they are negligible, for prompt processing they are bandwidth-bound anyway.
- Precision: `simd_sum` adds in hardware order, `exp`/`tanh` are the fast versions with fast math; expect differences in the last bits against the CPU, enough to flip
  a greedy token in rare ties.
