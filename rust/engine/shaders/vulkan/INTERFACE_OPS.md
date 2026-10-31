# Non-matmul Vulkan shaders — interface contract

Verification level: **L0**. Every shader compiles with `glslc --target-env=vulkan1.2 -O` and passes `spirv-val --target-env vulkan1.2`
(84 modules, all variants of `variants_ops.list`). The Rust side
(`src/backend/vulkan/ops.rs`, `ops/`) is unit-tested on the CPU, and the trickiest shaders (KV write, attention decode / merge / prompt tiles, MoE
routing, DeltaNet, Mamba) were transcribed to Rust lane by lane and compared with the CPU backend (`ops/mirror.rs`; these transcriptions check the
algorithm and the index arithmetic, not the GLSL text). **No shader has ever run on a GPU.**

Everything here is dispatched through *plans*: `ops::plan_*` turns the arguments of one `Stream` operation into `OpPlan`s (shader name, push
constants, workgroup counts, bindings by name). The stream resolves the names to buffers, uploads the host arrays, binds, pushes,
dispatches and puts a barrier between two dependent plans. The reference below was extracted from the shader sources (their headers, comments, `layout(...)` lines and push blocks); the unit tests compare every plan with
the header and the push block of its shader, so the plans cannot drift from the sources (the prose of this file can).

## Conventions (the same as the quantized-linear shaders)

- GLSL 450, entry `main`, descriptor set 0, storage buffers `std430` numbered `0..n` in the order of the tables below, one `push_constant` block of
  32-bit scalars only (`uint`, `int`, `float`; offsets are 4 bytes apart; at most 128 bytes). Activations and outputs are f32 arrays, one row per token.
- Device requirements: Vulkan 1.2 core, no optional shader feature (no 8/16-bit storage, no float16/int64, no subgroup operations, no atomics other than
  `uint` atomicAnd/atomicOr on storage buffers). Exceptions: the `moe_linear_*_bda` variants need `bufferDeviceAddress`
  (`SPV_KHR_physical_storage_buffer`, `GL_EXT_buffer_reference_uvec2`, compiled with glslc 2023.8). Limits used: `maxComputeWorkGroupInvocations >= 256`,
  `maxComputeSharedMemorySize >= 16384` (largest user: `attn_prefill`, 13.4 KiB), `maxPerStageDescriptorStorageBuffers >= 9` (`delta_rec`; the
  Vulkan minimum of 4 is not enough, every desktop and current mobile GPU offers 16 or more), `maxComputeWorkGroupCount >= 65535` per dimension.
- Storage buffers are bound at offsets that are multiples of `minStorageBufferOffsetAlignment`; `moe_linear` reads `x` as `vec4` (16-byte aligned).
  Weight and KV-pool word indices are 32 bit: a bound window of a weight or of a KV pool may not exceed 4 GiB (`plan_*` returns `UnsupportedOperation`
  otherwise; bind a sub-window and `KvLayout::rebased`).
- One workgroup per row for the reductions (`norm_*`, `misc_argmax`, `moe_topk`); logical grids larger than 65535 are split over x and y and the
  shaders recover the index with `x + y * numGroupsX` (`group_index()`); element-wise shaders use a grid-stride loop and a grid capped at 65535.
- Host arrays that travel with a dispatch (little endian; `SlotSource::Host`): positions `uint32[]`, KV slots `uint32[]`, token ids `uint32[]`,
  `meta` = `int32` lengths followed by the block tables (`rows x stride`, `int32`), `bias` = `float` sinks[heads] then ALiBi slopes[heads] (zeros when
  absent), gather / scatter indices `int32[]` and weights `float[]`. `ops::pack_u32 / pack_i32 / pack_f32` build them.
- `SlotSource::Dummy`: an optional input that is absent (bias, weight, sinks) still needs a valid descriptor: bind any storage buffer of 16 bytes.
  `SlotSource::Scratch { id, bytes }`: same id within the plans of one operation = one buffer (size = the maximum). `SlotSource::RunState(i)`: the
  state buffer of run `i`.

## Stream operation → plan → shaders

| Stream operation | plan function (`ops::`) | shaders |
|---|---|---|
| `rms_norm` | `plan_rms_norm`, `plan_rms_norm_strided` | `norm_rms` |
| `rms_norm_grouped` | `plan_rms_norm_grouped` | `norm_rms_grouped` |
| `layer_norm` | `plan_layer_norm` | `norm_layer` |
| `rope` | `plan_rope` | `rope_apply` |
| `embedding_lookup` | `plan_embedding` | `embed_{f32,f16,bf16,q4_0,q4_1,q5_0,q5_1,q8_0,q2_k,q3_k,q4_k,q5_k,q6_k,iq4_nl,mxfp4}` |
| `swiglu`, `geglu`, `swiglu_oai` (+ plain activations) | `plan_glu`, `plan_glu_strided` | `act_{swiglu,geglu,swiglu_oai,silu,gelu,relu2,relu}` |
| `residual_add` | `plan_residual_add` | `misc_residual` |
| `split_q_gate` | `plan_split_q_gate` | `misc_split_q_gate` |
| `sigmoid_gate_mul` | `plan_sigmoid_gate_mul` | `act_sigmoid_mul` |
| `head_gate_mul` (per-head gate, sigmoid or softplus) | `plan_head_gate` | `act_head_gate_{sigmoid,softplus}` |
| `scale_inplace` | `plan_scale` | `ewise_scale` |
| `argmax_rows` | `plan_argmax` | `misc_argmax` |
| `kv_cache_write` | `plan_kv_write` (2 dispatches: keys, values) | `kv_write_f16`, `kv_write_q8` |
| `attention_paged` | `plan_attention` | `attn_decode_{f16,q8}` (+ `attn_merge`), `attn_prefill_{f16,q8}_d{128,256,512}` |
| `dsa_select` (lightning indexer of DeepSeek-V3.2 / GLM-5: key write, scores, top-k; chunks of rows) | `dsa::plan_dsa_select` | `misc_dsa_keys`, `misc_dsa_scores`, `misc_dsa_topk` |
| `dsa_attention` (attention over the selected positions, f16 cache) | `dsa::plan_dsa_attention` | `attn_sparse` |
| `rotate_heads` (tail rotation of the Hy4 indexer heads) | `hc::plan_rotate_heads` | `misc_rotate_heads` |
| `hc_mix`, `hc_combine` (hyper-connections of Hy4 / GLM-5-Next) | `hc::plan_hc_mix`, `hc::plan_hc_combine` | `misc_hc_mix`, `misc_hc_combine` |
| `moe_topk`, `moe_route` | `plan_moe_topk`, `plan_moe_route` | `moe_topk` |
| `moe_linear` | `plan_moe_linear` | `moe_linear_<fmt>_{arena,bda}` (12 quantized formats) |
| `moe_combine` | `plan_moe_combine` | `moe_combine` |
| `moe_expert_gemm` | `plan_moe_expert_gemm` (= `moe_linear` + `moe_combine`) | as above |
| `gather_rows`, `scatter_add_rows` | `plan_gather_rows`, `plan_scatter_add_rows` | `moe_gather_rows`, `moe_scatter_add_rows` |
| `delta_layer` | `delta_supported`, `plan_delta_layer` | `delta_conv`, `delta_rec_hk{32,64,96,128}` |
| `kda_layer` (Kimi Delta Attention) | `kda_supported`, `plan_kda_layer` | `delta_conv` (same convolution, `k_heads = v_heads`), `kda_rec_hk{16,32,64,128}` (`kda_rec.comp`, bindings and push block in its `FLLM-SHADER` header) |
| `copy_columns` | `plan_copy_cols` | `misc_copy_cols` |
| `shortconv_layer` (LFM2) | `shortconv_supported`, `plan_shortconv` | `shortconv` |
| `rwkv_shift`, `rwkv_lerp`, `rwkv_act` | `plan_rwkv_shift`, `plan_rwkv_lerp`, `plan_rwkv_act` | `rwkv_shift`, `rwkv_lerp`, `rwkv_act` |
| `rwkv7_wkv`, `rwkv6_wkv` | `rwkv_supported`, `plan_rwkv7_wkv`, `plan_rwkv6_wkv` | `rwkv7_wkv_n{16,32,64,128}`, `rwkv6_wkv_n{16,32,64,128}` (15 / 10 storage buffers; bindings, push blocks and the formulas in the `FLLM-SHADER` headers of `rwkv7_wkv.comp` / `rwkv6_wkv.comp`) |
| *(no Stream method yet)* Mamba-2 mixer | `plan_mamba2_mixer` | `mamba_conv`, `mamba_scan2_{ns16,ns32,ns64,ns128,gen}`, `mamba_gate_norm` |
| *(no Stream method yet)* Mamba-1 mixer | `plan_mamba_conv`, `plan_mamba1_bc_norms`, `plan_mamba1_scan` | `mamba_conv`, `norm_rms`, `mamba_scan1_{ns8,ns16,ns32,gen}` |
| `zero_buffer`, `copy_buffer` | *(not a shader)* | `vkCmdFillBuffer` / `vkCmdCopyBuffer` (`Recorder::fill`, `copy_buffer`) |

Not covered by a shader: `quantized_linear` and the other matrix products (`INTERFACE_QUANT.md`), `sampling_logits` (the CPU backend path is used; `argmax_rows` covers
greedy decoding), `rope_kv_write_f16` and the other fused FP16 paths (they return "not done" and the caller takes the unfused path).

## Weight layouts that `embed_*` and `moe_linear_*` expect

The embedding table and the expert weights are `DeviceWeight`s that the stream also uses for `quantized_linear` (tied embeddings share the LM-head
weight), so they are in the **planar device layout of the quantized-linear shaders** (`INTERFACE_QUANT.md`, `weights/layout.rs`): the planes of a weight, each at a word
offset, plane 0 at word 0. The decoders in `_ops_wq.glsl` repeat that layout (they are not shared with the quantized-linear shader files, so that the op shaders build
alone); the plane word offsets (`DeviceLayout::plane_word_offset[1..4]`) are passed in the push constants (`off1..off3`; `EmbeddingTable::planes`,
`MoeLinear::planes`). F32 / F16 / BF16 tables are stored row-major as in the file (`embed_f32/f16/bf16`; the quantized-linear shaders have none for those weights, so
the weight uploader decides how they are uploaded; if it packs them differently, the embed shaders must follow). Formats without an `embed_*` shader (IQ1/IQ2/IQ3, IQ4_XS, Q8_K,
NVFP4, Q4Z, FP8) return `UnsupportedQuant`: the stream dequantizes the rows on the host like the CPU backend does and uploads them.

## `moe_linear`: how the experts are reached

The Stream contract passes `table` = "the device addresses of the experts' weights" (`upload_pointer_table`). The Vulkan core does not offer buffer device
addresses, so there are two variants and the stream chooses (`ExpertAccess`):

- `Arena` (`moe_linear_<fmt>_arena`): the experts of a projection are laid out back to back in **one storage buffer** (each in its planar layout, starting
  at a word offset that is a multiple of 4); `table` is `uint32[n_experts]` of word offsets, biases (optional) are `float`s in a second buffer
  (`bias_table` = float offsets). A bound range is limited by `maxStorageBufferRange` (a projection of 128 experts of 768 x 2048 Q4_K is ~113 MiB).
  `upload_pointer_table` therefore has to copy the experts into an arena (or `upload_weight` has to carve the experts of a projection out of one).
- `BufferAddress` (`moe_linear_<fmt>_bda`): `table[e]` = 64-bit device address (`uvec2`, low word first) of expert `e`'s weight buffer, exactly like the CUDA
  backend. Needs `bufferDeviceAddress` at device creation and `SHADER_DEVICE_ADDRESS` usage on the weight allocations; `weights`/`biases` are dummies.

An expert id outside `0..n_experts` gives zeros (the CPU backend skips such experts).

## KV pools (`attn_*`, `kv_write_*`)

The pools are bound as `uint[]` (whole pool or a window) and addressed in 16-bit units: f16 value `(slot, c)` is the binary16 at `base + slot * kv + c`; q8
value `(slot, c)` is the int8 at byte `2 * base + slot * kv + c` times the binary16 scale at `base + scale_base + slot * (kv / group) + c / group`, with
`kv = kv_heads * dim`, `group = kv_quant_group(dim)` and `scale_base = ceil(slots * kv / 2)` (all as documented at `DeviceKvCache`). `base` is the layer start
in 16-bit words relative to the bound window (`layer * words_per_layer_*` for a pool bound as a whole). A layer may start at an odd word (odd `kv`, or q8
layers whose scale area has an odd size): every access goes through 32-bit words and a lane select, and the writers merge half-owned words with
`atomicAnd`/`atomicOr` (`kv_write_*`). Only distinct slots per call are allowed (Stream contract); the neighbours of a written slot are never disturbed.
The KV format is hidden behind `KvLayout` (plans) and the two macros of `_ops_kv.glsl`; another cache format means another `_ops_kv.glsl` accessor, not new
planning.

## Attention planning

`plan_attention` validates the batch like `AttentionParams::validate` and splits the rows into **segments**:

- **prompt tiles** (`attn_prefill_*`): runs of up to 16 consecutive rows whose block tables are prefixes of one table (prefill passes one row per token of one
  sequence) and at least `prefill_min_rows` (4) rows; consecutive tiles are merged into one dispatch of `ceil(rows / 16)` x `heads` workgroups. The tile kernel
  keeps a 16-row block in registers and reads each key / value row once per block.
- **decode** (`attn_decode_*`): everything else. The key range of a (row, head) is split over `nsplit` workgroups so that rows x heads x splits reaches
  `target_groups` (512), but not below `min_split_keys` (256) keys per split, at most `max_splits` (64); `nsplit > 1` adds an `attn_merge` dispatch.

A row attends to `start .. lengths[r] - 1`; `start = (n - 1) / chunk * chunk` with a chunk, `n - min(n, window)` with a window, else 0 (`AttentionParams`).
Score = `dot(q, k) * scale`, plus `alibi[h] * (pos - (n - 1))`, then `cap * tanh(score / cap)`; a sink adds `exp(sink)` to the denominator. Head widths up to 512 for
keys and values (different widths allowed), GQA with any `heads % kv_heads == 0`.

## State layouts of the recurrent layers

- **DeltaNet** (`DeltaShape`, `delta_layer`): in the state buffer of a sequence, the layer's state starts at `state_offset` bytes: the recurrent matrices
  `S[value_head][key_row][value_col]` as f32 (`rec_bytes()`), then the convolution history `history[channel][k]` (`conv_dim x (kernel - 1)`, the k-th last
  input, oldest first). This is the layout of the CUDA backend; the CPU decoder uses its own and the two are never mixed (a state buffer belongs to one backend).
- **Mamba** (`Mamba2Shape`, `Mamba1Shape`): history `[channel][k]` first, then the recurrent state (`[head][channel][state]` for Mamba-2, `[channel][state]` for
  Mamba-1), `layer_state_bytes()` rounded to 256. The offsets are parameters of the plans.

## Numerics

f32 everywhere except the f16 / q8 KV pools. Reductions run in shared memory in a fixed tree order (not the order of the CPU); `exp`, `tanh`, `log` and division
are the Vulkan-precision functions (2.5 ULP or better; `tanh` arguments are clamped to +-15). `packHalf2x16` rounds like the driver does (round to nearest even
on every implementation the author knows; the spec leaves the mode open): a KV value may differ from the CPU backend in the last half-precision bit on a driver
that truncates. The q8 encoder divides `x / d` like the CPU encoder; a value exactly halfway between two steps may land on the other neighbour (division is not
correctly rounded).

## Known weaknesses (L0)

- No shader has run. Tuning constants (workgroups of 64 / 128 / 256, tile 16 x 32, splits) are guesses.
- GQA groups are not shared in the decode kernel: every query head of a group re-reads the key / value rows (served by the caches). A kernel that scores all heads
  of a group per key row would cut the traffic by the group size.
- `moe_linear` decodes 8-weight chunks with the generic planar decoders and does no tensor-core / cooperative-matrix or integer-dot work; it is a matrix-vector
  kernel (one workgroup per output row and slot), not a grouped GEMM: prompt processing of MoE layers should group tokens per expert (`gather_rows` + the
  quantized linear shaders + `scatter_add_rows`) as the CPU decoder does.
- The DeltaNet recurrence keeps one value column per lane pair with HK/2 registers each (`HK` 64 / 128: 32 / 64 registers); a long run of one head is serial by nature.
- Mamba has no `Stream` method: a trait method is not yet defined; until then the plans are only exercised by the unit tests.

---

# Reference of the shaders (extracted from the sources)

### `norm_rms.comp`

Compiled as `norm_rms`.

```text
RMSNorm (Stream::rms_norm): out[r][c] = x[r][c] * (1 / sqrt(mean(x[r]^2) + eps) * weight[c]); one workgroup per row.
x_stride/o_stride: floats between rows, x_off/o_off: first float of row 0 (so a slice of a wider row, e.g. Mamba B/C, can be normalised).
flags bit 0: weight present (otherwise weight is not read and 1 is used). `out` may be the same buffer range as `x`.
```

| binding | name | access | type |
|---|---|---|---|
| 0 | `x` | readonly | `float x[];` |
| 1 | `weight` | readonly | `float weight[];` |
| 2 | `out` | read/write | `float o[];` |

| push offset | type | field |
|---|---|---|
| 0 | `uint` | `cols` |
| 4 | `uint` | `rows` |
| 8 | `uint` | `x_stride` |
| 12 | `uint` | `x_off` |
| 16 | `uint` | `o_stride` |
| 20 | `uint` | `o_off` |
| 24 | `float` | `eps` |
| 28 | `uint` | `flags` |

Push block: 32 bytes. Grid: `[rows split over x/y (x <= 65535), 1, 1]`.


### `norm_rms_grouped.comp`

Compiled as `norm_rms_grouped`.

```text
Grouped RMSNorm (Stream::rms_norm_grouped): row is split into groups of group_size floats.
One workgroup per (row, group). out[r*cols_total + g*group_size + i] = x[...] * (1 / sqrt(mean(group^2) + eps) * weight[g*group_size + i]).
flags bit 0: weight present (otherwise weight is not read and 1 is used). out may be the same buffer range as x.
```

| binding | name | access | type |
|---|---|---|---|
| 0 | `x` | readonly | `float x[];` |
| 1 | `weight` | readonly | `float weight[];` |
| 2 | `out` | read/write | `float o[];` |

| push offset | type | field |
|---|---|---|
| 0 | `uint` | `group_size` |
| 4 | `uint` | `groups` |
| 8 | `uint` | `total_groups` |
| 12 | `uint` | `cols_total` |
| 16 | `float` | `eps` |
| 20 | `uint` | `flags` |

### `norm_layer.comp`

Compiled as `norm_layer`.

```text
LayerNorm (Stream::layer_norm): out = (x - mean) * (1 / sqrt(var + eps)) * weight + bias with the biased variance (CPU backend); one workgroup per row.
flags bit 0: bias present (otherwise bias is not read). `out` may be the same buffer range as `x`.
```

| binding | name | access | type |
|---|---|---|---|
| 0 | `x` | readonly | `float x[];` |
| 1 | `weight` | readonly | `float weight[];` |
| 2 | `bias` | readonly | `float bias[];` |
| 3 | `out` | read/write | `float o[];` |

| push offset | type | field |
|---|---|---|
| 0 | `uint` | `cols` |
| 4 | `uint` | `rows` |
| 8 | `uint` | `x_stride` |
| 12 | `uint` | `x_off` |
| 16 | `uint` | `o_stride` |
| 20 | `uint` | `o_off` |
| 24 | `float` | `eps` |
| 28 | `uint` | `flags` |

Push block: 32 bytes. Grid: `[rows split over x/y (x <= 65535), 1, 1]`.

### `rope_apply.comp`

Compiled as `rope_apply`.

```text
Rotary embedding in place (Stream::rope). Row r has position positions[r]; cos/sin are `positions x half_dim` tables (half_dim = rope_dim / 2).
One invocation rotates one pair of one head: (d, d + half_dim) for NeoX, (2d, 2d + 1) otherwise; query heads first, then key heads. The
dimensions of a head from rope_dim on (partial rotary) are not touched. q is `rows x heads x dim`, k is `rows x kv_heads x dim`.
```

| binding | name | access | type |
|---|---|---|---|
| 0 | `q` | read/write | `float q[];` |
| 1 | `k` | read/write | `float k[];` |
| 2 | `positions` | readonly | `uint pos[];` |
| 3 | `cos` | readonly | `float cs[];` |
| 4 | `sin` | readonly | `float sn[];` |

| push offset | type | field |
|---|---|---|
| 0 | `uint` | `rows` |
| 4 | `uint` | `heads` |
| 8 | `uint` | `kv_heads` |
| 12 | `uint` | `dim` |
| 16 | `uint` | `half_dim` |
| 20 | `uint` | `neox` |

Push block: 24 bytes. Grid: `[min(ceil(rows*(heads+kv_heads)*half_dim/256), 65535), 1, 1]`.

### `embed.comp`

Compiled as:

- `embed_f32` (`-DRAW_F32`)
- `embed_f16` (`-DRAW_F16`)
- `embed_bf16` (`-DRAW_BF16`)
- `embed_q4_0` (`-DFMT_Q4_0`)
- `embed_q4_1` (`-DFMT_Q4_1`)
- `embed_q5_0` (`-DFMT_Q5_0`)
- `embed_q5_1` (`-DFMT_Q5_1`)
- `embed_q8_0` (`-DFMT_Q8_0`)
- `embed_q2_k` (`-DFMT_Q2_K`)
- `embed_q3_k` (`-DFMT_Q3_K`)
- `embed_q4_k` (`-DFMT_Q4_K`)
- `embed_q5_k` (`-DFMT_Q5_K`)
- `embed_q6_k` (`-DFMT_Q6_K`)
- `embed_iq4_nl` (`-DFMT_IQ4_NL`)
- `embed_mxfp4` (`-DFMT_MXFP4`)

```text
Embedding lookup (Stream::embedding_lookup): out[t][c] = weight[ids[t]][c] for t < n_ids, c < cols (= hidden size); ids are token ids (uint32, checked
< vocab on the host). The format is selected by one macro per entry of variants_ops.list:
  RAW_F32 / RAW_F16 / RAW_BF16   row-major table of f32 / binary16 / bfloat16, element (row, c) at index row * cols + c
  FMT_Q4_0 FMT_Q4_1 FMT_Q5_0 FMT_Q5_1 FMT_Q8_0 FMT_Q2_K FMT_Q3_K FMT_Q4_K FMT_Q5_K FMT_Q6_K FMT_IQ4_NL FMT_MXFP4
                                 planar device layout of the quantized weights (see _ops_wq.glsl); nb = blocks per row = cols / QK, off1..off3 = word
                                 offsets of the planes 1..3. One invocation decodes one chunk of 8 weights (cols is a multiple of QK, so of 8).
Table rows must lie inside the bound range (word indices are 32-bit: tables up to 4 GiB / 16 GiB of words).
```

| binding | name | access | type |
|---|---|---|---|
| 0 | `weights` | readonly | `uint wbuf[];` |
| 1 | `ids` | readonly | `uint ids[];` |
| 2 | `out` | read/write | `float o[];` |

| push offset | type | field |
|---|---|---|
| 0 | `uint` | `n_ids` |
| 4 | `uint` | `cols` |
| 8 | `uint` | `nb` |
| 12 | `uint` | `off1` |
| 16 | `uint` | `off2` |
| 20 | `uint` | `off3` |

Push block: 24 bytes. Grid: `[min(ceil(work/256), 65535), 1, 1]  (work = n_ids * cols / 8 for block formats, n_ids * cols for RAW_*)`.

### `act_glu.comp`

Compiled as:

- `act_swiglu` (`-DMODE=0`)
- `act_geglu` (`-DMODE=1`)
- `act_swiglu_oai` (`-DMODE=2`)
- `act_silu` (`-DMODE=3`)
- `act_gelu` (`-DMODE=4`)
- `act_relu2` (`-DMODE=5`)
- `act_relu` (`-DMODE=6`)

```text
Gated and plain activations, selected by the macro MODE (one output per entry of variants_ops.list):
  0 SWIGLU      out = silu(g) * u                                      (Stream::swiglu)
  1 GEGLU       out = gelu_tanh(g) * u                                 (Stream::geglu; CPU: 0.5 g (1 + tanh(..)) u)
  2 SWIGLU_OAI  g' = min(g, limit), u' = clamp(u, -limit, limit), out = g' / (1 + exp(-alpha g')) * (u' + 1)   (Stream::swiglu_oai)
  3 SILU        out = silu(g)               (u is not read)
  4 GELU        out = gelu_tanh(g)          (u is not read)
  5 RELU2       out = max(g, 0)^2           (u is not read)
  6 RELU        out = max(g, 0)             (u is not read)
Element (r, c): gate[gate_off + r * gate_stride + c], up[up_off + r * up_stride + c], out[r * out_stride + c], r < n_rows, c < width. Gate and up may
be two ranges of the same buffer (fused gate/up projection). Grid-stride loop over n_rows * width elements.
```

| binding | name | access | type |
|---|---|---|---|
| 0 | `gate` | readonly | `float gate[];` |
| 1 | `up` | readonly | `float up[];` |
| 2 | `out` | read/write | `float o[];` |

| push offset | type | field |
|---|---|---|
| 0 | `uint` | `n_rows` |
| 4 | `uint` | `width` |
| 8 | `uint` | `gate_stride` |
| 12 | `uint` | `gate_off` |
| 16 | `uint` | `up_stride` |
| 20 | `uint` | `up_off` |
| 24 | `uint` | `out_stride` |
| 28 | `float` | `alpha` |
| 32 | `float` | `limit` |

Push block: 36 bytes. Grid: `[min(ceil(n_rows*width/256), 65535), 1, 1]`.

### `act_sigmoid_mul.comp`

Compiled as `act_sigmoid_mul`.

```text
Gated attention output (Stream::sigmoid_gate_mul): x[i] = x[i] * sigmoid(gate[i]) in place.
```

| binding | name | access | type |
|---|---|---|---|
| 0 | `x` | read/write | `float x[];` |
| 1 | `gate` | readonly | `float gate[];` |

| push offset | type | field |
|---|---|---|
| 0 | `uint` | `count` |

Push block: 4 bytes. Grid: `[min(ceil(count/256), 65535), 1, 1]`.

### `act_head_gate.comp`

Compiled as `act_head_gate_sigmoid` (`MODE=0`) and `act_head_gate_softplus` (`MODE=1`).

```text
Per-head attention output gate (Stream::head_gate_mul, Laguna and Step-3.5): x[i] = x[i] * act(gate[i / dv]) in place, one gate value per
head and row, dv output values per head. MODE 0: act(g) = sigmoid(g); MODE 1: act(g) = softplus(beta * g) / beta (the gate itself above scaled 20).
```

| binding | name | access | type |
|---|---|---|---|
| 0 | `x` | read/write | `float x[];` |
| 1 | `gate` | readonly | `float gate[];` |

| push offset | type | field |
|---|---|---|
| 0 | `uint` | `count` |
| 4 | `uint` | `dv` |
| 8 | `float` | `beta` |

Push block: 12 bytes. Grid: `[min(ceil(count/256), 65535), 1, 1]`. `ewise.comp` gained `MODE=2` (`ewise_scale`: `x = x * param`, `Stream::scale_inplace`).

### `misc_residual.comp`

Compiled as `misc_residual`.

```text
Stream::residual_add: x[i] += residual[i] * scale in place (grid-stride loop over count floats).
```

| binding | name | access | type |
|---|---|---|---|
| 0 | `x` | read/write | `float x[];` |
| 1 | `residual` | readonly | `float res[];` |

| push offset | type | field |
|---|---|---|
| 0 | `uint` | `count` |
| 4 | `float` | `scale` |

Push block: 8 bytes. Grid: `[min(ceil(count/256), 65535), 1, 1]`.

### `misc_split_q_gate.comp`

Compiled as `misc_split_q_gate`.

```text
Gated attention (Stream::split_q_gate): the query projection holds query and gate of every head. Row r of `projected` has 2 * heads * dk floats:
  per_head != 0: [q_h0 (dk), gate_h0 (dk), q_h1, gate_h1, ...]     otherwise: [q_h0 .. q_h(heads-1), gate_h0 .. gate_h(heads-1)].
q and gate are `rows x heads x dk`; total = rows * heads * dk. One invocation per output element.
```

| binding | name | access | type |
|---|---|---|---|
| 0 | `projected` | readonly | `float projected[];` |
| 1 | `q` | read/write | `float q[];` |
| 2 | `gate` | read/write | `float gate[];` |

| push offset | type | field |
|---|---|---|
| 0 | `uint` | `total` |
| 4 | `uint` | `heads` |
| 8 | `uint` | `dk` |
| 12 | `uint` | `per_head` |

Push block: 16 bytes. Grid: `[min(ceil(total/256), 65535), 1, 1]`.

### `misc_argmax.comp`

Compiled as `misc_argmax`.

```text
Greedy sampling (Stream::argmax_rows): out[r] = index of the largest of the `vocab` logits of row r (int32); the lowest index wins a tie and a NaN
never wins (comparisons use `>`); an all -inf / all NaN row gives 0 (the CPU `sampling_logits` behaves the same). One workgroup per row.
```

| binding | name | access | type |
|---|---|---|---|
| 0 | `logits` | readonly | `float logits[];` |
| 1 | `out` | read/write | `int best_index[];` |

| push offset | type | field |
|---|---|---|
| 0 | `uint` | `vocab` |
| 4 | `uint` | `rows` |

Push block: 8 bytes. Grid: `[rows split over x/y (x <= 65535), 1, 1]`.

### `kv_write_f16.comp`

Compiled as `kv_write_f16`.

```text
Stream::kv_cache_write for an f16 cache (kv_kind 1), one pool (keys or values) per dispatch. Row r of `src` (kv = kv_heads * dim floats) is stored as
kv binary16 values at 16-bit positions  base + slots[r] * kv + c  of the pool (layout of DeviceKvCache: base = layer * words_per_layer, or the
layer start relative to the bound window). The pool is addressed in 32-bit words; one invocation owns one word:
  * both halves belong to the row: plain store; * only one half does (the row starts or ends in the middle of a word, e.g. an odd layer
  start or an odd kv): atomicAnd/atomicOr of that half, so neighbouring rows written by other invocations are not disturbed.
Slots of one call are distinct (Stream contract). Work items: rows * (kv/2 + 1) (an upper bound of the words of one row; the rest exits).
```

| binding | name | access | type |
|---|---|---|---|
| 0 | `src` | readonly | `float src[];` |
| 1 | `slots` | readonly | `uint slots[];` |
| 2 | `pool` | read/write | `uint pool[];` |

| push offset | type | field |
|---|---|---|
| 0 | `uint` | `rows` |
| 4 | `uint` | `kv` |
| 8 | `uint` | `base` |

Push block: 12 bytes. Grid: `[min(ceil(rows*(kv/2+1)/256), 65535), 1, 1]`.

### `kv_write_q8.comp`

Compiled as `kv_write_q8`.

```text
Stream::kv_cache_write for a q8_0 cache (kv_kind 2), one pool (keys or values) per dispatch. Layout of DeviceKvCache (all positions in 16-bit units
from the start of the layer, `base` = layer start in the bound pool): kv signed bytes per slot (byte slot * kv + c, little endian inside the
16-bit words), then slots * (kv / group) binary16 scales (scale slot * (kv / group) + c / group) starting `scale_base` 16-bit words after the layer
start (scale_base = ceil(slots * kv / 2)). One invocation encodes one group of `group` values like the CPU encoder (`encode_q8_cache_group`):
d = amax / 127, scale = f16(d), q = round-half-away-from-zero(x / d) clamped to +-127, all zero for d = 0. The division of GLSL is not correctly
rounded (2.5 ULP), so a value exactly between two steps may land on the other neighbour. Words shared with other rows or groups are merged with
atomicAnd/atomicOr; a word owned entirely by one invocation is stored directly.
```

| binding | name | access | type |
|---|---|---|---|
| 0 | `src` | readonly | `float src[];` |
| 1 | `slots` | readonly | `uint slots[];` |
| 2 | `pool` | read/write | `uint pool[];` |

| push offset | type | field |
|---|---|---|
| 0 | `uint` | `rows` |
| 4 | `uint` | `kv` |
| 8 | `uint` | `group` |
| 12 | `uint` | `base` |
| 16 | `uint` | `scale_base` |

Push block: 20 bytes. Grid: `[min(ceil(rows*(kv/group)/128), 65535), 1, 1]`.

### `attn_decode.comp`

Compiled as:

- `attn_decode_f16` (`-DKV_Q8=0`)
- `attn_decode_q8` (`-DKV_Q8=1`)

```text
Paged attention for decode and for short batches (Stream::attention_paged), split over the key range (split-KV / flash-decoding), one KV format per
variant (macro KV_Q8 = 0: f16 cache, 1: q8_0 cache). One workgroup of 64 lanes per (row, query head, split):
  * the lanes score 64 keys at a time (one key per lane: dot(q, key) * scale, ALiBi, soft cap, mask), fold the tile into an online softmax
    (running maximum m, denominator l) and accumulate the value rows: lane `l` owns the value dimensions l, l + 64, ... (dv <= 512);
  * nsplit == 1: the result is final (sink folded in) and written to `out`; nsplit > 1: the split writes m, l and the unnormalised accumulator to
    `part` and attn_merge combines the splits (the sink is added there).
Row r (global row index, row0 + z) attends to positions start .. lengths[r]-1 (start: see AttentionParams: chunk, window, or 0) through its block
table row; q is `rows x heads x dk`, out `rows x heads x dv` (row index = global row). `meta` = int32: lengths[rows_total] then the block tables
(rows_total x stride). `bias` = float: sinks[heads] then ALiBi slopes[heads] (always bound; flags bit 0 = sinks used, bit 1 = ALiBi used).
`part`: ml pairs (m, l) for every (row, head, split) record, then the accumulators (dv floats per record); record = (z * heads + head) * nsplit + split.
Needs dk, dv <= 512, nsplit <= 256. Empty splits (rows shorter than the split grid) write l = 0 and are skipped by the merge.
```

| binding | name | access | type |
|---|---|---|---|
| 0 | `q` | readonly | `float q[];` |
| 1 | `kpool` | readonly | `uint kpool[];` |
| 2 | `vpool` | readonly | `uint vpool[];` |
| 3 | `meta` | readonly | `int meta[];` |
| 4 | `bias` | readonly | `float bias_tab[];` |
| 5 | `out` | read/write | `float outv[];` |
| 6 | `part` | read/write | `float part[];` |

| push offset | type | field |
|---|---|---|
| 0 | `uint` | `heads` |
| 4 | `uint` | `kv_heads` |
| 8 | `uint` | `dk` |
| 12 | `uint` | `dv` |
| 16 | `uint` | `stride` |
| 20 | `uint` | `block_size` |
| 24 | `float` | `scale` |
| 28 | `float` | `softcap` |
| 32 | `uint` | `window` |
| 36 | `uint` | `chunk` |
| 40 | `uint` | `flags` |
| 44 | `uint` | `nsplit` |
| 48 | `uint` | `rows_total` |
| 52 | `uint` | `row0` |
| 56 | `uint` | `k_base` |
| 60 | `uint` | `v_base` |
| 64 | `uint` | `k_scale` |
| 68 | `uint` | `v_scale` |
| 72 | `uint` | `k_group` |
| 76 | `uint` | `v_group` |

Push block: 80 bytes. Grid: `[nsplit, heads, rows in this dispatch]`.

### `attn_merge.comp`

Compiled as `attn_merge`.

```text
Second stage of the split-KV attention (attn_decode with nsplit > 1): combines the partial results of the splits of one (row, head):
  M = max(m_s of the non-empty splits, sink), L = sum_s l_s exp(m_s - M) (+ exp(sink - M)), out[d] = sum_s exp(m_s - M) acc_s[d] / L.
Splits with l_s == 0 are empty and skipped. `bias` = sinks[heads] then ALiBi slopes[heads] (flags bit 0: sinks used). `part` as written by attn_decode
(record = (z * heads + head) * nsplit + split, z = workgroup y; `out` rows are global rows row0 + z). nsplit <= 256.
```

| binding | name | access | type |
|---|---|---|---|
| 0 | `bias` | readonly | `float bias_tab[];` |
| 1 | `out` | read/write | `float outv[];` |
| 2 | `part` | readonly | `float part[];` |

| push offset | type | field |
|---|---|---|
| 0 | `uint` | `heads` |
| 4 | `uint` | `dv` |
| 8 | `uint` | `nsplit` |
| 12 | `uint` | `flags` |
| 16 | `uint` | `row0` |

Push block: 20 bytes. Grid: `[heads, rows in this dispatch, 1]`.

### `attn_prefill.comp`

Compiled as:

- `attn_prefill_f16_d128` (`-DKV_Q8=0 -DDVMAX=128`)
- `attn_prefill_f16_d256` (`-DKV_Q8=0 -DDVMAX=256`)
- `attn_prefill_f16_d512` (`-DKV_Q8=0 -DDVMAX=512`)
- `attn_prefill_q8_d128` (`-DKV_Q8=1 -DDVMAX=128`)
- `attn_prefill_q8_d256` (`-DKV_Q8=1 -DDVMAX=256`)
- `attn_prefill_q8_d512` (`-DKV_Q8=1 -DDVMAX=512`)

```text
Paged attention for prompt processing (Stream::attention_paged with one row per token): a workgroup of 128 lanes handles one query head of a BLOCK of
16 consecutive rows and walks the keys in tiles of 32, so each key / value row is decoded once per 16 query rows instead of once per row.
Variants (macros): KV_Q8 = 0 / 1 (f16 / q8_0 cache), DVMAX = 128 / 256 / 512 (largest value width dv the accumulators hold; dk is free up to 512).
  Lane = (li, jl) with li = lane / 8 the query row inside the block and jl = lane % 8: scoring gives lane the keys jl, jl + 8, jl + 16, jl + 24 of the
  tile for its row (K and Q are staged in shared memory in chunks of 32 dimensions), the value accumulation gives it the dimensions jl + 8 e of every
  value chunk (V staged in shared memory in chunks of 32 dimensions). Online softmax per row; masks per row (chunk / window / causal length).
REQUIREMENT (the planner checks it): all rows of a block read the keys through ONE block table: the rows' tables agree on every block the row needs
as the table of the block's longest row. Blocks that do not satisfy it go to attn_decode. Row r attends to start(r) .. lengths[r]-1 like attn_decode.
Shared memory: 13.6 KiB. Output: out[(r * heads + head) * dv + d] for the global row r.
```

| binding | name | access | type |
|---|---|---|---|
| 0 | `q` | readonly | `float q[];` |
| 1 | `kpool` | readonly | `uint kpool[];` |
| 2 | `vpool` | readonly | `uint vpool[];` |
| 3 | `meta` | readonly | `int meta[];` |
| 4 | `bias` | readonly | `float bias_tab[];` |
| 5 | `out` | read/write | `float outv[];` |

| push offset | type | field |
|---|---|---|
| 0 | `uint` | `heads` |
| 4 | `uint` | `kv_heads` |
| 8 | `uint` | `dk` |
| 12 | `uint` | `dv` |
| 16 | `uint` | `stride` |
| 20 | `uint` | `block_size` |
| 24 | `float` | `scale` |
| 28 | `float` | `softcap` |
| 32 | `uint` | `window` |
| 36 | `uint` | `chunk` |
| 40 | `uint` | `flags` |
| 44 | `uint` | `rows_total` |
| 48 | `uint` | `row0` |
| 52 | `uint` | `rows_end` |
| 56 | `uint` | `k_base` |
| 60 | `uint` | `v_base` |
| 64 | `uint` | `k_scale` |
| 68 | `uint` | `v_scale` |
| 72 | `uint` | `k_group` |
| 76 | `uint` | `v_group` |

Push block: 80 bytes. Grid: `[ceil((rows_end - row0) / 16), heads, 1]`.

### `moe_topk.comp`

Compiled as `moe_topk`.

```text
Softmax routing of mixture-of-experts layers (Stream::moe_topk and Stream::moe_route): for every row of `logits` (rows x experts) the k best experts
(ties to the lower index) as ids (int32) with their softmax scores (over ALL experts) as weights, renormalised over the selected ones when
`normalize` is set (divided by max(sum, 1e-8) like the CPU `moe_route`) and then multiplied by `scale`. moe_route is normalize = 1, scale = 1.
ids[r * k + j] / weights[r * k + j] are in descending order of score. One workgroup per row; experts <= 1024, k <= 32 (the planner checks).
```

| binding | name | access | type |
|---|---|---|---|
| 0 | `logits` | readonly | `float logits[];` |
| 1 | `ids` | read/write | `int ids[];` |
| 2 | `weights` | read/write | `float weights[];` |

| push offset | type | field |
|---|---|---|
| 0 | `uint` | `rows` |
| 4 | `uint` | `experts` |
| 8 | `uint` | `k` |
| 12 | `uint` | `normalize_scores` |
| 16 | `float` | `scale` |

Push block: 20 bytes. Grid: `[rows split over x/y (x <= 65535), 1, 1]`.

### `moe_linear.comp`

Compiled as:

- `moe_linear_q4_0_arena` (`-DMOE_ARENA -DFMT_Q4_0`)
- `moe_linear_q4_1_arena` (`-DMOE_ARENA -DFMT_Q4_1`)
- `moe_linear_q5_0_arena` (`-DMOE_ARENA -DFMT_Q5_0`)
- `moe_linear_q5_1_arena` (`-DMOE_ARENA -DFMT_Q5_1`)
- `moe_linear_q8_0_arena` (`-DMOE_ARENA -DFMT_Q8_0`)
- `moe_linear_q2_k_arena` (`-DMOE_ARENA -DFMT_Q2_K`)
- `moe_linear_q3_k_arena` (`-DMOE_ARENA -DFMT_Q3_K`)
- `moe_linear_q4_k_arena` (`-DMOE_ARENA -DFMT_Q4_K`)
- `moe_linear_q5_k_arena` (`-DMOE_ARENA -DFMT_Q5_K`)
- `moe_linear_q6_k_arena` (`-DMOE_ARENA -DFMT_Q6_K`)
- `moe_linear_iq4_nl_arena` (`-DMOE_ARENA -DFMT_IQ4_NL`)
- `moe_linear_mxfp4_arena` (`-DMOE_ARENA -DFMT_MXFP4`)
- `moe_linear_q4_0_bda` (`-DMOE_BDA -DFMT_Q4_0`)
- `moe_linear_q4_1_bda` (`-DMOE_BDA -DFMT_Q4_1`)
- `moe_linear_q5_0_bda` (`-DMOE_BDA -DFMT_Q5_0`)
- `moe_linear_q5_1_bda` (`-DMOE_BDA -DFMT_Q5_1`)
- `moe_linear_q8_0_bda` (`-DMOE_BDA -DFMT_Q8_0`)
- `moe_linear_q2_k_bda` (`-DMOE_BDA -DFMT_Q2_K`)
- `moe_linear_q3_k_bda` (`-DMOE_BDA -DFMT_Q3_K`)
- `moe_linear_q4_k_bda` (`-DMOE_BDA -DFMT_Q4_K`)
- `moe_linear_q5_k_bda` (`-DMOE_BDA -DFMT_Q5_K`)
- `moe_linear_q6_k_bda` (`-DMOE_BDA -DFMT_Q6_K`)
- `moe_linear_iq4_nl_bda` (`-DMOE_BDA -DFMT_IQ4_NL`)
- `moe_linear_mxfp4_bda` (`-DMOE_BDA -DFMT_MXFP4`)

```text
Expert-indexed matrix-vector product (Stream::moe_linear):  out[slot][r] = dot(W[ids[slot]][r], x[slot / x_div]) (+ bias[ids[slot]][r]),
slot < slots, r < rows, W[e] a rows x cols weight in the planar device layout of its format (see _ops_wq.glsl; one macro FMT_* per
variants_ops.list entry). cols is a multiple of QK. An expert id outside 0..n_experts gives zeros (the CPU skips such experts).
One workgroup per (row, slot); the lanes stride over the 8-weight chunks of the row and reduce in shared memory.
Two ways to find the weights of an expert (macro, one per variant):
  MOE_ARENA  all experts of the projection live in ONE storage buffer `weights`; table[e] = word offset of expert e in it (uint32); the
             biases (if any) live in ONE buffer `biases`, bias_table[e] = float offset of expert e's bias vector in it.
  MOE_BDA    needs bufferDeviceAddress (GL_EXT_buffer_reference); table[e] / bias_table[e] = the 64-bit device address of the expert's weight /
             bias buffer as a uvec2 (low word first), like `upload_pointer_table` of the CUDA backend. `weights` and `biases` are unused.
Bindings that a variant does not use stay declared so that all variants of the operation share one descriptor layout; bind any small buffer to them.
flags bit 0: bias_table present. nb = cols / QK (blocks per row), off1..off3 = word offsets of the planes 1..3 inside one expert (identical for all
experts, as they have the same shape and format). x is read as vec4 (its binding offset must be a multiple of 16 bytes; cols is a multiple of 8).
```

| binding | name | access | type |
|---|---|---|---|
| 0 | `table` | readonly | `uvec2 table[];` |
| 1 | `bias_table` | readonly | `uvec2 bias_table[];` |
| 0 | `table` | readonly | `uint table[];` |
| 1 | `bias_table` | readonly | `uint bias_table[];` |
| 2 | `ids` | readonly | `int ids[];` |
| 3 | `x` | readonly | `vec4 x4[];` |
| 4 | `out` | read/write | `float o[];` |
| 5 | `weights` | readonly | `uint arena[];` |
| 6 | `biases` | readonly | `float biases[];` |

| push offset | type | field |
|---|---|---|
| 0 | `uint` | `rows` |
| 4 | `uint` | `cols` |
| 8 | `uint` | `slots` |
| 12 | `uint` | `x_div` |
| 16 | `uint` | `n_experts` |
| 20 | `uint` | `flags` |
| 24 | `uint` | `nb` |
| 28 | `uint` | `off1` |
| 32 | `uint` | `off2` |
| 36 | `uint` | `off3` |

Push block: 40 bytes. Grid: `[rows, min(slots, 65535), ceil(slots / 65535)]`.

### `moe_combine.comp`

Compiled as `moe_combine`.

```text
Stream::moe_combine: out[t][h] = sum_j weights[t * k + j] * y[(t * k + j) * hidden + h] for t < tokens, h < hidden (grid-stride loop).
```

| binding | name | access | type |
|---|---|---|---|
| 0 | `out` | read/write | `float o[];` |
| 1 | `y` | readonly | `float y[];` |
| 2 | `weights` | readonly | `float weights[];` |

| push offset | type | field |
|---|---|---|
| 0 | `uint` | `tokens` |
| 4 | `uint` | `k` |
| 8 | `uint` | `hidden` |

Push block: 12 bytes. Grid: `[min(ceil(tokens*hidden/256), 65535), 1, 1]`.

### `moe_gather_rows.comp`

Compiled as `moe_gather_rows`.

```text
Stream::gather_rows: dst[i] = src[idx[i]] * weights[i] (weights optional: has_weights = 0 means 1) for i < n rows of row_len floats.
idx (int32, checked >= 0 on the host) and weights are small host arrays uploaded with the dispatch.
```

| binding | name | access | type |
|---|---|---|---|
| 0 | `dst` | read/write | `float dst[];` |
| 1 | `src` | readonly | `float src[];` |
| 2 | `idx` | readonly | `int idx[];` |
| 3 | `weights` | readonly | `float weights[];` |

| push offset | type | field |
|---|---|---|
| 0 | `uint` | `n` |
| 4 | `uint` | `row_len` |
| 8 | `uint` | `has_weights` |

Push block: 12 bytes. Grid: `[min(ceil(n*row_len/256), 65535), 1, 1]`.

### `moe_scatter_add_rows.comp`

Compiled as `moe_scatter_add_rows`.

```text
Stream::scatter_add_rows: dst[idx[i]] += src[i] * weights[i] (weights optional: has_weights = 0 means 1) for i < n rows of row_len floats. The rows of
one call have distinct destinations (Stream contract), so a plain read-modify-write is enough.
```

| binding | name | access | type |
|---|---|---|---|
| 0 | `dst` | read/write | `float dst[];` |
| 1 | `src` | readonly | `float src[];` |
| 2 | `idx` | readonly | `int idx[];` |
| 3 | `weights` | readonly | `float weights[];` |

| push offset | type | field |
|---|---|---|
| 0 | `uint` | `n` |
| 4 | `uint` | `row_len` |
| 8 | `uint` | `has_weights` |

Push block: 12 bytes. Grid: `[min(ceil(n*row_len/256), 65535), 1, 1]`.

### `delta_conv.comp`

Compiled as `delta_conv`.

```text
Gated DeltaNet, part 1 (port of delta_conv_kernel in cuda/deltanet.cu, reference cpu_delta.rs): causal convolution over the channels [q | k | v] of one run
(n consecutive tokens of one sequence, rows row0 .. row0+n-1 of qkv), SiLU, and the L2 normalisation of every query / key head:
  mixed[t][c] = silu( sum_{k < ksz-1} w[c][k] * x[t - (ksz-1) + k][c] + w[c][ksz-1] * x[t][c] )     (x before the run = the stored history, or 0 when reset)
  query heads: * 1/sqrt(sum_head(mixed^2) + 1e-6) * qscale (= hk^-0.5), key heads: * 1/sqrt(sum_head(mixed^2) + 1e-6), value channels unchanged.
Workgroup x: b < kh a query head, b < 2 kh a key head (channels b * hk .., hk lanes, the L2 norm is the workgroup reduction), otherwise a block of 128
value channels. Workgroup y: a chunk of 16 tokens (>= ksz - 1 so that only chunk 0 ever reads the history); chunk 0 rewrites the history at the end.
The state buffer `state` (float view of the sequence's state buffer): the history of channel c is state[hist_off + c * (ksz-1) + k] (the k-th last input,
oldest first), hist_off = (state_offset + rec_bytes) / 4 of the layer. hk <= 128, ksz in 2..8. One dispatch per run (every run has its own state buffer).
```

| binding | name | access | type |
|---|---|---|---|
| 0 | `qkv` | readonly | `float qkv[];` |
| 1 | `conv` | readonly | `float convw[];` |
| 2 | `state` | read/write | `float state[];` |
| 3 | `mixed` | read/write | `float mixed[];` |

| push offset | type | field |
|---|---|---|
| 0 | `uint` | `conv_dim` |
| 4 | `uint` | `kh` |
| 8 | `uint` | `hk` |
| 12 | `uint` | `ksz` |
| 16 | `uint` | `row0` |
| 20 | `uint` | `n` |
| 24 | `uint` | `reset` |
| 28 | `uint` | `hist_off` |
| 32 | `float` | `qscale` |

Push block: 36 bytes. Grid: `[2*kh + ceil((conv_dim - 2*kh*hk)/128), ceil(n/16), 1] per run`.

### `delta_rec.comp`

Compiled as:

- `delta_rec_hk32` (`-DHK=32`)
- `delta_rec_hk64` (`-DHK=64`)
- `delta_rec_hk96` (`-DHK=96`)
- `delta_rec_hk128` (`-DHK=128`)

```text
Gated DeltaNet, part 2 (port of delta_rec_kernel in cuda/deltanet.cu, reference cpu_delta.rs): the delta-rule recurrence of one run (n consecutive tokens of
one sequence) for one value head per workgroup, per token t:
  forget = exp(decay[j] * softplus(alpha[t][j] + dt_bias[j])),  sigma = beta_scale * sigmoid(beta[t][j])
  S = S * forget;  S += k (x) ((v - S^T k) * sigma);  o = S^T q            (q, k: the normalised query / key head of mixed, v: its value channels)
  out[t][j][c] = o[c] * (1 / sqrt(mean_c(o^2) + eps)) * norm[c] * silu(z[t][j][c])        (gated RMS norm of the head output)
The recurrent state S[j][i][c] (value head j, key row i < HK, value column c < hv) is FP32 at state[state_off + (j * HK + i) * hv + c] of the sequence's
state buffer (float view; state_off = state_offset / 4 of the layer). Value column c is handled by two neighbouring lanes (lane = 2 c + half): half 0
holds the first HK/2 rows of the column in registers, half 1 the others; the dot products over the key dimension meet in shared memory. Lanes with
c >= hv idle but join every barrier. HK is a compile-time macro (64 / 128 ...), hv <= 128 at run time. `mixed` is the output of delta_conv.
Key head of value head j: tiled != 0 ? j % kh : j / (vh / kh). reset != 0: the old state is ignored.
```

| binding | name | access | type |
|---|---|---|---|
| 0 | `mixed` | readonly | `float mixed[];` |
| 1 | `z` | readonly | `float z[];` |
| 2 | `beta` | readonly | `float beta[];` |
| 3 | `alpha` | readonly | `float alpha[];` |
| 4 | `dt_bias` | readonly | `float dt_bias[];` |
| 5 | `decay` | readonly | `float decay[];` |
| 6 | `norm` | readonly | `float norm_w[];` |
| 7 | `state` | read/write | `float state[];` |
| 8 | `out` | read/write | `float outv[];` |

| push offset | type | field |
|---|---|---|
| 0 | `uint` | `kh` |
| 4 | `uint` | `vh` |
| 8 | `uint` | `hv` |
| 12 | `uint` | `conv_dim` |
| 16 | `uint` | `tiled` |
| 20 | `uint` | `row0` |
| 24 | `uint` | `n` |
| 28 | `uint` | `reset` |
| 32 | `uint` | `state_off` |
| 36 | `float` | `beta_scale` |
| 40 | `float` | `eps` |

Push block: 44 bytes. Grid: `[vh, 1, 1] per run`.

### `mamba_conv.comp`

Compiled as `mamba_conv`.

```text
Mamba (1 and 2) causal depthwise convolution with SiLU over one run (n consecutive tokens of one sequence), reference: the loop over `mixed` in
`mamba_forward` / `mamba1_forward` of rust_decoder.rs:
  mixed[t][c] = silu( bias[c] + sum_{k < ksz-1} w[c][k] * x[t - (ksz-1) + k][c] + w[c][ksz-1] * x[t][c] )
with x before the run taken from the stored history (oldest first), or 0 when reset != 0. x[t][c] = x[(row0 + t) * in_stride + in_off + c] (a slice of
the projected row: Mamba-2 xBC = in_off inner, Mamba-1 xs = in_off 0); mixed is dense, mixed[(row0 + t) * channels + c]. The history of channel c is
state[hist_off + c * (ksz-1) + k]; the first chunk of 16 tokens (>= ksz - 1) rewrites it at the end. ksz in 2..8, has_bias != 0: convb[c] is added.
```

| binding | name | access | type |
|---|---|---|---|
| 0 | `x` | readonly | `float x[];` |
| 1 | `convw` | readonly | `float convw[];` |
| 2 | `convb` | readonly | `float convb[];` |
| 3 | `state` | read/write | `float state[];` |
| 4 | `mixed` | read/write | `float mixed[];` |

| push offset | type | field |
|---|---|---|
| 0 | `uint` | `channels` |
| 4 | `uint` | `ksz` |
| 8 | `uint` | `in_stride` |
| 12 | `uint` | `in_off` |
| 16 | `uint` | `row0` |
| 20 | `uint` | `n` |
| 24 | `uint` | `reset` |
| 28 | `uint` | `hist_off` |
| 32 | `uint` | `has_bias` |

Push block: 36 bytes. Grid: `[ceil(channels/64), ceil(n/16), 1] per run`.

### `mamba_scan2.comp`

Compiled as:

- `mamba_scan2_ns16` (`-DNS=16`)
- `mamba_scan2_ns32` (`-DNS=32`)
- `mamba_scan2_ns64` (`-DNS=64`)
- `mamba_scan2_ns128` (`-DNS=128`)
- `mamba_scan2_gen` (`-DNS=0`)

```text
Mamba-2 selective state update (reference: the scan loop of `mamba_forward` in rust_decoder.rs) of one run (n consecutive tokens of one sequence). Per
token, head h, channel p of the head (xp = mixed[t][h * head_dim + p]) and state index k < ns:
  dt = max(softplus(dt_raw[t][h] + dt_bias[h]), step_min);  da = exp(dt * decay[h])     (decay = A < 0; dt_raw = projected[t * proj_stride + proj_dt_off + h])
  cell[h][p][k] = cell[h][p][k] * da + dt * xp * B[g][k];   out[t][h * head_dim + p] = sum_k cell[h][p][k] * C[g][k] + skip[h] * xp
with g = h / (heads / groups); B[g] = mixed[t][inner + g * ns ..], C[g] = mixed[t][inner + groups * ns + g * ns ..] (mixed = output of mamba_conv, conv_dim
channels per row: [x (inner) | B | C]). The SSM state is state[state_off + (h * head_dim + p) * ns + k] (float view of the sequence's state buffer; reset != 0
ignores the old state). One lane per channel p, the tokens in order. Macro NS = state size: NS > 0 keeps the cell row in registers for the whole run;
NS = 0 (generic) reads and writes the state in memory for every token. ns <= 256 (shared staging of B and C).
```

| binding | name | access | type |
|---|---|---|---|
| 0 | `mixed` | readonly | `float mixed[];` |
| 1 | `projected` | readonly | `float projected[];` |
| 2 | `dt_bias` | readonly | `float dt_bias[];` |
| 3 | `decay` | readonly | `float decay[];` |
| 4 | `skip` | readonly | `float skip_d[];` |
| 5 | `state` | read/write | `float state[];` |
| 6 | `out` | read/write | `float outv[];` |

| push offset | type | field |
|---|---|---|
| 0 | `uint` | `inner` |
| 4 | `uint` | `ns` |
| 8 | `uint` | `heads` |
| 12 | `uint` | `groups` |
| 16 | `uint` | `head_dim` |
| 20 | `uint` | `conv_dim` |
| 24 | `uint` | `proj_stride` |
| 28 | `uint` | `proj_dt_off` |
| 32 | `uint` | `row0` |
| 36 | `uint` | `n` |
| 40 | `uint` | `reset` |
| 44 | `uint` | `state_off` |
| 48 | `float` | `step_min` |

Push block: 52 bytes. Grid: `[heads, ceil(head_dim/64), 1] per run`.

### `mamba_scan1.comp`

Compiled as:

- `mamba_scan1_ns8` (`-DNS=8`)
- `mamba_scan1_ns16` (`-DNS=16`)
- `mamba_scan1_ns32` (`-DNS=32`)
- `mamba_scan1_gen` (`-DNS=0`)

```text
Mamba-1 selective scan (reference: `mamba1_forward` in rust_decoder.rs) of one run (n consecutive tokens of one sequence), per channel ch < inner:
  step = softplus(dt[t][ch]);  s[k] = s[k] * exp(step * decay[ch * ns + k]) + step * B[k] * xc;  y = (sum_k s[k] * C[k] + skip[ch] * xc) * silu(gate[ch])
with xc = mixed[t][ch] (output of mamba_conv, `inner` channels), dt = the output of dt_proj (`inner` per token), B = split[t][rank + k], C = split[t][rank + ns + k]
(split = output of x_proj, split_stride = rank + 2 ns floats per token, B and C normalised beforehand when the model asks for it) and
gate[ch] = projected[t * proj_stride + proj_gate_off + ch]. The state s of channel ch is state[state_off + ch * ns + k]. NS > 0: the state row stays in
registers for the whole run; NS = 0 (generic): read and written in memory per token. ns <= 256.
```

| binding | name | access | type |
|---|---|---|---|
| 0 | `mixed` | readonly | `float mixed[];` |
| 1 | `dt` | readonly | `float dt[];` |
| 2 | `split` | readonly | `float split_bc[];` |
| 3 | `projected` | readonly | `float projected[];` |
| 4 | `decay` | readonly | `float decay[];` |
| 5 | `skip` | readonly | `float skip_d[];` |
| 6 | `state` | read/write | `float state[];` |
| 7 | `out` | read/write | `float outv[];` |

| push offset | type | field |
|---|---|---|
| 0 | `uint` | `inner` |
| 4 | `uint` | `ns` |
| 8 | `uint` | `rank` |
| 12 | `uint` | `split_stride` |
| 16 | `uint` | `proj_stride` |
| 20 | `uint` | `proj_gate_off` |
| 24 | `uint` | `row0` |
| 28 | `uint` | `n` |
| 32 | `uint` | `reset` |
| 36 | `uint` | `state_off` |

Push block: 40 bytes. Grid: `[ceil(inner/64), 1, 1] per run`.

### `mamba_gate_norm.comp`

Compiled as `mamba_gate_norm`.

```text
Mamba-2 gated group RMS norm (end of `mamba_forward` in rust_decoder.rs), in place on y (rows x inner): v = y * silu(gate), then for every group of
size = inner / norm_groups values  y = v * (1 / sqrt(mean(v^2) + eps)) * norm[g * size + i]. gate = projected[t * proj_stride + proj_gate_off + i].
One workgroup per (group, row); for the whole batch at once (no recurrent state involved).
```

| binding | name | access | type |
|---|---|---|---|
| 0 | `y` | read/write | `float y[];` |
| 1 | `projected` | readonly | `float projected[];` |
| 2 | `norm` | readonly | `float norm_w[];` |

| push offset | type | field |
|---|---|---|
| 0 | `uint` | `inner` |
| 4 | `uint` | `norm_groups` |
| 8 | `uint` | `rows` |
| 12 | `uint` | `proj_stride` |
| 16 | `uint` | `proj_gate_off` |
| 20 | `float` | `eps` |

Push block: 24 bytes. Grid: `[norm_groups, rows, 1]`.


### `misc_dsa_keys.comp`, `misc_dsa_scores.comp`, `misc_dsa_topk.comp`, `attn_sparse.comp`

Lightning indexer and sparse attention of DeepSeek-V3.2 / GLM-5 (`Stream::dsa_select`, `Stream::dsa_attention`; plans in `ops/dsa.rs`). The
bindings and push constants are listed in the `// FLLM-SHADER:` header of each file.

* `misc_dsa_keys`: writes the indexer keys of the new tokens (`rows x dim` f32) into the per-slot key store (`slots` host array).
* `misc_dsa_scores`: one lane per position, `sum_h relu(dot(q_h, key) * scale) * w_h * head_scale`; `meta` holds `lengths[rows]` then the block tables.
  Rows are processed in chunks of at most 4M (row, position) pairs per dispatch (`row0`, `score_stride` select the chunk).
* `misc_dsa_topk`: one workgroup of 256 per row, radix select of the k-th largest score (order-preserving keys of the floats, 4 x 8 bits), then
  the positions are collected in ascending order with workgroup prefix sums; equal scores keep the lower position, unused entries are -1.
  A row of at most `topk` tokens lists all positions without looking at scores.
* `attn_sparse`: one workgroup of 128 per (row, head); the listed positions are scored 128 at a time into an online softmax; f16 cache only, key
  width up to 512, value width up to 512.

### `misc_hc_mix.comp`, `misc_hc_combine.comp`, `misc_rotate_heads.comp`

Hyper-connections (`Stream::hc_mix`, `Stream::hc_combine`; plans in `ops/hc.rs`) and the tail rotation of the Hy4 indexer heads (`Stream::rotate_heads`).

* `misc_hc_mix`: one workgroup per row. Thread 0 turns the linear output `mixes` of the RMS-normalised streams into the weights of the block (`mode` 0
  independent, 1 Sinkhorn with `iters` normalisation steps, 2 head, 3 mean) and writes `post` (`rows x hc`) and, for mode 1, `comb`
  (`rows x hc x hc`); then all lanes write `collapsed[r] = sum_n pre[n] streams[r][n]`. At most 8 streams.
* `misc_hc_combine`: in place on the streams, one invocation per (row, d) reading all `hc` old values before it writes: `streams[k] = post[k] block + sum_j comb[j][k] old[j]`
  (`has_comb = 0`: `+ old[k]`).
* `misc_rotate_heads`: `out[g * dim + (d + shift) % dim] = x[g * dim + d]`.

### `act_glu_ext.comp`, `misc_dsa_pscores.comp`, `misc_dsa_expand.comp`

* `act_glu_ext`: clamped gated activations (`Stream::glu_clamped`), kind 0 clamped SwiGLU (`p0` = limit), kind 1 SiTU (`p0` = beta, `p1` = linear beta); dense gate, up and out buffers.
* `misc_dsa_pscores`: scores of the complete pools of the k-pool indexer (GLM-5-Next): per channel the softmax over the pool's tokens of gate score plus position bias weights the keys, then the usual head sum. At most 128 channels and 16 tokens per pool.
* `misc_dsa_expand`: the first `kp` entries of a row of `sel` (written by `misc_dsa_topk` with `len_div = kpool` and `out_stride = sel_width`) are the chosen pools; the row is rewritten in place as their tokens plus the tokens of the incomplete last pool.
* `dsa_select` and `dsa_attention` return their plans in groups; the stream runs each group and submits it on its own (about 2e9 multiply-adds per scoring group, 512k position triples per attention group).

### `misc_block_scores.comp`, `attn_block_sparse.comp` (MiniMax-M3 block-sparse attention; `Stream::block_select`, `Stream::block_attention`; plans in `ops/block.rs`)

* `misc_block_scores`: one workgroup of 128 per (row, indexer head, block): the maximum over the keys of the block of `dot(q, key)` (plain dot product, no scale, no ReLU) from
  the per-slot f32 key store, and `3e38` (a finite stand-in for +infinity) for the newest `local` blocks of the row. Grid `[blocks of the longest row, heads, rows of the group]`; `meta` holds
  `lengths[rows_total]`, then the block tables. Rows are cut into groups of about 2e9 multiply-adds, each its own submission (`ops::dsa::GROUP_MACS`).
* The selection is `misc_dsa_topk` over rows * heads rows of block scores (`len_div = 1`, `out_stride = topk`, row lengths = blocks per row in its own `meta`): ascending block numbers, `-1` after the last, equal scores keep the lower block, a row with at most `topk` blocks lists all of them.
* `attn_block_sparse`: like `attn_sparse`, but `sel` is `rows x groups x topk` block numbers (blocks of `block` keys): head `h` reads the list of indexer head `h / (heads / groups)`, and keys at or beyond `lengths[r]` (the unfinished last block) are skipped. One dispatch holds at most 512k (row, head, key) triples.
* The indexer keys are written by `misc_dsa_keys` (width = dim, no gate).

### `conv_resid.comp`, `attn_rel.comp` (Inkling; `Stream::conv_resid`, `Stream::rel_attention`; plans in `ops/inkling.rs`)

* `conv_resid`: `out = x + depthwise causal convolution of x` for one run (like `shortconv` without the gating and the bias): one invocation per channel walks the tokens, the history of the last `taps - 1` inputs per channel is `state[state_off + ch * (taps - 1) + j]` (oldest first), `reset` ignores the old history, taps 2..8.
* `attn_rel`: attention without rotary embeddings, one workgroup of 128 per (row, head), tiles of 128 keys with an online softmax: `score(j) = dot(q * tau, key_j) * scale + tau * sum_c rel[c] * profile[c][distance]` (the bias exists for `distance < extent`), `tau = 1 + log_alpha * ln(max(1, (position + 1) / log_floor))` when `log_floor > 0`; the visible keys are the last `window` ones (0: all). `rel` holds `d_rel <= 128` values per row and head, `profile` is `d_rel x extent`; key and value width `hd <= 512`; at most 512k (row, head, visible key) triples per dispatch.
