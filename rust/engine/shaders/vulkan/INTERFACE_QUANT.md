# Quantized linear shaders — interface contract

Verification level: **L0** (every shader compiles with `glslc --target-env=vulkan1.2 -O` and passes `spirv-val`; the layouts are checked by CPU unit tests
against `compute::dequantize`; the shader text was additionally executed as C++ on the CPU against an f64 reference, see `emu/README.md`; no shader has ever run on a GPU).

`Stream::quantized_linear`: `out[b][r] = scale * dot(x[b], w[r]) + bias[r]` for `b < batch`, `r < rows`.
`rows/cols/scale/bias` come from `DeviceWeight`. The Rust side is `src/backend/vulkan/weights.rs` (`pack_weight`, `plan_quantized_linear`).

## Files

| file | purpose |
|---|---|
| `_q_ext.glsl`, `_q_common.glsl`, `_q_formats.glsl`, `_q_fmt_<fmt>.glsl` | includes: extensions, push constants + helpers, chunk decoders per format |
| `mv_scalar.comp` / `mv_idot.comp` | matrix-vector for batch 1..8: f32 arithmetic / int8 activations + dot product |
| `mm_scalar.comp` / `mm_idot.comp` / `mm_coopmat.comp` | matrix-matrix for batch > 8: f32 / int8 dot / cooperative matrices |
| `quant_act_q8.comp` (+ `PERM`) | prepass: f32 activations -> int8 blocks of 32 with float scales (for the `_i` shaders) |
| `quant_permute.comp` | prepass: gathers the columns for act-order GPTQ (`Q4ZP*`) with the f32 shaders |
| `variants_quant.list` | the 83 compiled shaders: `mv_<fmt>_s` / `_i`, `mm_<fmt>_s` / `_i` / `_c`, `quant_act_q8`, `quant_act_q8_perm`, `quant_permute` |
| `emu/` | CPU-only checks: compile + validate all variants, run the decoders and the whole shaders as C++ against the dequantized reference (`emu/README.md`) |

`<fmt>` = `q4_0 q4_1 q5_0 q5_1 q8_0 q2_k q3_k q4_k q5_k q6_k iq4_nl mxfp4 nvfp4 q4z32 q4z64 q4z128` (`Q4ZP32/64/128` use the `q4z*` shaders and a prepass).
Not covered here: F32/F16/BF16 weights, the other IQ formats, FP8M. There is no cooperative-matrix matrix-vector shader on purpose: with batch <= 8 a 16-wide
matrix tile would waste at least half of the math and the decode is memory bound; `mm_*_c` serves batch >= 9.

## Variants and device requirements

| suffix | arithmetic | needs |
|---|---|---|
| `_s` | f32 activations, f32 FMA, shared-memory reduction | nothing beyond Vulkan 1.0 core (`uint`/`vec4`/`float` storage buffers only; no 8/16-bit storage, no float16, no subgroup ops) |
| `_i` | activations rounded to int8 (blocks of 32), integer dot products | prepass `quant_act_q8*`. If the compiler knows `GL_EXT_integer_dot_product` (glslang >= 15.2 / shaderc 2025+) the module uses `OpSDot` (**SPIR-V capabilities `DotProduct`, `DotProductInput4x8BitPacked`, extension `SPV_KHR_integer_dot_product`**, device feature `shaderIntegerDotProduct` must be enabled); older compilers emit an exact emulation with plain integer ops that runs everywhere. The planner selects `_i` only if `caps.int_dot` is set. |
| `_c` | f16 tiles, `coopMatMulAdd` 16x16x16 (f16 x f16 + f32 -> f32) | **capabilities `CooperativeMatrixKHR`, `Float16`, `GroupNonUniform`, `VulkanMemoryModel`**, extension `SPV_KHR_cooperative_matrix`; device features `cooperativeMatrix` (shape 16x16x16 f16/f16/f32/f32, subgroup scope), `shaderFloat16`, `vulkanMemoryModel`, subgroup operations in compute; shared memory >= 14336 bytes. **Workgroup size (spec constant 0) must be 4 x the subgroup size** and the pipeline must really run with that subgroup size (`VK_EXT_subgroup_size_control` / fixed-size device). |

Precision: `_s` meets the FP32 worst-case bound of `selftest-gpu`'s exact pass (`FLLM_VULKAN_EXACT=1` / `PlanOptions::exact` makes the planner use only `_s`). `_i` adds the rounding of
the activations to 8 bit (relative error ~ 0.4 % of the largest value of each 32-block), `_c` rounds weights and activations to f16; both belong to the "fast" tolerance.

Build: all 83 variants are compiled by `build.rs` when feature `vulkan` is on (about 20 s with glslc). The `_c` lines end in `#optional`; see the notes in `build.rs`.

## Bindings (descriptor set 0, all `std430` storage buffers)

Main shaders `mv_*`, `mm_*`:

| binding | name | type | contents |
|---|---|---|---|
| 0 | weights | `uint[]` readonly | `PackedWeight::bytes` (device layout below) |
| 1 | x | `_s`, `_c`: `vec4[]`; `_i`: `uint[]` | f32 activations `batch x cols` (offset multiple of 16 bytes; `cols` multiple of 8) / prepass output |
| 2 | out | `float[]` writeonly | `out[token * out_stride + row_base + r]` |
| 3 | bias | `float[]` readonly | `bias[row_base + r]`; always bind a buffer (any small one when `flags & 1 == 0`) |

Prepass `quant_act_q8`: 0 x (`float[]`), 1 xq (`uint[]` out), 2 perm (`uint[]`, only `quant_act_q8_perm`). Prepass `quant_permute`: 0 x, 1 y (`float[]` out), 2 perm.
`perm` is `PackedWeight::perm` (u32 per column, uploaded as `DeviceWeight::perm`): the shaders see `x'[c] = x[perm[c]]`.

Quantized activations (`xq`): `batch * cols / 4` words of int8 (row `b` starts at word `b * cols / 4`, byte `i % 4` of word `i / 4` is value `i`), then (word offset `xq_d = batch * cols / 4`)
`batch * cols / 32` float scales (`d = amax / 127`, bit pattern in a word). Size = `(xq_d + batch * cols / 32) * 4` bytes (`PrepassPlan::output_bytes`).
Several weights reading the same activations (q, k, v; gate, up) can share one prepass result.

## Push constants (`QPush`, 48 bytes, little endian words)

`rows, cols, batch, flags, scale (f32), nb, off1, off2, off3, out_stride, row_base, xq_d`

- `rows`: weight rows of this dispatch; `cols`: K (multiple of the block size); `batch`: activation rows.
- `flags` bit 0: add bias. `scale`: `DeviceWeight::scale` (NVFP4 per-tensor factor; 1.0 otherwise).
- `nb = cols / QK`; `off1..off3`: word offsets of planes 1..3 (`DeviceLayout::plane_word_offset`).
- `out_stride`: floats between output rows of two tokens (normally the full number of weight rows); `row_base`: first output feature of this dispatch. Weights with more rows than fit
  `maxStorageBufferRange` are uploaded as slices of `rows_per_slice(..)` rows (`pack_weight_slices`) and dispatched once per slice with their `row_base` and the full `out_stride`.
- `xq_d`: see above (only the `_i` shaders read it).

## Specialization constants and grids

| shader | spec constants | grid (`vkCmdDispatch`) |
|---|---|---|
| `mv_*` | 0 = workgroup size (power of two, 64/128), 1 = `NROWS` weight rows per workgroup (2), 2 = `NCOLS` = batch (1..8) | `ceil(rows / NROWS)` workgroups, split into `x <= 65535`, `y = ceil(n / 65535)` (group = `x + y * numGroupsX`) |
| `mm_*_s`, `mm_*_i` | none (local size 256) | `ceil(rows / 64)`, `ceil(batch / 64)`, 1 |
| `mmq_*_i` (`mm_q.comp`, binding 1 is read as `uvec4[]`, so the prepass buffer is a multiple of 16 bytes) | 0 = workgroup size = BM * BN / 16, 1 = BM (32/64/128), 2 = BN (64/128) | `ceil(rows / BM)`, `ceil(batch / BN)`, 1; planned instead of `mm_*_i` from batch 32 up |
| `mm_*_c` | 0 = workgroup size = 4 x subgroup size | same |
| `quant_act_q8*` | none (local size 128) | `ceil(batch * cols / 32 / 128)`, split like `mv_*` |
| `quant_permute` | none (local size 256) | `ceil(batch * cols / 256)`, split like `mv_*` |

Shared memory: `mv_*` uses `NROWS * NCOLS * WG * 4` bytes, `mm_*_s` 8 KiB, `mm_*_i` ~6.4 KiB, `mm_*_c` 14 KiB.
All of this (and the choice of the variant) is computed by `plan_quantized_linear`; the pipeline cache key is `(shader_name, spec_constants)`.

## Chunk model

All formats are decoded in *chunks* of 8 consecutive weights of a row: `weight[i] = ws * q[i] + wm` with `q[i]` a small signed integer (packed as int8 lanes in two `uint`s) and two floats.
`wm` is non-zero only for `Q4_1, Q5_1, Q2_K, Q4_K, Q5_K` (`HAS_WM`). The scalar shaders unpack the lanes to float, the `_i` shaders feed them to `dotPacked4x8` directly
(`acc += da * (ws * dot(q, a) + wm * sum(a))`); the tile loaders of `mm_*_s` / `mm_*_c` write `ws * q + wm`.
`g` = block index `row * nb + block`, `cin` = chunk in the block.

## Device layout of the weights (planes)

The buffer is a concatenation of planes, each starting at a word offset that is a multiple of 4 words (16 bytes); plane 0 starts at 0, planes 1..3 at `off1..off3`.
`P{i}` below is per block (g = block index), sizes in bytes per block. "f16" = IEEE binary16; word = 32-bit little endian.

| format (QK) | P0 | P1 | P2 | P3 |
|---|---|---|---|---|
| Q4_0 (32) | 16: qs as in the file | 2: f16 d | | |
| Q4_1 (32) | 16: qs | 4: word `d \| m << 16` | | |
| Q5_0 (32) | 16: qs | 4: qh word | 2: f16 d | |
| Q5_1 (32) | 16: qs | 4: qh word | 4: word `d \| m << 16` | |
| Q8_0 (32) | 32: int8 qs | 2: f16 d | | |
| Q2_K (256) | 64: qs | 16: scales (low nibble scale, high nibble min) | 4: word `d \| dmin << 16` | |
| Q3_K (256) | 64: qs | 32: hmask | 16: int8 scales (`scale - 32`, decoded) | 4: f32 d |
| Q4_K (256) | 128: qs | 16: eight u8 6-bit scales, eight u8 6-bit mins (decoded) | 4: word `d \| dmin << 16` | |
| Q5_K (256) | 128: qs | 32: qh | 16: scales/mins as Q4_K | 4: word `d \| dmin << 16` |
| Q6_K (256) | 128: ql | 64: qh | 16: int8 scales | 4: f32 d |
| IQ4_NL (32) | 16: qs | 2: f16 d | | |
| MXFP4 (32) | 16: qs | 1: E8M0 scale byte (4 blocks per word) | | |
| NVFP4 (64) | 32: qs | 4: the four UE4M3 scales (one word) | | |
| Q4Z32/64/128 (32/64/128) | `group/2`: codes, repacked per chunk | 4: word `f16 scale \| zero << 16` | | |

Notes: nibble formats keep the file's order (value `j` = low nibble of byte `j`, `j + 16` = high nibble) except Q4Z, which is repacked per chunk of 8 values into one word
(byte `k`: value `k` in the low and value `k + 4` in the high nibble) because the file stores consecutive values in the two nibbles of one byte. `Q4ZP*` blocks are the same as `Q4Z*`;
the permutation table that follows the blocks in the file is split off into `PackedWeight::perm`. Weight values: see the comment at the top of each `_q_fmt_*.glsl`.
Memory use: the file size, plus the widened K-quant scales (Q3_K +5 %, Q4_K +3 %, Q5_K +2 %, Q6_K +1 %), plus 16-byte plane padding.

## Reference semantics

`compute::dequantize` (CPU) defines the weights; `Backend::quantized_linear` of the CPU backend defines the product (Q4ZP: `sum_c x[perm[c]] * w[c]`).
The Rust mirror in `weights/mirror.rs` repeats the GLSL chunk decoders word for word and is checked against `compute::dequantize` in `weights/tests.rs`.

## Known weaknesses (L0)

- Never executed: numerical behaviour, bank conflicts, occupancy and the tuning constants (workgroup 64/128, `NROWS` 2, 64x64 tiles) are unmeasured guesses.
- `mm_*_c` assumes the 16x16x16 cooperative-matrix shape and a workgroup of exactly 4 subgroups; devices that only offer other shapes (e.g. 8x8x16) fall back to `_i`/`_s`.
- The `_i` matvec shaders re-read the weight words per chunk (no register blocking across chunks); decode speed relative to ggml-vulkan is unknown.
- f16 overflow in `_c` for activations above 65504 is not guarded.
