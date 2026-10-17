# Metal kernels for MoE, gated attention, DeltaNet and Mamba - interface

Verification level: **L0** (see [test levels](../../../docs/test-levels.md)). All kernels are syntax-checked with `mslcheck.sh` (clang + shim, coarse) and run **on the CPU** in an emulation
(`emu_recurrent/`, see section 9). **No kernel has ever been compiled under Metal or run on a GPU.** The planning files
(`src/backend/metal/ops/moe_plan.rs`, `recurrent_plan.rs`) are unit-tested in Rust, but only against a local substitute for the types of `ops.rs`
(the non-matmul kernel interface); they compile for `aarch64-apple-darwin`.

The kernels are ports of the Vulkan shaders (`shaders/vulkan/{moe_,delta_,mamba_,misc_split_q_gate,act_sigmoid_mul}*.comp`, same
semantics, same split into kernels); the models for the computation are `cuda/deltanet.cu`, `cpu_delta.rs` and `mamba_forward` / `mamba1_forward` /
`moe_route` in the CPU decoder. No code from MLX, llama.cpp or other third-party sources was taken over (MLX served only as a structural model:
template kernels with `host_name` instantiation).

## 1. Files

| File | Contents |
|---|---|
| `common_rec.metal` | shared code (prefix `r_` / `R_`): activations, threadgroup reductions, byte readers, format traits `r_fmt_*` |
| `moe_route.metal` | `moe_topk` |
| `moe_linear.metal` | `moe_linear_<fmt>` (10 formats) |
| `moe_combine.metal` | `moe_combine` |
| `moe_rows.metal` | `moe_gather_rows`, `moe_scatter_add_rows` |
| `gate_attention.metal` | `gate_split_q`, `gate_sigmoid_mul` |
| `delta_conv.metal`, `delta_rec.metal` | `delta_conv`, `delta_rec_hk{32,64,96,128}` |
| `mamba_conv.metal` | `mamba_conv` |
| `mamba_scan2.metal` | `mamba_scan2_ns{16,32,64,128}`, `mamba_scan2_gen` |
| `mamba_scan1.metal` | `mamba_scan1_ns{8,16,32}`, `mamba_scan1_gen` |
| `mamba_norm.metal` | `mamba_gate_norm`, `mamba_bc_norm` |
| `emu_recurrent/` | CPU emulation of the kernels (section 9) |
| `../src/backend/metal/ops/moe_plan.rs`, `recurrent_plan.rs` | dispatch planning (`OpPlan`) without Metal types |

Layout rules of the Metal core (`API.md` section 4): `common_rec.metal` begins with `common`, so it is shared code and is prepended to every other file;
every other file is its own library and uses only `common.metal`, `common_rec.metal` and its own text. Template kernels
are instantiated with literal `template [[host_name("...")]] [[kernel]] decltype(f<T>) f<T>;` lines (the name scanner finds them). Function
constants are **not** used (`OpPlan` has no field for them): every variant is its own kernel name.

## 2. Conventions of all kernels

- **Constants**: a `struct ..._params` made only of 4-byte words (`uint`, `float`; no padding, little endian), bound with `setBytes` to **buffer index 0**
  (`moe_plan::CONSTANTS_BUFFER_INDEX`). `OpPlan::constants` contains exactly these bytes.
- **Buffers** from index 1 in the order of `OpPlan::slots` (`BufferSlot::index` is the Metal index). All activations are `float` (f32), one
  row per token. The access of each slot (`Read`/`Write`/`ReadWrite`) is **binding**: the hazard tracker places its barriers according to it;
  a wrongly declared slot is a data race. With `FLLM_METAL_SERIAL=1` this class of error disappears.
- **Dummy slots**: an input the kernel does not read (for example `bias_table` without bias) must still be bound to a valid buffer (at least 16 bytes).
- **Host slots** (`SlotSource::Host`): small host arrays (row indices `int32`, weights `float`) that the stream places in a
  buffer for this dispatch (unified memory: `memcpy`; for <= 4096 bytes `setBytes` also works).
- **Threadgroups**: `OpPlan::threadgroups` (number of groups) and `threads_per_group` (from the kernel header). Element-wise kernels run as
  grid-stride loops over `threads_per_grid` (at most 65535 groups x 256 threads), so every grid is correct.
- Every kernel has a machine-readable header in its source `// FLLM-KERNEL: <name> | constants: ... | buffers: n name(r|w|rw), ... | threads: x,y,z`;
  the Rust tests compare every plan with this header **and** with the `struct ..._params` of the same file (names, order, type, index, access).
- Threadgroup reductions run in threadgroup memory (no simdgroup operations assumed, so correct on any GPU, but not optimised for
  Apple silicon).

## 3. Operation -> plan -> kernel

| `Stream` operation | Plan (`moe_plan::` / `recurrent_plan::`) | Kernel |
|---|---|---|
| `moe_topk`, `moe_route` | `plan_moe_topk`, `plan_moe_route` | `moe_topk` |
| `moe_linear` | `plan_moe_linear` (`MoeLinear`, `ExpertAccess`) | `moe_linear_<q4_0\|q4_1\|q5_0\|q5_1\|q8_0\|iq4_nl\|mxfp4\|q4_k\|q5_k\|q6_k>` |
| `moe_combine` | `plan_moe_combine` | `moe_combine` |
| `moe_expert_gemm` | `plan_moe_expert_gemm` (= `moe_linear` + `moe_combine`) | as above |
| `gather_rows`, `scatter_add_rows` | `plan_gather_rows`, `plan_scatter_add_rows` | `moe_gather_rows`, `moe_scatter_add_rows` |
| `split_q_gate`, `sigmoid_gate_mul`, `supports_attention_gate` | `plan_split_q_gate`, `plan_sigmoid_gate_mul`, `attention_gate_supported` | `gate_split_q`, `gate_sigmoid_mul` |
| `delta_layer`, `supports_delta` | `plan_delta_layer`, `delta_supported` | `delta_conv`, `delta_rec_hk<N>` |
| *(no Stream method)* Mamba-2 mixer | `plan_mamba2_mixer` (`plan_mamba_conv`, `plan_mamba2_scan`, `plan_mamba_gate_norm`) | `mamba_conv`, `mamba_scan2_*`, `mamba_gate_norm` |
| *(no Stream method)* Mamba-1 mixer | `plan_mamba_conv`, `plan_mamba1_bc_norm`, `plan_mamba1_scan` | `mamba_conv`, `mamba_bc_norm`, `mamba_scan1_*` |

Not covered: the projections (`x_proj`, `dt_proj`, input/output projections, router) are ordinary `quantized_linear` calls (see `INTERFACE_QUANT.md`).
`upload_pointer_table` / `upload_bias_table` build the tables of section 5; `moe_linear` itself is a matrix-vector kernel, not a grouped GEMM:
for long prompts the stream should group the tokens per expert (`gather_rows` + `quantized_linear` + `scatter_add_rows`), as the CPU decoder does.

## 4. The kernels in detail

The headers of the `.metal` files are authoritative; this is the short form (constants in struct order, buffers from index 1, threads per group, grid in groups).

| Kernel | Constants | Buffers | Threads | Grid |
|---|---|---|---|---|
| `moe_topk` | rows, experts, k, normalize_scores, scale(f) | logits(r), ids(w, int32), weights(w) | 128 | [rows,1,1] |
| `moe_linear_<fmt>` | rows, cols, slots, x_div, n_experts, flags, nb | table(r), bias_table(r), ids(r), x(r), out(w), arena(r), bias_arena(r) | 64 | [rows, slots, 1] |
| `moe_combine` | tokens, k, hidden | out(w), y(r), weights(r) | 256 | [min(ceil(tokens*hidden/256), 65535),1,1] |
| `moe_gather_rows` | n, row_len, has_weights | dst(w), src(r), idx(r), weights(r) | 256 | as above with n*row_len |
| `moe_scatter_add_rows` | n, row_len, has_weights | dst(rw), src(r), idx(r), weights(r) | 256 | as above |
| `gate_split_q` | total, heads, dk, per_head | projected(r), q(w), gate(w) | 256 | ceil(total/256) |
| `gate_sigmoid_mul` | count | x(rw), gate(r) | 256 | ceil(count/256) |
| `delta_conv` | conv_dim, kh, hk, ksz, row0, n, reset, hist_off, qscale(f) | qkv(r), conv(r), state(rw), mixed(w) | 128 | [2*kh + ceil(vh*hv/128), ceil(n/16), 1] per run |
| `delta_rec_hk<N>` | kh, vh, hv, conv_dim, tiled, row0, n, reset, state_off, beta_scale(f), eps(f) | mixed(r), z(r), beta(r), alpha(r), dt_bias(r), decay(r), norm(r), state(rw), out(w) | 256 | [vh,1,1] per run |
| `mamba_conv` | channels, ksz, in_stride, in_off, row0, n, reset, hist_off, has_bias | x(r), convw(r), convb(r), state(rw), mixed(w) | 64 | [ceil(channels/64), ceil(n/16), 1] per run |
| `mamba_scan2_*` | inner, ns, heads, groups, head_dim, conv_dim, proj_stride, proj_dt_off, row0, n, reset, state_off, step_min(f) | mixed(r), projected(r), dt_bias(r), decay(r), skip(r), state(rw), out(w) | 64 | [heads, ceil(head_dim/64), 1] per run |
| `mamba_scan1_*` | inner, ns, rank, split_stride, proj_stride, proj_gate_off, row0, n, reset, state_off | mixed(r), dt(r), split(r), projected(r), decay(r), skip(r), state(rw), out(w) | 64 | [ceil(inner/64),1,1] per run |
| `mamba_gate_norm` | inner, norm_groups, rows, proj_stride, proj_gate_off, eps(f) | y(rw), projected(r), norm(r) | 256 | [norm_groups, rows, 1] |
| `mamba_bc_norm` | rows, rank, state, split_stride, eps(f) | split(rw) | 128 | [3, rows, 1] |

Limits: `moe_topk` at most 2048 experts and k <= 32; `delta_rec` key width 32/64/96/128 and value width <= 128; convolution kernel 2..8;
Mamba state size <= 256 (register variants for 16/32/64/128 and 8/16/32, otherwise `_gen`). The planners check this and return `UnsupportedOperation`.

### Semantics in brief

- `moe_topk`: softmax over all experts, the `k` largest (ties: smaller index), with `normalize_scores` divided by `max(sum, 1e-8)`, times `scale`
  (`moe_route` = normalised, scale 1). Output sorted in descending order.
- `moe_linear`: `out[slot][r] = dot(W[ids[slot]][r], x[slot / x_div]) (+ bias)`; an expert number outside `0..n_experts` gives 0 (like the CPU backend).
- `delta_*`: convolution over `[q|k|v]`, SiLU, L2 normalisation of the q/k heads (`qscale = hk^-1/2` for q), delta rule `S = S*exp(decay*softplus(alpha+dt_bias)); S += k (x) ((v - S^T k)*sigma);
  o = S^T q`, gated RMS norm `o * rsqrt(mean(o^2)+eps) * norm * silu(z)`; key head of value head `j`: `tiled ? j % kh : j / (vh/kh)`.
- `mamba_*`: as `mamba_forward` / `mamba1_forward` (convolution with history, selective scan, gated group RMS norm; for Mamba-1 an optional RMS norm of dt/B/C).

## 5. Expert weights and tables (`moe_linear`)

**Assumption about the weight layout (the quantized-linear layout may differ):** the weight of an expert is the raw GGUF tensor, row by row, `rows x (cols / QK)`
blocks exactly in the GGUF block structure (little endian, no padding, no repacking):

| Format | QK | Bytes | Layout |
|---|---|---|---|
| Q4_0 | 32 | 18 | `half d; uchar qs[16]` - `w = d*(q-8)`, nibbles 0..15 low, 16..31 high |
| Q4_1 | 32 | 20 | `half d, m; uchar qs[16]` - `w = d*q + m` |
| Q5_0 | 32 | 22 | `half d; uchar qh[4], qs[16]` - `w = d*(q-16)`, 5th bit from `qh` |
| Q5_1 | 32 | 24 | `half d, m; uchar qh[4], qs[16]` |
| Q8_0 | 32 | 34 | `half d; char qs[32]` |
| IQ4_NL | 32 | 18 | `half d; uchar qs[16]` - `w = d*kvalues_iq4nl[q]` |
| MXFP4 | 32 | 17 | `uchar e; uchar qs[16]` - `w = 2^(e-128)*fp4[q]` (E2M1 doubled) |
| Q4_K | 256 | 144 | `half d, dmin; uchar scales[12], qs[128]` |
| Q5_K | 256 | 176 | `half d, dmin; uchar scales[12], qh[32], qs[128]` |
| Q6_K | 256 | 210 | `uchar ql[128], qh[64]; char scales[16]; half d` |

The blocks are read **byte-exactly** (17- and 18-byte blocks are not 4-byte aligned). Dequantisation matches
`compute::dequantize` / `quant_extra` / `quant_iq` (CPU reference); the emulation checks this against the real Rust dequantiser.
Not covered (planner: `UnsupportedQuant`): float formats, Q2_K, Q3_K, the remaining IQ formats, NVFP4, Q4Z/Q4ZP, FP8, `Q4_K_Marlin`.

Two ways to reach the experts (pipeline and kernel are the same; `flags` bit 1 selects):

1. **Arena** (`ExpertAccess::Arena`): all experts of a projection lie in **one** buffer (`arena`, buffer 6); `table[e]` = **byte offset** of expert `e`
   inside it (`ulong`). The bias vectors (`rows` floats each, 4-byte aligned) lie in `bias_arena` (buffer 7), `bias_table[e]` = byte offset.
   Helpers: `moe_plan::expert_bytes`, `arena_offsets(sizes, align)` (experts at multiples of 256, bias at 4), `pack_u64`. The buffer is bound as a whole
   (limit `maxBufferLength`; 128 experts of 768x2048 Q4_K = about 113 MiB).
2. **Address table** (`ExpertAccess::Address`): `table[e]` / `bias_table[e]` = absolute 64-bit GPU address of the weight/bias buffer of expert `e`
   (`AddressTable::build` from `buffers.rs`, i.e. `upload_pointer_table`); the stream must make the experts resident with `useResource`
   (`builder.table(...)`), `arena`/`bias_arena` are dummies.

## 6. State layouts (recurrent layers)

- **DeltaNet** (`DeltaShape`): in the state buffer of a sequence the state of the layer begins at `state_offset` bytes: first the recurrent matrices
  `S[value head][key row][value column]` (f32, `rec_bytes()`), then the convolution history `history[channel][k]` (`conv_dim x (kernel-1)`, the k-th last
  input word, oldest first). This is the CUDA layout; the CPU decoder uses its own, and the two are never mixed. `layer_state_bytes()` is rounded up to
  256; `state_offset` must be divisible by 4 (the plans compute in float indices).
- **Mamba** (`Mamba2Shape`, `Mamba1Shape`): first the history `[channel][k]`, then the state (`[head][channel][state]` for Mamba-2, `[channel][state]` for
  Mamba-1), `layer_state_bytes()` rounded up to 256. The offsets are parameters of the plans.
- `reset = 1`: the old state is not read (the emulation fills it with NaN to prove this); a run that is not the first of a sequence sets
  `reset = 0` and continues the state.
- A run is one dispatch pair per sequence (each sequence has its own state buffer, `SlotSource::RunState(i)`); runs are independent and may execute in parallel.
  Within a run, threadgroups with y > 0 (token 16...) process the convolution without reading the history; only group y = 0 writes the new history back.

## 7. Basis for `Stream::supports_delta` (Metal)

`recurrent_plan::delta_supported(&DeltaShape)` is `true` if all of the following hold:

1. `k_dim` in {32, 64, 96, 128} (one kernel `delta_rec_hk<N>` each; half of a value column lies in registers),
2. `1 <= v_dim <= 128` (two threads per value column, 256 threads per threadgroup),
3. `k_heads > 0`, `v_heads > 0`, `v_heads` a multiple of `k_heads`,
4. `2 <= conv_kernel <= 8` (the history of at most 7 values per channel lies in registers),
5. `conv_dim` and the head count fit in the 32-bit indices and the grid.

In addition the stream must keep the weights (`conv`, `dt_bias`, `decay`, `norm`) as f32 device buffers and size the state buffer of every run to at least
`state_offset + layer_state_bytes()` bytes (not checkable here). `supports_delta` should return exactly `delta_supported`; `supports_attention_gate`
is always `true` for the gate kernels (`attention_gate_supported`).

## 8. Numerics

Everything in f32. `exp`/`log` of the gates (`forget`, `da`, `softplus`, softmax) use `precise::` so that fast math does not degrade the accuracy of the recurrent factors;
`exp` arguments of the sigmoid/SiLU functions are clamped at 80 so that `inf` never arises. The summation order (tree reduction across
threads) differs from the CPU; the emulation tolerates 1e-4..3e-4 relative against a double-precision reference. Nothing was measured against real models (L2 is missing).

## 9. CPU emulation and tests (`emu_recurrent/`)

`run_emulation.sh <vectors.bin>` compiles the **real kernel text** (after a sed rewrite of the attributes) with a C++ compiler: every GPU thread is a
`std::thread`, threadgroup memory a static array, `threadgroup_barrier` a real barrier. With `TSAN=1` (ThreadSanitizer) a missing barrier
becomes a reported data race. The comparison is against independent double-precision references (ports of `mamba_forward`, `mamba1_forward`, `cpu_delta.rs`,
`moe_route`); the weights of `moe_linear` come from the real Rust dequantiser (`harness/src/bin/gen_vectors.rs`, random blocks of all ten formats).
Tested are single-pass runs, multiple runs (history/state across run boundaries), token by token, with a NaN-prefilled state on `reset`, with a state offset,
both expert accesses, with/without bias, `x_div > 1` and invalid expert numbers. `mutation_check.sh` builds deliberately faulty kernels and requires
the emulation to detect each one.

The emulation does **not** check: acceptance by the MSL compiler (address spaces, attributes, `[[kernel]]` instantiation), register pressure / threadgroup memory / GPU limits,
the behaviour of fast math on the GPU, performance. A test run on a Mac covers those.
`harness/` contains the test crate that compiles and tests the planning files against the real types of `fllm-engine` with a substitute for `ops.rs`
(the module `backend::metal` is only compiled for Apple targets).

## 10. Unverified assumptions and known weaknesses (to check on hardware)

1. Nothing ran under Metal. Possible compile errors the shim does not see: `static constant constexpr const` data members in structs, integer-to-pointer casts
   `(device const uchar *)ulong`, `template [[host_name]] [[kernel]] decltype(...)` instantiations, `threadgroup` fields in template kernels, `_Pragma("clang loop unroll(full)")`.
2. The weight layout (raw GGUF blocks) is an assumption relative to the quantized-linear kernels; if that layout changes, only the `r_fmt_*` traits change.
3. `moe_linear`: 64 threads per (row, slot), byte-wise reading of the blocks; not tuned for bandwidth (vector loads, simdgroup reduction, several rows per group are missing).
4. `delta_rec`: HK=128 keeps 64 floats per thread in registers plus 3 KiB of threadgroup memory; whether the compiler spills is unknown. A long run of one head is serial.
5. Mamba has no `Stream` method; the plans are backed only by unit tests and emulation. `mamba_conv`/`mamba_scan*` put the intermediate buffers (`mixed`) in scratch.
6. `BufferSlot`/`OpPlan` in `ops.rs` come from the non-matmul kernel interface; the plans construct `BufferSlot` in exactly one place (`moe_plan::make_slot`), field names: `index`, `name`, `access`, `source`.
7. Barriers between dispatches (`delta_conv` -> `delta_rec`, `moe_linear` -> `moe_combine`) depend on the declared accesses of the slots (section 2).
