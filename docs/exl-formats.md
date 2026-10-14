# EXL checkpoint import

This description was written from the upstream encoders and reconstruction kernels
before implementing the readers. It describes checkpoint bytes, not the shuffled
in-memory representation used by the CUDA GEMM kernels.

## Sources and confirmed EXL2 layout

ExLlamaV2 revision `7dc12af3a81f34ac3f27cd7602ed539b638933ca`:
[adaptivegptq.py](https://github.com/turboderp-org/exllamav2/blob/7dc12af3a81f34ac3f27cd7602ed539b638933ca/exllamav2/conversion/adaptivegptq.py),
[pack_tensor.cu](https://github.com/turboderp-org/exllamav2/blob/7dc12af3a81f34ac3f27cd7602ed539b638933ca/exllamav2/exllamav2_ext/cuda/pack_tensor.cu),
[ext.py](https://github.com/turboderp-org/exllamav2/blob/7dc12af3a81f34ac3f27cd7602ed539b638933ca/exllamav2/ext.py),
and `cuda/q_matrix.cu`, `cuda/quant/qdq_util.cuh` in the same revision.

For a linear weight `[out, in]`, `q_weight` is I32 `[packed_input_rows, out]`.
Each output column has an independent little-endian bitstream along the input
axis. `q_groups` is an I16 vector of pairs `(bits, packed_row_start)`; starts
refer to rows of q_weight, not byte offsets. Consecutive starts determine group
lengths. Widths implemented upstream are 2, 3, 4, 5, 6 and 8; 7 is unsupported.
A single matrix can contain groups of different widths.

`q_scale` is I32 `[groups, out / 8]`, eight low-to-high nibbles per word.
For stored nibble s, the scale is `(s + 1)^2 * (q_scale_max[g] / 256)`.
q_scale_max is F16 `[groups]`. The division, scale multiplication and final
`(code - 2^(bits-1)) * scale` use F16 rounding in the upstream reconstruction.
q_invperm is I32 `[in]`: original input index -> packed input index. Optional
q_perm is its inverse. This permutes input rows of the upstream `[in, out]`
matrix, equivalently columns of the conventional `[out, in]` weight. It does
not permute output channels. Stored q_weight has not undergone the CUDA runtime
shuffle. Output padding is present in the encoder; the importer must reject
dimensions inconsistent with the architecture instead of silently inventing a
crop. q_groups offsets are unsigned bit patterns in I16 storage.

## Sources and confirmed EXL3 layout

ExLlamaV3 revision `16a49792a3c93d8432d72e6c4bce800841566577`:
[quantize.py](https://github.com/turboderp-org/exllamav3/blob/16a49792a3c93d8432d72e6c4bce800841566577/exllamav3/modules/quant/exl3_lib/quantize.py),
[exl3.py](https://github.com/turboderp-org/exllamav3/blob/16a49792a3c93d8432d72e6c4bce800841566577/exllamav3/modules/quant/exl3.py),
and `exllamav3_ext/quant/{pack.cu,exl3_dq.cuh,codebook.cuh,reconstruct.cu}`.

trellis is I16 `[in/16, out/16, 16*K]` for integer K=1..8. Each 16x16
tile is an independent tail-biting ring of 256 symbols. The packer appends the
low K bits of each state MSB-first, then swaps the two u16 halves of each u32.
The state at symbol t is the preceding 16 bits ending at `(t+1)*K`, wrapping
within the tile. These are overlapping codebook indices, not independent K-bit
integer weights. The encoder's tensor_core_perm maps index `8*lane+j` to:
rows `2*(lane%4)+[0,1,8,9,0,1,8,9]`, columns
`lane/4+[0,0,0,0,8,8,8,8]`.

Default codebook: wrapping u32 `x = state*89226354 + 64248484`, then
`x = (x & 0x8fff8fff) ^ 0x3b603b60`; interpret both halves as F16 and
add with F16 rounding. Presence of mcg selects multiplier 0xCBAC1FED and
no additive term. Presence of mul1 selects 0x83DCD12D; sum its four unsigned
bytes plus 0x6400, interpret as F16, then perform a half FMA with half constants
0x1eee and 0xc931. Marker tensor values are not read upstream. Reject conflicting
markers instead of choosing an undocumented precedence.

Reconstruct the rotated matrix R in `[in,out]` order. The original matrix is
`diag(suh) * H128 * R * H128 * diag(svh)`, using independent normalized
Sylvester 128 blocks on each axis. Upstream converts each Hadamard result to
F16, multiplies the corresponding vector in F16, then proceeds to the other
axis. Modern suh/svh are F16 vectors containing scales as well as signs.
Older su/sv are I16 bitfields (bit zero is the first sign, set means negative).

### Explicit import boundaries

Integer trellis widths and the three codebooks above are source-confirmed.
Fractional 1.5/2.5/3.5-bit mul1 uses a different alternating transition mask;
its reconstruction is deliberately left open for a separate reviewed reader.
Deprecated separate `scale`, unpacked `no_pack` conversion intermediates,
non-128-divisible linear dimensions and n-gram embedding rings are not covered
by the linear reader. Unsupported representations must fail with the tensor
name. CPU Hadamard accumulation order may differ from upstream CUDA/PyTorch;
bitwise equality of transformed F16 values is not established.

Both imports should retain mmap-backed payloads and decode bounded chunks or
128x128 transformation tiles; no complete model or full dequantized matrix is
needed. Architecture names and mappings remain the existing conversion plan's
responsibility. Preserve quantization_config as provenance, including bits,
head_bits and calibration, without using average bits to decode any matrix.

## Phase 2: native EXL2 CPU design (no kernel implementation)

Keep q_weight, group offsets, input permutations, scale nibbles and maxima in
their original mapped representation. For GEMV, first gather input activations
into packed-input order using q_perm (or derive it from q_invperm once). Traverse
groups in their stored order and dispatch a width-specific decoder for
2/3/4/5/6/8 bits. Process a small tile of output columns; each packed row then
supplies consecutive u32 words to SIMD lanes. Handle cross-word 3/5/6-bit codes
using the checkpoint bitstream rather than the upstream GPU runtime shuffle.

Decode group scales once per output tile. Convert centered codes to vector
floats and accumulate dot products without materializing a dense weight matrix.
A fidelity mode can reproduce the importer's intermediate F16 rounding; a faster
F32 scale multiply needs a separately stated accuracy contract. Prefill/GEMM can
reuse the same decoder into a bounded micro-panel consumed immediately by a
matrix multiplication kernel. Dispatch based on matrix dimensions and row count,
with a scalar fallback for tails. Cache only descriptors and bounded panels.

This is a design, not a performance result. Alignment, SIMD widths, group ordering,
permutation gather cost and interaction with the existing thread pool need
measurement in a later task authorized to run models and benchmarks. Corrupt
groups and permutations must be validated before entering any kernel.

## MTP / NextN draft source

Input semantics were checked against llama.cpp revision
`d0b490f25edec147b4dcd69cce31c7d1099dff1d`:
[deepseek2.cpp](https://github.com/ggml-org/llama.cpp/blob/d0b490f25edec147b4dcd69cce31c7d1099dff1d/src/models/deepseek2.cpp),
[glm4-moe.cpp](https://github.com/ggml-org/llama.cpp/blob/d0b490f25edec147b4dcd69cce31c7d1099dff1d/src/models/glm4-moe.cpp),
and Transformers revision `e598fbad926d80bc356c0c4d96532030e3dfdb62`:
[MtpLayer/MtpModel](https://github.com/huggingface/transformers/blob/e598fbad926d80bc356c0c4d96532030e3dfdb62/src/transformers/modeling_layers.py).

The main decoder exports h_nextn **after** its output RMS norm. For target
position p, the first MTP block receives that state and the embedding of the
chosen token at p+1. Apply separate hnorm/enorm RMS norms, concatenate
`[normalized_embedding, normalized_hidden]`, then apply eh_proj from 2h to h.
Run the full appended attention/FFN block at RoPE position p+1, its shared-head
norm (fallback: output_norm), and its shared vocabulary head (fallback: main
head). This yields the proposal at p+2. Transformers supports a reversed
concatenation flag for other checkpoints; that variant is outside the two
families supported here.

`--speculative mtp` is opt-in. DeepSeek-V2/V3 and GLM-4 MoE (including the MLA
family mapping) use the first NextN block and propose one token at a time. The current implementation requires CPU execution; GPU hidden-state capture awaits
integration with the separately developed GPU backend and is rejected explicitly.
The existing scheduler verifies it by target greedy argmax. Non-greedy and
constrained requests use ordinary target decoding; stateful DeltaNet/Mamba
models and combinations with an external draft or lookup speculation are
rejected. Additional trained prediction blocks are retained but not used in
this first implementation; recurrent application of a single block is not
assumed equivalent to additional trained heads.

MTP owns a separate one-layer paged KV cache. Warm it with shifted tokens and
committed main-model states, including prompt chunks. Hold the final prompt
state until the chosen next token is available. Never warm it with rejected
verification rows. Logical MTP slot p has RoPE position p+1, eliminating a
fictional cache entry at position zero. Prefix reuse is disabled while MTP is
enabled because main KV prefix entries do not contain MTP KV/residual states.
Preemption rewinds and recomputes that cache from the prompt.

New conversion maps NextN attention/FFN and expert matrices through the normal
architecture mapping. Earlier imports with raw HF NextN names are supported
through a private tensor-alias view sharing the original mmap; individual
expert matrices can be read directly. Conflicting aliases or missing required
tensors produce named errors. No GGUF rewrite is required for these aliases.

Prometheus exposes proposed/accepted MTP token counters and an acceptance ratio.
They count candidates actually scheduled and accepted, not warmup tokens or
proposals discarded for scheduler budget limits. End-to-end acceptance,
equivalence to upstream logits, GPU execution and performance remain unmeasured;
no tests, models or benchmarks were executed for this change.

## Upstream license

EXL3 codebook constants and layout are derived from ExLlamaV3 (MIT):

Copyright (c) 2025 Turboderp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
