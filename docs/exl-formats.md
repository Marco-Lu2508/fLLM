# EXL checkpoint import

This description was written from the upstream encoders and reconstruction kernels
before implementing the readers. The first two sections describe checkpoint bytes, not the shuffled
in-memory representation used by the upstream CUDA GEMM kernels; the native fLLM storage and
kernels are described under "Native storage and inference".

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

## Native storage and inference (GGUF types `EXL2` and `EXL3`)

Since this change the converter (and the direct loader, which writes the same bytes) keeps EXL matrices packed: a matrix
becomes one self-describing blob of the fLLM-specific GGUF tensor type `EXL2` (0x8000_0030) or `EXL3` (0x8000_0031). The blob
layout is defined, and documented byte by byte, in `rust/gguf-probe/src/exl.rs`; the length of a blob is stored in its 64-byte
header because it depends on the bit widths of the matrix (the GGUF probe reads it from there).

* **EXL2 blob**: header, a 16-byte entry per group (`n`, `bits`, `q_scale_max`, word offset, first stored input), the input
  permutation `perm` (`q_perm`, the inverse of the checkpoint's `q_invperm`; `x'[k] = x[perm[k]]`), then one packed row per output
  feature: the LSB-first bitstreams of all groups (each group starts on a 32-bit word, 2..8-bit codes may straddle words) followed
  by the 4-bit scale index of every group. Codes, scale indices and group maxima are copied bit for bit from `q_weight`,
  `q_scale` and `q_scale_max`; nothing is re-quantized, so the original quality is kept exactly. Row order follows the converter's
  row reordering of attention weights (rows are independent), which is why a native tensor can be sliced by rows.
* **EXL3 blob**: header, `suh` and `svh` (F16), an output map (the converter's row reordering is applied *after* the output
  transform, as a scatter, because the 128-point transform mixes outputs) and the trellis tiles of the checkpoint (16x16 tail-biting
  rings of K-bit symbols), ordered tile column by tile column. The codebook (default, `mcg`, `mul1`) and K are in the header.
* **Memory**: EXL2 at 4 bits per weight is about 0.5 byte per weight plus 4 bytes per input column and matrix for the
  permutation; the F16 expansion this replaces took 2 bytes per weight. EXL3 takes K bits per weight (K = 1..8). Nothing but the
  blob is held per matrix; the engine decodes while multiplying.
* **Not native** (still expanded to F16, as before): `--dtype f16|f32`; stacked expert tensors (one blob cannot hold the different
  group structures and permutations of several experts); fractional EXL3 widths, separate `scale` tensors and n-gram rings (rejected
  at import, see above).

Where the blobs run:

| Backend | Behaviour |
| --- | --- |
| CPU (every OS/architecture) | `rust/engine/src/cpu_exl.rs`, portable Rust, parallel over output rows (EXL2) or 16-output tile columns (EXL3). EXL2: gather the activations into stored order, decode each group (2/4/8-bit by word, 3/5/6-bit through a 64-bit bit buffer) into a small scratch vector, dot it with the tokens of the batch, scale once per group. EXL3: scale and Hadamard-transform the input, decode every 16x16 tile (16-bit ring window, 65536-entry codebook table) and multiply it against all tokens at once, transform the result back and scatter through the output map. No dense matrix is built. |
| CUDA (NVIDIA, sm_60 and later) | `rust/engine/cuda/exl.cu`: CUDA-core kernels reading the blob in place. EXL2: one warp per output row and tile of up to 16 tokens. EXL3: input transform kernel, trellis product kernel (one block per 16 outputs, one thread per tile symbol), output transform and scatter kernel; scratch is the stream's activation buffer. No tensor cores yet. |
| Metal (Apple silicon) | EXL2: native kernels (`q_fmt_exl2`, groups of a multiple of 8 inputs; written without a Mac, linted and emulated only; `FLLM_METAL_NO_EXL=1` switches them off, see [gpu-backends.md](gpu-backends.md#metal-block-wise-fp8-bitsandbytes-and-exl2)). EXL3: expanded to F16 at upload like below. |
| AMD (HIP), Vulkan/wgpu, SYCL, TPU, NPU, older NVIDIA | no native kernel: the decoder expands the blob to F16 once, at upload (`compute::dequantize`, one notice on stderr), then uses the normal F16 path. Correct, but not memory-efficient on those devices. |
| Tensor-parallel splitting (`parallel/`) | not supported for EXL blobs (a clear error names the type); use `--dtype f16` for such a run. |

Indicative CPU speed (`cpu_exl::tests::bench_cpu_decode_speed`, one 4096x4096 matrix, 10 threads of a shared build server, so
only an order of magnitude): EXL2 7.2 G weights/s at batch 1 (2.3 ms), 3.3 G/s at batch 8, 0.65 G/s at batch 64; EXL3 (3 bits) 1.4 G
weights/s at batch 1 (12 ms), 1.0 G/s at batch 8, 0.35 G/s at batch 64. That is roughly 1 token/s (EXL2) and 0.2 token/s (EXL3) for a
7B model on such a machine: the kernels are portable scalar code that the compiler vectorizes, not tuned SIMD, and EXL3 pays for a
ring-window extraction and a table lookup per weight. They make every EXL model run on every CPU with the memory of the packed
weights; they are not a speed claim.

Accuracy contract: the importer's expanded path rounds to F16 after every stage; the native kernels apply the group scale and the
Hadamard transforms in F32. The two agree to about 1e-3 relative per weight for EXL2 (F16 rounding of scale and product) and to
about 1e-2 of the matrix RMS for EXL3 (checked per element against the expanded matrix); logits of an end-to-end model agree to
5e-3. Corrupt groups, permutations and output maps are rejected when the weight is loaded, before a kernel sees them.

What is verified, and how:

* Format arithmetic: pack/unpack of every width at every bit offset including words straddled by 3-, 5- and 6-bit codes, group
  offsets, permutation, ring windows and tile positions (`fllm_gguf_probe` `exl::tests`, `cpu_exl::mirror_tests`).
* Converter: direct load equals converted file byte for byte, native and expanded (`direct_format_tests::exl2_*`, `exl3_*`);
  native blob against the importer's expanded matrix for all EXL2 widths and EXL3 widths 1..8 with all codebooks, with and without row
  reordering (`exl2::tests::native_blob_*`, `exl3::tests::native_*`); all 3 x 65536 codebook values equal the importer's.
* CPU kernels against the dequantized matrix for all batch shapes and widths (`cpu_exl::tests`) and an end-to-end synthetic EXL2
  model against an independent dequantization, logits within 5e-3, both native and `--dtype f16`
  (`import_tests::exl2_checkpoint_stays_packed_and_matches_the_dequantized_reference`). An end-to-end EXL3 model needs dimensions
  of 128, larger than the engine test model; EXL3 is covered by the converter and kernel tests above.
* CUDA: `exl.cu` compiles with nvcc 12.8 for sm_75, sm_80, sm_89, sm_90 and sm_120 (and `api.cu` with its wrapper), and the
  `cuda` feature of the engine type-checks. The device arithmetic (bit extraction, scale nibbles, permutation, ring window, codebook
  intrinsics modelled with exact arithmetic, tile placement, Hadamard butterflies, output map) is repeated on the host and compared
  with the CPU kernels for every width and codebook (`cpu_exl::mirror_tests`, including all 3 x 65536 codebook states). **Not yet run on
  a GPU**: the kernels have never executed. `fllm-rs selftest --quants exl2,exl3` (group `linear`, fast cases: odd group counts,
  mixed widths, batches 1..356, EXL3 passes of 128 tokens) compares them with the CPU reference and must be run on an NVIDIA GPU
  before the CUDA path is relied on. Performance is unmeasured; the decode kernels are written for correctness and simplicity (no
  shared-memory staging of the activations, no tensor cores), so expect them to be well below the bandwidth limit.

The MIT notice of ExLlamaV2/V3 below applies to the layout facts and constants that were taken from them; no kernel source code
of either project was copied.

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
