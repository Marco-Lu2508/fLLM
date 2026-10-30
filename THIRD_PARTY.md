# Third-party code, licences and terms

This file covers every backend and tool of fLLM: the CPU kernels (scalar, AVX2, AVX-512/AMX, ARM), CUDA, HIP (AMD),
Metal, Vulkan/wgpu, SYCL, the NPU backends (Intel OpenVINO, AMD XDNA, Qualcomm Hexagon, Apple Neural Engine), the TPU
backend (PJRT) and the build, release and CI tooling. fLLM itself is Apache-2.0 (`LICENSE`).

How to read it:

* "copied" means source text of the upstream file is in this repository (with its licence header, kept).
* "adapted" means a kernel or function was derived from upstream source and then changed; the changes are listed in the file header.
* "layout facts / constants" means data that defines a file format or an interface (block layouts, codebooks, struct member order,
  enum values) was restated; the code around it is fLLM's own.
* "design reference" means the technique was read and re-implemented; no source text was taken.
* "loaded at run time" means fLLM opens the library with `dlopen` / `LoadLibrary`; it is not linked and not shipped.

Licence texts are in `licenses/`. The modifications of Apache-2.0 licensed files are summarised in `NOTICE`.
Rust crate licences are checked by `cargo deny` (`deny.toml`, workflow `supply-chain.yml`); a summary is in
`docs/changes/hygiene-a.md` and the exact versions are in `Cargo.lock`.

## Inventory by upstream

| Upstream | Licence | Text in `licenses/` | Nature of use |
| --- | --- | --- | --- |
| ggml / llama.cpp (ggml authors) | MIT | `ggml-MIT.txt` | layout facts / constants, tables adapted into Rust and generated for CUDA |
| vLLM | Apache-2.0 | `vLLM-Apache-2.0.txt` | copied (Marlin), adapted (HIP skinny GEMM, NVFP4 configuration), design reference |
| Marlin (IST-DASLab, Elias Frantar; Neural Magic changes inside vLLM) | Apache-2.0 | `MARLIN-Apache-2.0.txt` | adapted (Q4_K kernel), copied through vLLM |
| FlashInfer | Apache-2.0 | `FlashInfer-Apache-2.0.txt` | header-only dependency (not copied), adapted configuration and call sequence |
| CUTLASS (NVIDIA) | BSD-3-Clause | `CUTLASS-BSD-3-Clause.txt` | header-only dependency (not copied), adapted collective builders |
| MLX (Apple) | MIT | `MLX-MIT.txt` | design reference with notice kept in three `.metal` files |
| ExLlamaV3 (turboderp) | MIT | `ExLlamaV3-MIT.txt` | layout facts / codebook constants |
| rocWMMA (AMD) | MIT | `rocWMMA-MIT.txt` | header library from the ROCm install (not copied) |
| OpenXLA PJRT / XLA | Apache-2.0 | `OpenXLA-Apache-2.0.txt` | layout facts / constants transcribed to Rust |
| OpenVINO (Intel) | Apache-2.0 | `OpenVINO-Apache-2.0.txt` | C API signatures redeclared, library loaded at run time |
| mlir-aie / IRON, Peano (AMD) | Apache-2.0 with LLVM exceptions | `mlir-aie-Apache-2.0-LLVM-exception.txt` | the NPU programs in `rust/engine/xdna/kernels/` are compiled output of its GEMM design (redistributed); the tools are downloaded by a script, not shipped |
| XRT (AMD / Xilinx) | Apache-2.0 | – (not copied) | public C++ headers used to build the shim; library loaded at run time |
| ik_llama.cpp, SGLang, FlashAttention, ONNX, onnxruntime, coremltools, StableHLO | MIT / Apache-2.0 / BSD-3-Clause | not needed | design or format reference only, no source text |
| Qualcomm QAIRT (QNN, Genie) | proprietary | none | binary interface restated, SDK not distributed; terms not checked (see below) |
| AMD Ryzen AI Software | proprietary | none | documentation only; terms not checked (see below) |
| Intel oneAPI DPC++ / SYCL runtime | Apache-2.0 with LLVM exceptions (compiler), Intel EULA (runtime redistributables) | none | build tool, not distributed; redistribution not examined (see below) |
| NVIDIA CUDA toolkit | NVIDIA CUDA EULA | none | build tool; the release binary links the static CUDA runtime, the driver is the user's |

### ggml / llama.cpp: GGUF layouts, dequantization and codebooks

Upstream: https://github.com/ggml-org/llama.cpp (ggml), commit b1ff4ca23630ac7c5a275405353ddb5c486332c9 (the Q2_0 layout
also from 58cb9138e46180d45854cf45d4c1e25ad5f7e1c4). Licence: MIT, Copyright (c) 2023-2026 The ggml authors
(`licenses/ggml-MIT.txt`). No other llama.cpp source is copied. The block layouts and dequantization equations are the GGML
format definition, a GGUF file is only decodable with them.

| File | What was taken |
| --- | --- |
| `rust/engine/src/quant_iq_tables.rs` | integer codebooks of `ggml-common.h` (IQ1/IQ2/IQ3 grids, sign tables, `kvalues_iq4nl`), generated; header: "Portions Copyright (c) The ggml authors, MIT" |
| `rust/engine/src/quant_iq.rs` | IQ dequantizers following `ggml-quants.c` (adapted into Rust) |
| `rust/engine/src/quant_extra.rs` | further scalar and K-quant decoders (for example Q2_K, Q3_K, IQ4_XS), adapted from the ggml implementation |
| `rust/engine/cuda/iq_tables.cuh` | the same codebooks, generated from `quant_iq_tables.rs` |
| `rust/engine/cuda/iq_decode.cuh` | device decoders following the Rust reference line by line |
| `rust/engine/cuda/kv_quant.cu`, `metal/common.metal`, `metal/common_ops.metal`, `metal/common_rec.metal`, `shaders/vulkan/_ops_wq.glsl`, `shaders/vulkan/_q_fmt_iq4_nl.glsl`, `src/backend/vulkan/weights/mirror.rs`, `src/cpu_arm/formats.rs`, `src/kv_format.rs`, `rust/gguf-probe/src/quant_types.rs` | the 16-entry `IQ4_NL` grid and block sizes / type ids of the GGUF format (format facts) |
| `rust/engine/src/backend/**`, `cuda/*.cu`, `metal/*.metal`, `shaders/vulkan/*` | quantized block layouts and dequantization equations as used by the files above; the kernels were written for fLLM and are checked against the Rust reference decoders by `selftest-gpu` |

GGUF metadata key names (`tokenizer.ggml.*`, `*.rope.*`, ...) and the llama.cpp-compatible HTTP routes
(`rust/engine/src/server/api/llamacpp.rs`) are interface names; no code is taken. Format description:
https://github.com/ggml-org/ggml/blob/master/docs/gguf.md.

Because the MIT licence requires its notice to travel with copies of the tables, the notice is `licenses/ggml-MIT.txt`.
Binary and source distributions of fLLM carry the `licenses/` directory.

### vLLM

https://github.com/vllm-project/vllm, Apache-2.0, "Copyright contributors to the vLLM project" (NVIDIA contributions: "Copyright (c) 2025, NVIDIA CORPORATION").
Licence text `licenses/vLLM-Apache-2.0.txt`.

| File | Upstream file and commit | Nature |
| --- | --- | --- |
| `rust/engine/cuda/marlin_vllm/marlin_template.h`, `dequant.h`, `marlin_mma.h`, `marlin_dtypes.cuh`, `marlin.cuh`, `kernel.h`, `core/scalar_type.hpp` | `csrc/libtorch_stable/quantization/marlin/`, commit 9367d8b9693c3d2fb6c3f99d8ccb4aa7673c0981 | copied; `core/scalar_type.hpp` has the PyTorch exception include replaced by `torch_shim.h` |
| `rust/engine/cuda/marlin_vllm/generate_kernels.py` | same directory | copied, restricted to FP16 activations and int4 / int8 / fp8 (kFE4M3fn) / fp4 weights, FP8-activation clause removed |
| `rust/engine/cuda/marlin_vllm/sm75_kernel_*.cu`, `sm80_kernel_*.cu`, `kernel_selector.h` | output of `generate_kernels.py` | generated |
| `rust/engine/cuda/marlin_vllm/marlin_vllm.cu` | `marlin.cu` | adapted: PyTorch tensor glue replaced by a C entry point |
| `rust/engine/cuda/marlin_vllm/marlin_vllm_awq_repack.cu`, `marlin_vllm_gptq_repack.cu` | `awq_marlin_repack.cu`, `gptq_marlin_repack.cu` | adapted: same glue change |
| `rust/engine/hip/hip_skinny.cu` | `csrc/rocm/skinny_gemms.cu` (`wvSplitK_hf_`), commit 7fad7db3309347a9d6288a09ed0993728e474a34 | adapted (changes in the file header) |
| `rust/engine/hip/hip_q4z.cu` (kernel `q4z_int4_sml_kernel`) | `csrc/rocm/skinny_gemms_int4.cu` (`wvSplitK_int4_hf_sml_`), same commit | adapted (changes in the file header); the rest of the file is fLLM's |
| `rust/engine/cuda/nvfp4_cutlass.cu`, `nvfp4_quant.cuh` | `csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu`, `csrc/quantization/fp4/nvfp4_quant_kernels.cu`, `modelopt.py` | adapted configuration and arithmetic of the activation quantization recipe (UE4M3 block scales of `gs * amax / 6`, E2M1 round-to-nearest-even, `input_global_scale = 1 / input_scale`) |
| `rust/engine/cuda/fused.cu`, `src/mem/block_allocator.rs`, `src/mem/prefix_cache.rs` | `fused_add_rms_norm`, `reshape_and_cache`, `silu_and_mul`, BlockPool | design reference; arithmetic repeated for bit-identical results, allocator all-or-nothing behaviour matches `BlockPool.get_new_blocks`; no source copied |

The Marlin files taken from vLLM carry `SPDX-License-Identifier: Apache-2.0` and the vLLM / Neural Magic / Marlin copyright headers in `rust/engine/cuda/marlin_vllm/`. Scale / zero-point permutations of `marlin_utils.py` are reimplemented. The port was checked against an FP32 reference. vLLM `csrc/rocm/attention.cu`, `q_gemm_rdna3*.cu` and
`moe_q_gemm_rdna3.cu` are not used. The `Q4Z` tensor format, converter, weight upload and activation glue
(`marlin_vllm_glue.cu`) are fLLM's (docs/awq-gptq-marlin.md).

### Marlin (IST-DASLab)

https://github.com/IST-DASLab/marlin, Apache-2.0, Copyright (C) Marlin.2024 Elias Frantar (`licenses/MARLIN-Apache-2.0.txt`).

| File | Nature |
| --- | --- |
| `rust/engine/cuda/marlin_q4k.cu` | adapted from `marlin_cuda_kernel.cu` for GGUF Q4_K grouped affine quantization; header kept, changes listed in it |
| `rust/engine/cuda/marlin_vllm/*` | Marlin as modified by Neural Magic and vLLM (see above) |

`marlin_q4k.cu`: on Ampere and newer GPUs (sm_80, sm_86, sm_89, sm_90, sm_120) Q4_K matrix multiplications run through the Marlin kernel design.

- Weights are repacked once at upload time (`q4_k_marlin_device_layout` in `backend/cuda.rs`) into Marlin's 16x64 tile
  layout directly ordered for `mma.m16n8k16` tensor core fragments.
- Sub-block scales and minimums (group size 32) are extracted and stored in permuted 8x8 layout; dequantization runs
  directly in registers via `lop3` into unsigned fp16 in [0..15], fused with scale and min via a single `hfma2` instruction.
- Matrix outputs accumulate in FP32 with optional fused bias addition.
- `FLLM_CUDA_NO_MARLIN=1` falls back to the in-place Q4_K GEMM (`qgemm4k.cu`).

### FlashInfer

https://github.com/flashinfer-ai/flashinfer, Apache-2.0, Copyright (c) 2023 by FlashInfer team; the fp4 GEMM templates are
"Copyright (c) 2020-2023, NVIDIA CORPORATION" (`licenses/FlashInfer-Apache-2.0.txt`). Version used: v0.6.5.

| File | Nature |
| --- | --- |
| `rust/engine/cuda/flashinfer_prefill.cu` | host glue for FlashInfer's batched prefill kernel and work planner (`include/flashinfer/attention/*.cuh`), following `csrc/batch_prefill.cu` and `csrc/single_prefill.cu`: page-table setup, FP32 <-> FP16 conversion; the kernel itself is a header-only dependency compiled into the binary |
| `rust/engine/cuda/nvfp4_cutlass.cu` | epilogue and tile list of `include/flashinfer/gemm/fp4_gemm_template_sm120.h` and `fp4_gemm_cutlass_template_sm120.h` (no source tensor C when the product is only stored, an explicit epilogue schedule, the 128x128x256 tile, the 256x128 / 128x256 / 128x64 tiles, the stream-K scheduler) and the autotuner idea (time each candidate once per problem shape) |
| `rust/engine/cuda/fused.cu` | design reference (`apply_rope_pos_ids`), no source |

FlashInfer is not part of this repository: `build.rs` uses a checkout in `third_party/flashinfer` or `FLLM_FLASHINFER_DIR`
(`git clone --depth 1 --branch v0.6.5 https://github.com/flashinfer-ai/flashinfer third_party/flashinfer`). Without it the built-in
tensor-core attention (`attention3.cu`) is used; `FLLM_CUDA_NO_FLASHINFER=1` switches it off. Binaries built with it contain
FlashInfer templates and must carry the Apache-2.0 text and the notices of `NOTICE`. SGLang (Apache-2.0) is a design reference
(RadixCache: longest page-aligned prefix matching, shared-path splitting, leaf eviction); no SGLang source is copied.

### CUTLASS

https://github.com/NVIDIA/cutlass v4.8.0, BSD-3-Clause, Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES
(`licenses/CUTLASS-BSD-3-Clause.txt`). `rust/engine/cuda/nvfp4_cutlass.cu` adapts the collective builders of the example
`examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu` (Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES, SPDX BSD-3-Clause).
CUTLASS itself (header-only templates) is not in the repository: `build.rs` uses a checkout in `third_party/cutlass` or `FLLM_CUTLASS_DIR`
(`git clone --depth 1 --branch v4.8.0 https://github.com/NVIDIA/cutlass third_party/cutlass`). Binaries built with it contain CUTLASS templates and
must carry that licence text (the release and Docker CUDA builds do, via `licenses/`). Without the checkout the built-in NVFP4 kernel serves every
batch size (about 25 % slower for prompts). The activation and weight conversions around the GEMM, the device weight layout and the small-batch
kernel (`nvfp4.cu`) are fLLM's.

### MLX (Apple)

https://github.com/ml-explore/mlx, MIT, Copyright (c) 2023-2024 Apple Inc. (the header of the files says "Copyright (c) 2023-2024 Apple Inc."; `licenses/MLX-MIT.txt`).

| File | MLX source it follows | Nature |
| --- | --- | --- |
| `rust/engine/metal/attn_decode.metal` | `mlx/backend/metal/kernels/sdpa_vector.h` (`sdpa_vector`, `sdpa_vector_2pass_1`, `sdpa_vector_2pass_2`) | structure (one simdgroup per key, `simd_sum` for the score, online softmax per simdgroup, combination through threadgroup memory, two-pass split); written for fLLM, nothing copied verbatim; full MIT text in `norm_rms.metal` |
| `rust/engine/metal/norm_rms.metal` | `rms_norm.metal` | structure (one row per threadgroup, `simd_sum`, partials through threadgroup memory); carries the full MIT notice |
| `rust/engine/metal/rope_apply.metal` | `rope.metal` | one-thread-per-pair layout; MIT line in the header |

The paged KV cache, q8_0 cache, ALiBi, soft cap, sinks, window and chunk masks, GQA, arbitrary head widths and the `simdgroup_matrix` prompt attention are
fLLM's. The template-kernel pattern with `[[host_name]]` instantiation and the several-rows-per-simdgroup matrix-vector layout used in the other `.metal`
files are MLX-style structural models only; no MLX source is included.

### ExLlamaV3 (turboderp)

https://github.com/turboderp-org/exllamav3, MIT, Copyright (c) 2025 Turboderp (`licenses/ExLlamaV3-MIT.txt`, also reproduced in `docs/exl-formats.md`).
`rust/checkpoint/src/exl3.rs` (and the layout description in `docs/exl-formats.md`): the EXL3 codebook constants (`mcg` / `mul1` multipliers) and the trellis
layout are derived from ExLlamaV3 (layout facts / constants); the decoder is written in Rust for fLLM. The native kernels (`rust/engine/src/cpu_exl.rs`, `rust/engine/cuda/exl.cu`) and
the blob layout (`rust/gguf-probe/src/exl.rs`) are written from those facts; no ExLlamaV2 or ExLlamaV3 kernel source was copied or adapted.

### rocWMMA and ROCm (AMD)

rocWMMA (https://github.com/ROCm/rocWMMA, MIT, Copyright (c) Advanced Micro Devices, Inc., `licenses/rocWMMA-MIT.txt`) is used as a header library from the ROCm
install by the HIP branch of `rust/engine/cuda/tc_gemm.cu` (fallback; the primary matrix-core path is fLLM's own `hip/hip_mma.cuh`, `hip_gemm.cuh`, `hip_gemm.cu`,
whose lane layouts and builtin signatures were checked against rocWMMA's headers). Nothing is copied. `cuda/hip_compat.h`, `cuda/hip_stubs.cu`, `hip/hip_dpp.cuh`,
`hip/hip_device.cu` and the rest of `hip_q4z.cu` were written for fLLM. The AMD build compiles with `hipcc` (ROCm, MIT / Apache-2.0 with LLVM exceptions) and links the ROCm
runtime dynamically; ROCm is not distributed by fLLM.

### OpenXLA PJRT (TPU)

* `xla/pjrt/c/pjrt_c_api.h` (Apache-2.0, Copyright 2022 The OpenXLA Authors; `licenses/OpenXLA-Apache-2.0.txt`): the structure and table layout (API 0.116) and the enum values
  were transcribed to Rust in `rust/engine/src/backend/tpu/pjrt/sys.rs`, which keeps the licence header (layout facts / constants). No header is built or distributed.
* `compile_options.proto`, `xla_data.proto` (Apache-2.0): field numbers and message structure in `pjrt/proto.rs` (numbers only, hand-written serialization; named in the file header).
* StableHLO specification (Apache-2.0 with LLVM exception): read only; the generated programs follow its text format, no code was taken. `jax` / `jaxlib` is used only
  for the developer check `scripts/colab/stablehlo_verify.py` and is not distributed. `libtpu` is loaded at run time and not distributed.

### Intel NPU over OpenVINO

OpenVINO (Intel Corporation, Apache-2.0, https://github.com/openvinotoolkit/openvino; `licenses/OpenVINO-Apache-2.0.txt`): `rust/engine/src/backend/npu/openvino.rs` and `openvino/*.rs`
redeclare signatures of the public C API (`openvino/c/*.h`) and load the library at run time (`dlopen`, no link); the IR v11 file format and the NNCF weight-decompression
pattern (Const, Convert, Multiply, Reshape, MatMul) were rebuilt from documentation and tests (layout facts). No source text was copied and OpenVINO is not distributed with fLLM:
the user installs it (`pip install openvino` or the Intel archives) under its own licence. The pip package `openvino` was used as a parser check during development only.

### AMD NPU, fLLM's own kernels (XDNA)

* mlir-aie / IRON (AMD and contributors, Apache-2.0 with LLVM exceptions, https://github.com/Xilinx/mlir-aie, v1.4.4; `licenses/mlir-aie-Apache-2.0-LLVM-exception.txt`):
  the `.xclbin` and the instruction streams in `rust/engine/xdna/kernels/` are **compiled output** of its `whole_array.py` GEMM design and its `aie_kernels` (matrix
  multiplication and zeroing kernels), built with `scripts/xdna-build-kernels.sh`. The generated binaries are redistributed under the same licence; the sources are not
  copied into this repository. The build tools themselves (mlir-aie wheel, Peano / llvm-aie nightly 22.0.0.2026102001+4f5df13d, Apache-2.0 with LLVM exceptions) are
  downloaded by the script and not distributed.
* Adapted from mlir-aie (same licence, headers kept in the files): `rust/engine/xdna/kernel_src/fllm_mm_bfp.cc` (from `mm_bfp_mixed.cc`), the unpack in `fllm_dq.cc` (from `q4nx_dequant.cc`), `whole_array_dq.py`, `whole_array_dqbfp.py` (from `whole_array.py`).
* XRT (AMD / Xilinx, Apache-2.0, https://github.com/Xilinx/XRT, 2.21.x): the shim `rust/engine/xdna/shim/fllm_xdna_shim.cpp` is compiled against its public C++ headers
  by the user; XRT is loaded at run time and not distributed.
* FastFlowLM's `.xclbin` files are AMD's proprietary binaries with an undocumented ABI; the native backend does not use, copy or redistribute them.

### Other NPU probes

* apple/coremltools (BSD-3-Clause, Copyright Apple Inc.): field numbers and structures of `Model.proto`, `MIL.proto`, `FeatureTypes.proto`, the MIL blob storage format
  (`StorageFormat.hpp`) and the `.mlpackage` manifest were used as a specification; `rust/engine/src/backend/npu/ane/mlpackage.rs` is fLLM's own code.
* onnx/onnx (Apache-2.0): field numbers of `onnx.proto3` for the `ModelProto` emitter in `npu/xdna/onnx.rs`; own code.
* microsoft/onnxruntime (MIT): slot order of `struct OrtApi` and signatures of `onnxruntime_c_api.h` v1.22.0 in `npu/xdna/ort.rs`; own code, the library is loaded at run time.

### Components with terms that have NOT been checked

For these the repository contains no vendor code, and fLLM neither ships nor links them. Do not redistribute binaries together with them on the strength of this file.

| Component | What fLLM contains | What is NOT redistributed | What the user installs themselves |
| --- | --- | --- | --- |
| Qualcomm QAIRT SDK (QNN 2.27 / 2.37 headers, Genie 1.11; "Confidential and Proprietary - Qualcomm Technologies, Inc.") | the binary interface (member order, sizes, constants) restated in fLLM's own Rust types in `npu/hexagon/{qnn,graph,genie}.rs`; no header text | the SDK, any QNN / Genie / HTP library, any context binary generated with the SDK | the QAIRT SDK from Qualcomm under its download licence; fLLM loads `libQnn*.so` / `Genie` at run time. The licence terms of the SDK download have not been checked; this must happen before any binary that loads it is redistributed |
| AMD Ryzen AI Software (XDNA, Vitis AI EP) | PCI identifiers and provider option names from the public documentation (`npu/xdna/pci.rs`, `npu/xdna/ort.rs`) | Ryzen AI Software, the NPU driver, ONNX Runtime builds with the VitisAI provider, xclbin files | Ryzen AI Software and the NPU driver from AMD; fLLM loads onnxruntime at run time. Documentation only, no licence check |
| Intel oneAPI DPC++ / SYCL runtime | the SYCL backend source `rust/engine/sycl/` (SPDX Apache-2.0, own code) | the oneAPI compiler and runtime libraries; whether a Windows installer or Docker image may carry them has not been examined | oneAPI Base Toolkit (or the runtime packages) to build and run the SYCL backend. `docs/research/sycl-evaluation.md` cites public documents and measurement reports by link; no text was taken over |
| NVIDIA CUDA toolkit and driver | CUDA source files written for fLLM | the NVIDIA driver (`libcuda.so.1` is loaded from the host at run time); the toolkit | the driver; for source builds the CUDA toolkit. Release and Docker CUDA builds link the static CUDA runtime and use the `nvidia/cuda` base images, whose terms are NVIDIA's |
| Apple Metal SDK, Windows/DirectX | none | none | the OS provides them |

### Own implementations compared with other engines

The own Rust tensor, BERT and SigLIP/Idefics3 implementations use model configuration and GGUF format descriptions as references, rather than an external inference
engine. Model equations and configuration references:

- https://huggingface.co/BAAI/bge-small-en-v1.5
- https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct
- https://github.com/huggingface/transformers/tree/main/src/transformers/models/idefics3
- https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

The fLLM block allocator and prefix cache are independent implementations. The allocator checks capacity before committing an ensure_blocks request, matching vLLM
BlockPool.get_new_blocks all-or-nothing exhaustion behavior at the pinned revision. Regression tests verify the local table and free pool remain unchanged on exhaustion.
The broader policies were compared with pinned Apache-2.0 references: vLLM uses reference-counted blocks and a free-block pool; SGLang RadixCache performs longest
page-aligned prefix matching, splits shared paths, and evicts cache leaves. No source files from those repositories are copied into fLLM.

`cpu_kquant.rs` (eight-row panels), `cpu_float.rs` (sixteen-row float panels) and `cpu_delta.rs` (gated DeltaNet recurrence over token runs) are independent implementations.
The eight-row layout (the weights of eight rows interleaved so that one `vpdpbusd` against a broadcast four-value slice of an activation advances eight dot products in separate
lanes, scales applied once per sub-block in float) follows the technique of the repacked R4/R8 kernels of ik_llama.cpp (https://github.com/ikawrakow/ik_llama.cpp,
`ggml/src/iqk/iqk_gemm_kquants.cpp`, MIT licence), which was read as a reference (design reference). No source lines of ik_llama.cpp or llama.cpp are copied. The AVX-512 and AMX kernels in
`cpu_avx512/` use the same idea (weight rows in the vector lanes, repacked row panels), known from public descriptions of llama.cpp and ik_llama.cpp and from the existing eight-row
path; no source was copied. The constant values they need (`arch_prctl` request numbers for the AMX permission, CPU model numbers of `intel-family.h`) are facts taken from the Linux kernel
headers and the Intel Software Developer's Manual; no header text is included.

The kernels in `rust/engine/cuda/` not listed above were written for fLLM. They implement the GGUF/ggml block layouts and standard techniques such as online softmax
(FlashAttention, BSD-3-Clause; technique only). Whenever code is adopted from a project, it keeps its copyright notice in the file header and is listed here with its licence.
The native backends (`rust/engine/src/backend/{vulkan,metal,hip,sycl,npu,tpu}`, `rust/engine/shaders/vulkan`, `rust/engine/metal`, `rust/engine/sycl`, `rust/engine/hip`,
`rust/engine/src/cpu_arm`, `rust/engine/src/cpu_avx512`) were written for fLLM; the DeltaNet kernels are ports of fLLM's own CUDA kernels and CPU reference. The GPU shaders were compiled
with shaderc/glslang and checked with SPIRV-Tools, the DPC++ code with the Intel LLVM compiler, the HIP code with the ROCm LLVM compiler; these are build and check tools and are not
distributed. No llama.cpp or ggml backend source was copied for these backends.

## Rust crates

Tensor transport to hardware uses wgpu (MIT/Apache-2.0) and independently authored WGSL kernels. Image decoding uses image; numeric half conversion uses half; CPU parallelism uses rayon.
Native structured output uses the pinned crates `llguidance=1.8.0` and `jsonschema=0.29.1`; tokenization uses `tokenizers=0.23.2` (exact pins in `rust/engine/Cargo.toml`); `Cargo.lock` records
all transitive dependencies and checksums. These run grammar masks and argument validation in fLLM's sampler, without using libllama. `cargo deny check licenses` enforces the allow-list
in `deny.toml`; the result of the last run is in `docs/changes/hygiene-a.md`. Static release binaries contain these crates; their licence notices are part of the crates' sources
(`cargo about` or `cargo deny list` produce a list). Test-only model downloads have their own model licences and are excluded from distributions.

## Build, release and CI tooling

The workflows in `.github/workflows/` reference these third-party actions, pinned by commit SHA; none is embedded in a release asset, and their licences (MIT or Apache-2.0 for the
GitHub-owned and Docker-owned actions; check each repository for the others) apply to their own use only: `actions/checkout`, `actions/upload-artifact`, `actions/download-artifact`,
`dtolnay/rust-toolchain`, `Swatinem/rust-cache`, `jakoch/install-vulkan-sdk-action`, `Jimver/cuda-toolkit`, `docker/build-push-action`, `docker/login-action`, `docker/metadata-action`,
`docker/setup-buildx-action`, and (`supply-chain.yml`) `EmbarkStudios/cargo-deny-action` and `rustsec/audit-check` or the cargo-deny / cargo-audit binaries. The licences of the individual actions are not part of the release assets. Release binaries are unsigned (see `docs/releasing.md`).
