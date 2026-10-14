# Third-party CPU kernels

The own Rust tensor, BERT and SigLIP/Idefics3 implementations use model
configuration and GGUF format descriptions as references, rather than an
external inference engine. Tensor transport to hardware uses wgpu (MIT/Apache-2.0)
and independently authored WGSL kernels. Image decoding uses image; numeric
half conversion uses half; CPU parallelism uses rayon. Locked dependency
versions and licenses remain in Cargo manifests/locks. Model equations and
configuration references:
- https://huggingface.co/BAAI/bge-small-en-v1.5
- https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct
- https://github.com/huggingface/transformers/tree/main/src/transformers/models/idefics3
- https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

Native structured output uses pinned Rust crates `llguidance=1.8.0` and
`jsonschema=0.29.1`; the Cargo lock records their transitive dependencies and
checksums. These run grammar masks and argument validation in fLLM's sampler,
without using libllama. Consult the locked crates' license notices when
redistributing. Test-only model downloads have their own model licenses and are
excluded from distributions.

The fLLM block allocator and prefix cache are independent implementations. The
allocator checks capacity before committing an ensure_blocks request, matching
vLLM BlockPool.get_new_blocks all-or-nothing exhaustion behavior at the pinned
revision. Regression tests verify the local table and free pool remain unchanged
on exhaustion. The broader policies were compared with pinned Apache-2.0
references: vLLM uses reference-counted blocks and a free-block pool; SGLang
RadixCache performs longest page-aligned prefix matching, splits shared paths,
and evicts cache leaves. No source files from those repositories are copied into
fLLM.

## CUDA kernels

The kernels in rust/engine/cuda/ were written for fLLM. They implement the GGUF/ggml block layouts
(documented in the ggml repository, MIT) and standard techniques such as online softmax
(FlashAttention, BSD-3-Clause). Apart from `nvfp4_cutlass.cu` (below) no source files from llama.cpp, vLLM, FlashInfer, CUTLASS or other
engines are copied. Whenever code is adopted from such a project, it keeps its
copyright notice in the file header and is listed here with its license.

### AMD (HIP) build

`cuda/hip_compat.h`, `cuda/hip_stubs.cu`, the matrix-core GEMM, the DPP reductions and the device code in `rust/engine/hip/` were written for fLLM;
the AMD build compiles the fLLM kernels with the ROCm toolchain (`hipcc`, MIT/Apache-2.0 with LLVM exceptions) and links the ROCm runtime
dynamically. The prompt GEMM for the matrix cores of RDNA3/4 and CDNA (`tc_gemm.cu`, HIP branch) uses rocWMMA
(https://github.com/ROCm/rocWMMA, MIT, Copyright (c) Advanced Micro Devices, Inc.) from the ROCm install as a header library; it is not copied.

#### AMD (HIP) build: vLLM ROCm kernels

`rust/engine/hip/hip_skinny.cu` is adapted from vLLM `csrc/rocm/skinny_gemms.cu` (`wvSplitK_hf_`; https://github.com/vllm-project/vllm, commit
7fad7db3309347a9d6288a09ed0993728e474a34, Apache License 2.0, Copyright contributors to the vLLM project, license text in `licenses/vLLM-Apache-2.0.txt`).
The kernel `q4z_int4_sml_kernel` in `rust/engine/hip/hip_q4z.cu` is adapted from vLLM `csrc/rocm/skinny_gemms_int4.cu` (`wvSplitK_int4_hf_sml_`, same commit and license).
Both files keep the license header and state their changes (PyTorch glue replaced by C++ entry points, FP32 activations and results, fLLM weight layouts).
`hip/hip_mma.cuh`, `hip/hip_gemm.cuh`, `hip/hip_gemm.cu`, `hip/hip_dpp.cuh`, `hip/hip_device.cu` and the rest of `hip_q4z.cu` were written for fLLM; the lane layouts
and builtin signatures of the matrix instructions were checked against rocWMMA (https://github.com/ROCm/rocWMMA, MIT, Copyright (c) Advanced Micro Devices, Inc.),
whose headers come with the ROCm install and are not copied. The older `tc_gemm.cu` HIP branch still uses rocWMMA as a fallback.
vLLM `csrc/rocm/attention.cu`, `q_gemm_rdna3*.cu` and `moe_q_gemm_rdna3.cu` are not used.

### Q4_K Marlin Tensor-Core GEMM (`rust/engine/cuda/marlin_q4k.cu`)

On Ampere and newer GPUs (sm_80, sm_86, sm_89, sm_90, sm_120) Q4_K matrix multiplications run through the Marlin
kernel design (https://github.com/IST-DASLab/marlin, Apache License 2.0, Copyright (C) Marlin.2024 Elias Frantar,
license text in `licenses/MARLIN-Apache-2.0.txt`). The kernel adapts `marlin_cuda_kernel.cu` for GGUF Q4_K grouped
affine quantization:

- Weights are repacked once at upload time (`q4_k_marlin_device_layout` in `backend/cuda.rs`) into Marlin's 16x64 tile
  layout directly ordered for `mma.m16n8k16` tensor core fragments.
- Sub-block scales and minimums (group size 32) are extracted and stored in permuted 8x8 layout; dequantization runs
  directly in registers via `lop3` into unsigned fp16 in [0..15], fused with scale and min via a single `hfma2` instruction.
- Matrix outputs accumulate in FP32 with optional fused bias addition.
- `FLLM_CUDA_NO_MARLIN=1` falls back to the in-place Q4_K GEMM (`qgemm4k.cu`).

### Prompt attention (`rust/engine/cuda/flashinfer_prefill.cu`)

On Turing and newer GPUs the attention of prompts with an f16 KV cache runs through FlashInfer's single-request prefill
kernel (https://github.com/flashinfer-ai/flashinfer, Apache License 2.0, Copyright (c) 2023 by FlashInfer team). The file
only prepares its inputs (gather of the paged cache, FP32 <-> FP16) following FlashInfer's `csrc/single_prefill.cu`.
FlashInfer is header-only and not part of this repository: `build.rs` uses a checkout in `third_party/flashinfer` or
`FLLM_FLASHINFER_DIR` (`git clone --depth 1 --branch v0.6.5 https://github.com/flashinfer-ai/flashinfer third_party/flashinfer`).
Without it the built-in tensor-core attention (`attention3.cu`) is used. `FLLM_CUDA_NO_FLASHINFER=1` switches it off.

### NVFP4 prompt GEMM (`rust/engine/cuda/nvfp4_cutlass.cu`)

Large batches of NVFP4 weights (prompt processing on Blackwell GeForce, sm_120) run through the CUTLASS block-scaled GEMM,
configured as vLLM and SGLang (through FlashInfer) do it. The file adapts

- the kernel configuration of vLLM `csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu`
  (Copyright (c) 2025, NVIDIA CORPORATION, Apache License 2.0, https://github.com/vllm-project/vllm), and
- the collective builders of CUTLASS example `79a_blackwell_geforce_nvfp4_bf16_gemm.cu`
  (Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES, BSD-3-Clause, https://github.com/NVIDIA/cutlass), and
- the epilogue and tile list of FlashInfer `include/flashinfer/gemm/fp4_gemm_template_sm120.h` and `fp4_gemm_cutlass_template_sm120.h`
  (no source tensor C when the product is only stored, an explicit epilogue schedule, the 128x128x256 tile; Copyright (c) 2020-2023, NVIDIA CORPORATION,
  Apache License 2.0, https://github.com/flashinfer-ai/flashinfer).

The changes are listed in the header of the file. CUTLASS itself (header-only templates, BSD-3-Clause, text in
`licenses/CUTLASS-BSD-3-Clause.txt`) is not part of this repository: `build.rs` uses a checkout in `third_party/cutlass`
or `FLLM_CUTLASS_DIR` (`git clone --depth 1 --branch v4.8.0 https://github.com/NVIDIA/cutlass third_party/cutlass`);
binaries built with it contain CUTLASS templates and must carry that license text. Without the checkout the built-in
NVFP4 kernel serves every batch size (about 25 % slower for prompts). The activation and weight conversions around the
GEMM, the device weight layout and the small-batch kernel (`nvfp4.cu`) were written for fLLM.

### IQ and K-quant decoders on the GPU (`rust/engine/cuda/iq_decode.cuh`, `iq_tables.cuh`, `mmvq_x.cuh`)

Q2_K, Q3_K and IQ1_S / IQ1_M / IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S run natively on the GPU. The block layouts, the
dequantization equations and the integer codebooks are those of the GGML formats (ggml-quants.c / ggml-common.h of llama.cpp commit
b1ff4ca23630ac7c5a275405353ddb5c486332c9; Copyright (c) 2023-2026 The ggml authors, MIT license), which fLLM already carried in
Rust (`quant_iq.rs`, `quant_extra.rs`, `quant_iq_tables.rs`); `iq_tables.cuh` is generated from the last file and the decoders
port the Rust reference line by line. The GPU kernels (generic GEMV, tensor-core GEMM through `tc_gemm.cu`, integer dot product in
`mmvq_x.cuh`) were written for fLLM; `selftest-gpu` compares every format with the Rust reference decoders.

### Marlin int4 GEMM for AWQ / GPTQ weights (`rust/engine/cuda/marlin_vllm/`)

The tensor-core GEMM for 4-bit and 8-bit weights with group scales (AWQ and GPTQ, FP16 activations), FP8 weights (compressed-tensors / neuralmagic FP8 e4m3fn, W8A16) and the weight repack kernels are
the Marlin kernels of vLLM, commit 9367d8b9693c3d2fb6c3f99d8ccb4aa7673c0981 of https://github.com/vllm-project/vllm (directory
`csrc/libtorch_stable/quantization/marlin`, Apache License 2.0, text in `licenses/vLLM-Apache-2.0.txt`; "Modified by Neural Magic,
Copyright (C) Marlin.2024 Elias Frantar", derived from https://github.com/IST-DASLab/marlin, also Apache-2.0). Taken unchanged:
`marlin_template.h`, `dequant.h`, `marlin_mma.h`, `marlin_dtypes.cuh`, `marlin.cuh`, `kernel.h`, `core/scalar_type.hpp` (except the include
of the PyTorch exception header, replaced by `torch_shim.h`) and `generate_kernels.py` (restricted to FP16 activations and the int4 / int8 / fp8 (kFE4M3fn) /
fp4 weight types, and without the FP8-activation clause); the `sm75_kernel_*.cu` / `sm80_kernel_*.cu` files and `kernel_selector.h` are its
output. `marlin_vllm.cu` and `marlin_vllm_{awq,gptq}_repack.cu` are the host and repack code of `marlin.cu`, `awq_marlin_repack.cu` and
`gptq_marlin_repack.cu` with the PyTorch tensor glue replaced by plain C entry points. The scale and zero-point permutations of
vLLM's `marlin_utils.py` are reimplemented for fLLM. `bench/test_vllm_marlin.cu` checks the port against an fp32 reference.
The `Q4Z` tensor format that carries AWQ / GPTQ weights to this kernel, the converter, the weight upload and the activation glue
(`marlin_vllm_glue.cu`) were written for fLLM (docs/awq-gptq-marlin.md).

### Fused decode kernels (`rust/engine/cuda/fused.cu`)

The fusions of residual add + RMSNorm, rotary embedding + KV-cache write and gated activation + quantization follow the design of vLLM
(`csrc/layernorm_kernels.cu` `fused_add_rms_norm`, `csrc/pos_encoding_kernels.cu` + `csrc/cache_kernels.cu` `reshape_and_cache`,
`csrc/activation_kernels.cu` `silu_and_mul`; Apache License 2.0), SGLang and FlashInfer (`apply_rope_pos_ids`; Apache License 2.0). No
source code of these projects was copied: the kernels were written for fLLM on top of its own `rms_norm_vec_kernel`, `residual_kernel`,
`rope_kernel`, `quantize_x8_kernel` and `nvfp4_quantize_kernel` and repeat their arithmetic, so that the results are bit-identical.

## CPU prompt kernels (eight-row panels, DeltaNet, sixteen-row float panels)

`cpu_kquant.rs` (eight-row panels), `cpu_float.rs` (sixteen-row float panels) and `cpu_delta.rs`
(gated DeltaNet recurrence over token runs) are independent implementations. The eight-row layout
(the weights of eight rows interleaved so that one `vpdpbusd` against a broadcast four-value slice of
an activation advances eight dot products in separate lanes, scales applied once per sub-block in
float) follows the technique of the repacked R4/R8 kernels of ik_llama.cpp
(https://github.com/ikawrakow/ik_llama.cpp, `ggml/src/iqk/iqk_gemm_kquants.cpp`, MIT license),
which was read as a reference. No source lines of ik_llama.cpp or llama.cpp are copied.
The AVX-512 and AMX kernels in `cpu_avx512/` use the same idea (weight rows in the vector lanes, repacked row panels), known from public descriptions of llama.cpp and
ik_llama.cpp and from the existing eight-row path; no source was copied. The constant values they need (`arch_prctl` request numbers for the AMX permission,
CPU model numbers of `intel-family.h`) are facts taken from the Linux kernel headers and the Intel Software Developer's Manual; no header text is included.

## Native backends (Vulkan, Metal, SYCL, NPU, TPU, ARM, x86)

The native backends added in the multi-backend round (`rust/engine/src/backend/{vulkan,metal,hip,sycl,npu,tpu}`, `rust/engine/shaders/vulkan`,
`rust/engine/metal`, `rust/engine/sycl`, `rust/engine/hip`, `rust/engine/src/cpu_arm`, `rust/engine/src/cpu_avx512`) were written for fLLM. Their quantization block
layouts and dequantization equations are the GGML ones that fLLM already carries in Rust (`compute.rs`, `quant_extra.rs`, `quant_iq.rs`; GGML authors, MIT license,
see above), and `common.metal` carries the same notice. The DeltaNet kernels are ports of fLLM's own CUDA kernels and CPU reference. The GPU shaders were compiled with
shaderc/glslang and checked with SPIRV-Tools, the DPC++ code with the Intel LLVM compiler, the HIP code with the ROCm LLVM compiler; these are build and check tools and are
not distributed. No llama.cpp or ggml backend source was copied for these backends. Where designs, specifications or interfaces of other projects were
used, they are listed here.

### Metal kernels: MLX

The structure of `rust/engine/metal/attn_decode.metal`, `norm_rms.metal` and `rope_apply.metal` follows kernels of MLX (https://github.com/ml-explore/mlx, MIT license,
Copyright (c) 2023-2024 Apple Inc.): `mlx/backend/metal/kernels/sdpa_vector.h` (`sdpa_vector`, `sdpa_vector_2pass_1`, `sdpa_vector_2pass_2`: one simdgroup per key, `simd_sum` for the
score, online softmax per simdgroup, combination through threadgroup memory, two-pass split), `rms_norm.metal` and `rope.metal`. The code was written for fLLM (paged KV cache,
q8_0 cache, ALiBi, soft cap, sinks, window and chunk masks, GQA, arbitrary head widths and the `simdgroup_matrix` prompt attention are fLLM's own additions); nothing was copied
verbatim. The MIT notice with the copyright line is the header of the three files. The template-kernel pattern with `[[host_name]]` instantiation and the
several-rows-per-simdgroup matrix-vector layout used in the other `.metal` files are MLX-style structural models only; no MLX source is included.

### Intel NPU over OpenVINO

OpenVINO (Intel Corporation, Apache License 2.0, https://github.com/openvinotoolkit/openvino): `rust/engine/src/backend/npu/openvino.rs` and `openvino/*.rs` redeclare signatures of the
public C API (`openvino/c/*.h`) and load the library at run time (`dlopen`, no link); the IR v11 file format and the NNCF weight-decompression pattern
(Const, Convert, Multiply, Reshape, MatMul) were rebuilt from documentation and tests. No source text was copied and OpenVINO is not distributed with fLLM. The pip package
`openvino` was used as a parser check during development only.

### AMD XDNA, Qualcomm Hexagon, Apple Neural Engine (probes)

* apple/coremltools (BSD-3-Clause, Copyright Apple Inc.): field numbers and structures of `Model.proto`, `MIL.proto`, `FeatureTypes.proto`, the MIL blob storage format
  (`StorageFormat.hpp`) and the `.mlpackage` manifest were used as a specification; `rust/engine/src/backend/npu/ane/mlpackage.rs` is fLLM's own code.
* onnx/onnx (Apache-2.0): field numbers of `onnx.proto3` for the `ModelProto` emitter in `npu/xdna/onnx.rs`; own code.
* microsoft/onnxruntime (MIT): slot order of `struct OrtApi` and signatures of `onnxruntime_c_api.h` v1.22.0 in `npu/xdna/ort.rs`; own code, the library is loaded at run time.
* Qualcomm QAIRT SDK (QNN, Genie; proprietary, "Confidential and Proprietary - Qualcomm Technologies, Inc."): only the binary interface (field order and sizes, constant values) of
  QNN 2.27 and Genie 1.11 is reproduced in fLLM's own Rust types in `npu/hexagon/{qnn,graph,genie}.rs`. No header text is copied, the SDK is neither in the repository nor distributed
  and is loaded at run time. The license terms of the SDK download have not been checked; this must happen before binaries that load it are redistributed.
* AMD Ryzen AI Software documentation: PCI identifiers and provider options (`npu/xdna/pci.rs`, `npu/xdna/ort.rs`); documentation only, no license check.

### Google TPU over PJRT

* OpenXLA PJRT C API (`xla/pjrt/c/pjrt_c_api.h`, Apache License 2.0, Copyright The OpenXLA Authors): structure and table layout (API 0.116) and enum values were transcribed to Rust in
  `rust/engine/src/backend/tpu/pjrt/sys.rs`, which keeps the license header. No header is built or distributed.
* OpenXLA `compile_options.proto` and `xla_data.proto` (Apache-2.0): field numbers and message structure of the compile options in `pjrt/proto.rs` (numbers only, hand-written serialization; the source is named in the file header).
* StableHLO specification (OpenXLA, Apache-2.0 with LLVM exception): read only; the generated programs follow its text format, no code was taken. `jax`/`jaxlib` is used only for the developer check
  `scripts/colab/stablehlo_verify.py` and is not distributed. `libtpu` is loaded at run time and not distributed.

### Intel SYCL

The SYCL backend (`rust/engine/sycl/`) is compiled with Intel oneAPI DPC++ (open-source line: Apache License 2.0 with LLVM exceptions) and links the oneAPI SYCL runtime. Neither is in the
repository. Redistribution of the oneAPI runtime libraries (for instance in a Windows installer) has not been examined. The comparison in `rust/engine/sycl/EVALUATION.md` cites public
documents and measurement reports by link; no text was taken over.

### Continuous integration

The workflows in `.github/workflows/` and `.github/workflows/` reference these third-party actions by name; none is embedded, none has run, and their licenses have not been checked in
this round: `actions/checkout`, `actions/upload-artifact`, `dtolnay/rust-toolchain`, `Swatinem/rust-cache`, `jakoch/install-vulkan-sdk-action`, `Jimver/cuda-toolkit`, `softprops/action-gh-release`.
