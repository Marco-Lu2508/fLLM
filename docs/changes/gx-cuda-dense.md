# gx-cuda-dense: CUDA Dense Path Completion (Fragment)

This change completes the remaining items for the CUDA dense path across the Sm100 (Blackwell), Sm90 (Hopper), and general Tensor Core pipelines, building upon the merged base of cuda-act and hopper.

All CUDA kernels are delivered at **test level L0** (compiled and structurally verified against CPU reference or emulator; full hardware validation will be executed on Colab).

## 1. Sm100 NVFP4 Enhancements
- **Direct FP16 Activation Quantization**: Extended 
vfp4.cu and 
vfp4_cutlass_sm100.cu with HALF_IN templates to quantize direct FP16 device tensors to NVFP4 without requiring intermediate FP32 copies.
- **SwiGLU Fusion**: Integrated SwiGLU activation directly into the NVFP4/Int8 quantization pass for down-projections.
- **CUTLASS Sm100 Epilogues**:
  - Accumulation epilogue (Acc=true, ElementC=float, eta=fllm_fp4_acc_scale).
  - Direct FP16 output epilogue (out_half=true, ElementD=cutlass::half_t).
- **Autotuner**: Added dynamic runtime autotuning (FLLM_NVFP4_SM100_TUNE=1 / FLLM_NVFP4_TUNE=1) that benchmarks tile configurations and caches best performance in g_tune_cache.
- **Grouped / MoE Sm100 NVFP4 GEMM**: Added llm_cuda_nvfp4_sm100_moe_gemm for grouped stacked NVFP4 experts.

## 2. Hopper Blockwise FP8 (CUTLASS Example 67)
- **128x128 Blockwise Scaling**: Ported CUTLASS 3.x Example 67 blockwise GEMM (llm_fp8_cutlass_blockwise_gemm) in cuda/fp8_cutlass_sm90.cu and cuda/fp8_gemm.cu for native Hopper execution (wblock == 1).
- **Native FP8 Weights**: Retained blockwise FP8 scale inversions (weight_scale_inv) in the weight loader and converter without unrolling to FP16.

## 3. FlashAttention-3 (FA3) Prefill Extensions
- **Soft-Capping**: Integrated soft-capping support (lashinfer::LogitsSoftCap) into a3_prefill.cu when softcap > 0.0f.
- **DType Support**: Added dispatch paths for BF16 and FP8 prefill inputs alongside FP16.

## 4. FlashInfer Decode & BF16 KV Cache
- **FlashInfer Core Decode**: Wired lashinfer::BatchDecodeWithPagedKVCacheDispatched in cuda/fi_attn.cu for single/multi-batch decode.
- **BF16 KV Cache**: Added templated support for __nv_bfloat16 KV cache pages (kind == 3) in addition to FP16.

## 5. Grouped Layer Projections & Architecture Support
- **7-Projection Grouped GEMM**: Implemented llm_cuda_tc_grouped_gemm in cuda/tc_gemm.cu allowing batched execution of Q, K, V, Gate, Up, Down, and LM head projections.
- **MTP Latent Cache**: Enabled MTP on CUDA with latent cache retention in ust_decoder.rs.
- **CudaStream Compute Capability**: Added cc: u32 tracking to CudaStream for immediate device architecture queries.

