# Expected skips of `fllm-rs selftest-gpu --device vulkan`

A check is skipped ("SKIPPED (unsupported)") when the backend answers `UnsupportedOperation` / `UnsupportedQuant` for something it does not list
in its capabilities. A skip is not a failure, but it is a gap: this file is the committed list of the gaps, so that a new gap (a regression) or
a closed one (a feature that now works) shows up as a difference. `fllm-rs selftest-gpu --device vulkan --expect-skips <this file>` compares the
skipped checks of a run, per group and op, with the block below and fails on any difference in either direction. Remove a line when its
feature is implemented; never add a line to make a run pass without a reason in the comment.

Recorded on Mesa lavapipe (llvmpipe, LLVM 20, Vulkan 1.4) by `scripts/vulkan-soft-test.sh selftest`: both passes (exact FP32 shaders, then the
fast variants). The set does not depend on the GPU model: every skip is an operation or format with no Vulkan implementation. A real device
adds nothing to this list; if it prints other skips, the capability reporting of the backend differs between the devices and that is a bug.

What the lines mean:

- `fuse | ...`: fused decode kernels of the CUDA backend (add + RMSNorm, rotary + cache write, SwiGLU + int8 quantization, Marlin add): the Vulkan
  stream runs the unfused sequence instead.
- `linear | nvfp4 w4a4`: activation-quantizing NVFP4 GEMM on FP4 tensor cores (Blackwell); Vulkan stores NVFP4 weights in its own layout.
- Not skipped but also missing: `moe_expert_gemm` (not exercised by a
  selftest check: the device path is `moe_linear` + `moe_combine`).

```expected-skips
# group | op | skipped checks (both passes together)
fuse | add_rms_norm | 5    # add_rms_norm
fuse | add_rms_norm +nvfp4 | 4    # add_rms_norm
fuse | add_rms_norm +q4_k | 2    # add_rms_norm
fuse | add_rms_norm +q4z32 | 2    # add_rms_norm
fuse | add_rms_norm +q5_k | 2    # add_rms_norm
fuse | add_rms_norm +q6_k | 2    # add_rms_norm
fuse | geglu + int8 quantization q4_k | 1    # swiglu + int8 quantization
fuse | geglu + int8 quantization q5_k | 1    # swiglu + int8 quantization
fuse | geglu + int8 quantization q6_k | 1    # swiglu + int8 quantization
fuse | marlin add_rms_norm | 2    # quantized_linear_add
fuse | rope_kv_write | 4    # rope_kv_write
fuse | swiglu + int8 quantization q4_k | 2    # swiglu + int8 quantization
fuse | swiglu + int8 quantization q5_k | 2    # swiglu + int8 quantization
fuse | swiglu + int8 quantization q6_k | 2    # swiglu + int8 quantization
linear | nvfp4 w4a4 | 20    # NVFP4 W4A4 needs the CUDA tensor-core layout
```
