# The AMD XDNA NPU with fLLM's own kernels

fLLM can compute on the NPU of an AMD Ryzen AI processor (XDNA 2, "npu2" / AIE2P) with kernels of its own, loaded and run through XRT. No
vendor inference library is involved: not FastFlowLM, not ONNX Runtime, not AMD's proprietary `.xclbin` files. The route through FastFlowLM
([user/amd-npu.md](user/amd-npu.md)) stays as it was and works without any of this.

**Status (2026-10-30).** Built and measured on one machine: a Ryzen AI 5 PRO 435 laptop (Radeon 840M, 14 GB LPDDR, Ubuntu 26.04, kernel 7.0,
`amdxdna` driver, NPU firmware 1.1.2.64, XRT 2.21.75 from the distribution). Everything else is untested: Windows, other XDNA 2 chips (Strix
Point, Strix Halo), XDNA 1 (Phoenix / Hawk Point: a different architecture, `npu1`, not built), macOS and Linux on aarch64 compile only.

## What it does and what it does not

* **Prompt processing: yes.** The large projections `y = x . W^T` of a prompt chunk (64 or more rows) run on the NPU; everything else runs on
  the processor. The weights stay quantized in memory (GGUF blocks, as loaded) and are expanded to bf16 when a tile is packed.
* **Decode (one row at a time): no.** It is memory bound, the NPU is slower than the processor for it (below), so the rows stay on the CPU.
* **Precision.** Activations and weights are rounded to bf16, products accumulate in f32 on the NPU, the result is f32. Against an f64
  reference the error of a projection is about 4e-4 of the error scale (the self test allows 4e-3); a 714-token prompt through Llama-3.2-3B
  gives the same 60 greedy tokens on the NPU and on the processor.
* **Formats.** Q4_0, Q8_0, Q4_K, Q6_K, F16, BF16, F32 are decoded straight to bf16 (fast loops, AVX2 chosen at run time); every other
  format the engine can dequantize goes through the generic dequantizer (same values, slower): Q4_1, Q5_x, Q2_K, Q3_K, IQ*, MXFP4, NVFP4 ...
  Formats without a dequantizer (Q4Z, bitsandbytes, EXL2/3 ...) are computed on the processor.

## Parts

| Part | Where | What |
|---|---|---|
| NPU programs | `rust/engine/xdna/kernels/` | one `.xclbin` (bf16 x bf16 -> f32 GEMM on the 8 x 4 compute tiles, `B` stored `[n][k]`, tile 32 x 64 x 64) and eight instruction streams `m{128,512}_k{1024,2048,3072,4096}_n1024.insts`; embedded in the binary |
| Shim | `rust/engine/xdna/shim/fllm_xdna_shim.cpp` | 12 C functions over the C++ API of XRT (device, hardware context, kernel, buffers); built by `fllm-rs amd-npu build-shim` (or `scripts/build-xdna-shim.sh`) into `libfllm_xdna_shim.so` |
| Binding | `backend/npu/xdna/native/shim.rs` | opens the shim at run time (`libloading`); finds it via `FLLM_XDNA_SHIM`, the folder of the executable, `~/.cache/fllm/npu` |
| Engine | `backend/npu/xdna/native/engine.rs`, `dequant.rs`, `kernels.rs` | row/depth tiling, bf16 packing, resident or streamed weights, double-buffered runs |
| Backends | `backend/npu/backend/hybrid.rs`, `backend/npu/host.rs` | `--device npu` (hybrid with the reference CPU backend, shared with the Intel route) and `FLLM_NPU_HOST=1` (the fast host decoder offloads prompt projections) |

Why a C++ shim: the `amdxdna` driver only accepts hardware contexts, and XRT exposes those in its C++ API only. The C functions of XRT
(`xrtDeviceLoadXclbinFile` and friends) fail on this driver with `load_axlf: Operation not supported`; this was tried first. The shim is the
only C++ in fLLM. It is **optional at build time** (nothing is compiled or linked by `cargo build`; the feature `npu-xdna-native` is in the
default set and only loads a library at run time) **and at run time**: with no shim, no XRT or no device, `fllm-rs devices` lists no XDNA device,
`--device npu` says why, and nothing else changes. Windows, macOS and aarch64 builds compile (the Windows cross-build was run).

## Using it

1. Install XRT with the NPU plugin (Ubuntu 26.04: `sudo apt install libxrt2 libxrt-npu2 libxrt-dev uuid-dev`, the kernel's `amdxdna` driver is
   there already) and let your user open `/dev/accel/accel0`. The memlock limit must be `unlimited` (`ulimit -l`), as for FastFlowLM.
2. Build the shim once: `fllm-rs amd-npu build-shim` (the source is inside the binary; needs a C++17 compiler and the XRT headers; `XRT_INCLUDE` / `XRT_LIB` name them when they
   are in an unusual place; `scripts/build-xdna-shim.sh` does the same from a checkout). It writes `~/.cache/fllm/npu/libfllm_xdna_shim.so`.
3. `fllm-rs devices` now lists `AMD XDNA NPU` with `--device npu:0`; `fllm-rs amd-npu status` shows `"native": {"available": true ...}` and the
   Studio's Hardware card says "Native fLLM kernels: ready".
4. Run prompts through the NPU:
   * **Recommended: the host decoder.** `FLLM_NPU_HOST=1 fllm-rs serve --model ... --device cpu`. The decoder keeps its memory-mapped weights, the
     NPU only receives prompt projections of 64 or more rows. Add `FLLM_XDNA_STATS=200` to print a time summary every 200 offloaded projections.
   * `--device npu:0`: the hybrid backend. It uploads every weight into the reference CPU backend and keeps a second copy of the quantized tensor
     for the NPU (as the Intel route does), so a 3B model needs about 6 GB before the first token; use it for the self test and for small models.

| Variable | Meaning |
|---|---|
| `FLLM_XDNA=0` | switch the native backend off |
| `FLLM_XDNA_SHIM` | the shim library file |
| `FLLM_XDNA_MIN_ROWS` | fewest prompt rows worth an NPU call (default 64); the rest of a prompt chunk goes to the CPU |
| `FLLM_XDNA_MIN_ELEMS`, `FLLM_XDNA_MIN_EFF` | smallest weight offloaded (65536 elements) and smallest useful share after padding to the tile sizes (0.5) |
| `FLLM_XDNA_RESIDENT_MB` | bf16 copies of weights kept for the NPU (default: 25 % of the free memory at start). Weights over the budget are packed again for every call ("streamed") |
| `FLLM_XDNA_STATS` | print a summary line every N offloaded projections |
| `FLLM_NPU_HOST=1` | the host decoder offloads (picks this engine when it is usable, else the OpenVINO one) |

The NPU shares the LPDDR with the processor and the iGPU, so resident bf16 weights (2 bytes per weight, four times a Q4 model) compete with the
model, the KV cache and the GPU for the same memory. The device inventory therefore reports the free and total memory of the computer, and the
resident budget is a quarter of what is free when the engine starts. A 3B Q4_K_M model needs 6.4 GB of bf16 to be fully resident.

## Measured

All numbers: Ryzen AI 5 PRO 435 laptop, Ubuntu 26.04, fLLM built from this tree with default features, 6 CPU threads (the default). Other agents
ran jobs on the same laptop at other times; the figures below are from runs during which no other benchmark job was seen in `ps` (the micro
benchmarks and the "Whole model" rows quoted as ranges), unless marked.

**Stage 1: can a kernel of our own run on the NPU?** Yes. mlir-aie 1.4.4 (Apache-2.0 with LLVM exceptions) and the Peano compiler (llvm-aie
nightly 22.0.0.2026102001+4f5df13d) were installed into a prefix of their own from the published wheels, without root; XRT's `xclbinutil`, the
headers and the Python bindings were unpacked from the distribution's `.deb` files into the same prefix. `aiecc` produced an `.xclbin` and an
instruction stream; XRT ran them. Single-core 256 x 256 x 256 bf16 GEMM: correct (largest error 7.6e-6 against 21 in the result), 28-41 GFLOPS.
Whole array (8 columns), 512 x 1024 x 1024 bf16 -> f32: 0.52 ms, **1.98 TFLOPS** in a Python harness; 512 x 4096 x 1024: up to 2.35 TFLOPS.

**Stage 2: a projection of a quantized weight, by the Rust engine** (`fllm-rs amd-npu native-bench ROWS COLS BATCH FORMAT`, Q4_K weights, bf16 copy
resident, best of 8; the CPU column is the engine's own AVX-512 kernels on 6 threads):

| projection | rows of `x` | NPU ms | CPU ms | NPU / CPU |
|---|---|---|---|---|
| 8192 x 3072 | 1 | 2.95 | 0.86 | 0.3x |
| | 16 | 2.95 | 2.97 | 1.0x |
| | 64 | 2.98 | 5.86 | 2.0x |
| | 128 | 3.05 | 8.15 | 2.7x |
| | 512 | 11.70 | 26.23 | 2.2x (2.2 TFLOPS) |
| 3072 x 8192 | 1 | 2.97 | 0.75 | 0.3x |
| | 64 | 3.03 | 5.19 | 1.7x |
| | 512 | 11.88 | 25.20 | 2.1x |

A call up to 128 rows costs the same 3 ms (the program computes 128 rows), so the NPU loses a single-row matrix-vector product by 3.4x and
wins from about 16 rows. Weights streamed instead of resident (budget 0): 1.26 TFLOPS at 512 rows (the packing is hidden behind the NPU,
but not completely).

**Self test** (`FLLM_XDNA_MIN_ROWS=1 FLLM_XDNA_MIN_ELEMS=1 FLLM_XDNA_MIN_EFF=0 fllm-rs selftest-gpu --device npu:0 --only linear,host,embedding`):
exact pass (offload paused) 183 checks ok; fast pass 576 checks, 0 failed, 32 skipped (formats the CPU reference does not support), 388
projections computed on the NPU; in the Q8_0 and Q4_K cases printed in an earlier run the largest error was 6.4e-4 of the error scale, the tolerance is 4e-3.

**Whole model** (Llama-3.2-3B-Instruct Q4_K_M, `fllm-rs serve`, one request at a time, prompts of random words, 1 generated token, prompt tokens
per second of the whole request, median of 3-5 runs):

| where the prompt runs | 1180-token prompt | 400-token prompt |
|---|---|---|
| processor, 6 threads | 131-136 tok/s | 168 tok/s |
| NPU (host decoder), 2.8 GiB of bf16 weights resident, the rest streamed | 168-169 | 190-192 |
| NPU (host decoder), all weights resident (5.4 GiB of bf16) | 179-180 | 212-231 |
| iGPU, fLLM Vulkan (same laptop) | 211-216 | 237 |
| FastFlowLM on the NPU (docs/benchmarks.md, other prompt) | 306 | |

Two other series of the "NPU, 2.8 GiB" row gave 90-133 (made while another benchmark job was running on the laptop) and 130-138 (cause unknown,
probably the same): the offload shares the processor with everything else, so treat the 168-169 as the figure for an otherwise idle machine.
The last runs of the iGPU row (136-164 on the 400-token prompt) were disturbed the same way and are not in the table. So with enough memory for the bf16 weights the NPU matches the iGPU on a ~400-token prompt and is 15 % behind on a ~1200-token
one; it is clearly faster than the processor, and it leaves both free. FastFlowLM, with AMD's kernels, is faster than ours. Where the time goes
with all weights resident (`FLLM_XDNA_STATS`): 17.7 s of 25.7 s in the offload were spent waiting for the NPU, 4.9 s adding partial sums, 1.9 s
converting activations, 1.3 s packing; the rest of the prompt (attention, norms) runs on the processor. Decode is the processor's, unchanged
(NPU decode is not attempted).

The `--device npu` hybrid backend was checked by the self test only: loading Llama-3.2-3B into it needed more than 9 GB (the reference CPU
backend's copies plus the NPU registry) on a 14 GB machine that also hosts the GPU, so a whole-model run was not made.

## Making the programs

`scripts/xdna-build-kernels.sh TOOLS_DIR` rebuilds `rust/engine/xdna/kernels/` from the open toolchain (needs `xclbinutil`). The design is
mlir-aie's `programming_examples/basic/matrix_multiplication/whole_array/whole_array.py` (v1.4.4) with `--dev npu2 --n-aie-cols 8 --b-col-maj 1
--dtype_in bf16 --dtype_out f32 -m 32 -k 64 -n 64`. Its `.xclbin` does not depend on the matrix shape (checked: an instruction stream built for
another shape runs on it), so the build keeps one `.xclbin` and one instruction stream per shape. Tile sizes: a 64 x 64 f32 output tile does not
fit the 64 KB of a compute tile with double buffering (`aie.buffer ... could not be placed`); 32 x 64 x 64 does. The kernels are compiled from
mlir-aie's `aie_kernels` with Peano; mlir-aie and llvm-aie are Apache-2.0 with LLVM exceptions ([THIRD_PARTY.md](../THIRD_PARTY.md)). The files
in the repository are generated, not written by hand. Re-running the script gave byte-identical instruction streams and an `.xclbin` of the
same size that differs from the shipped one only in header bytes (it starts at byte 297: the identifier and time stamp).

## Not done, and what would help

* Dequantization on the NPU (Q4_K/Q8_0 kept quantized in the NPU buffers, expanded in the compute tiles): needs a custom AIE kernel; mlir-aie
  has a `q4nx` kernel for FastFlowLM's own format, not for GGUF blocks. Today the weights are expanded by the processor into bf16 (2 bytes per
  weight in memory, against 0.56 for Q4_K), which is what limits the model size that can stay resident.
* A matrix-vector program for decode (one row): the NPU's 3 ms floor is its 128-row tile; a vector kernel could read the weights once at
  memory speed, but decode is bandwidth bound and the iGPU and the processor have the same LPDDR.
* Larger output tiles than 1024 columns, the bfp16 emulation of bf16 products (`--emulate-bf16-mmul-with-bfp16`, faster, less precise: not used),
  fusing the activation conversion and the partial-sum addition into the NPU program.
* XDNA 1 (`npu1`, 4 columns), Strix Halo (more memory, more room for resident weights) and Windows: untested.

## Windows (written down, not tested)

Nothing below has been run by us. It restates what mlir-aie's native-Windows guide (`buildHostWinNative.md` in its docs folder, v1.4.4) and XRT say.

* Driver: the AMD NPU driver from AMD Software / Adrenalin; mlir-aie names 32.0.20101.3760 (XRT 2.21.0) as the minimum. `C:\Windows\System32\AMD\xrt-smi.exe examine` shows the NPU.
* XRT: the Windows XRT SDK (`xrt_windows_sdk.zip` of the XRT 2.21.75 release on GitHub, unpacked to `C:\Xilinx\XRT` in that guide) supplies the headers
  and `xrt_coreutil.lib`; the runtime itself comes with the driver (the guide runs `xrt-smi.exe` from `C:\Windows\System32\AMD`).
* Shim: `cl /LD /EHsc /std:c++17 /I<XRT include> fllm_xdna_shim.cpp /link /LIBPATH:<XRT lib> xrt_coreutil.lib /OUT:fllm_xdna_shim.dll` (the source
  already exports with `__declspec(dllexport)`). fLLM looks for `fllm_xdna_shim.dll` next to `fllm-rs.exe`, then in `%LOCALAPPDATA%\fllm\npu`,
  or at `FLLM_XDNA_SHIM`.
* The `.xclbin` and instruction streams are the same files: they are embedded in the binary, there is no path to configure.
* Unknown: whether the hardware-context calls of the shim behave the same under the Windows driver, and the memory-lock rules there.
