# The AMD XDNA NPU with fLLM's own kernels

fLLM can compute on the NPU of an AMD Ryzen AI processor (XDNA 2, "npu2" / AIE2P) with kernels of its own, loaded and run through XRT. No
vendor inference library is involved: not FastFlowLM, not ONNX Runtime, not AMD's proprietary `.xclbin` files. The route through FastFlowLM
([user/amd-npu.md](user/amd-npu.md)) stays as it was and works without any of this.

**Status (2026-10-30).** Built and measured on one machine: a Ryzen AI 5 PRO 435 laptop (Radeon 840M, 14 GB LPDDR, Ubuntu 26.04, kernel 7.0,
`amdxdna` driver, NPU firmware 1.1.2.64, XRT 2.21.75 from the distribution). Everything else is untested: Windows, other XDNA 2 chips (Strix
Point, Strix Halo), XDNA 1 (Phoenix / Hawk Point: a different architecture, `npu1`, not built), macOS and Linux on aarch64 compile only.

## What it does and what it does not

* **Prompt processing: yes.** The large projections `y = x . W^T` of a prompt chunk (64 or more rows) run on the NPU; everything else runs on
  the processor.
* **Q4_0, Q4_1 and Q4_K weights stay 4-bit** (0.625 byte per weight with group scales; 1.4 GiB for the 3B model) in NPU-visible buffers and are
  expanded inside the compute tiles, right before the multiply, on the NPU itself (round 2 below). **Other formats** (Q6_K, Q8_0, F16 ...) are
  decoded to bf16 by the processor (fast loops for Q4_0, Q8_0, Q4_K, Q6_K, F16, BF16, F32, the generic dequantizer for the rest), held as bf16
  tiles while the budget lasts and multiplied with bf16 products at 2 TFLOPS; formats without a dequantizer (Q4Z, bitsandbytes, EXL2/3 ...)
  are computed on the processor.
* **Decode (one row at a time): no.** The NPU is slower than the processor for it in every setup measured (round 2), so the rows stay on the CPU.
* **Precision.** Default (`FLLM_XDNA_PRECISION=fast`): 4-bit weights are expanded to block floating point (eight values share an exponent, 8-bit
  mantissas) and the activations are converted the same way inside the tile; products accumulate in f32. A projection is off by about 0.75 % of
  its largest result (about the size of int8 activation quantization); the self test passes its 4e-3 tolerance, and a 714-token prompt through
  Llama-3.2-3B gives the same 60 greedy tokens as the processor. `exact`: bf16 products (error about 4e-4 of the error scale), at 2 to 2.5 times
  lower speed.

## Parts

| Part | Where | What |
|---|---|---|
| NPU programs | `rust/engine/xdna/kernels/` | four `.xclbin`s, each with instruction streams per `(rows, depth)`, embedded in the binary: bf16 x bf16 -> f32 GEMM (`gemm_bf16_f32_8col`, tile 32 x 64 x 64, `B` stored `[n][k]`); the same with 4-bit weights expanded to bf16 in the tile (`gemm_dq_q4_8col`); 4-bit weights expanded to block floating point and multiplied on the native bfp16 MAC (`gemm_dqb_q4_8col`, tile 64 x 64 x 64, 256 / 512 rows per call; `gemm_dqbs_q4_8col`, tile 16, 64 rows per call, for small batches) |
| Tile kernels | `rust/engine/xdna/kernel_src/` | `fllm_dq.cc` (4-bit tile -> bf16 or bfp16), `fllm_mm_bfp.cc` (bf16 x bfp16 matmul with f32 C), `whole_array_dq*.py` (the array designs, derived from mlir-aie's `whole_array.py`), `dq_pack.py` (host reference of the packed layout) |
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
| `FLLM_XDNA_RESIDENT_MB` | weight memory the NPU may hold (default: 25 % of the free memory at start; 4-bit weights count 0.625 byte, bf16 tiles 2). Weights over the budget are packed again for every call ("streamed", bf16 program) |
| `FLLM_XDNA_PRECISION` | `fast` (default: block floating point on the native MAC for 4-bit weights) or `exact` (bf16 products) |
| `FLLM_XDNA_SMALL=0` | do not use the 64-row program for batches of up to 64 rows |
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

## Round 2: weights packed on the NPU, the block-floating-point MAC (2026-10-30)

Goal set after the first round: close the gap to FastFlowLM (306 prompt tokens/s on this 3B model). What was measured and built, in the order it
was found.

**Where the time went (profile of round 1, bf16 weights resident, 1180-token prompt, 6.6 s).** The NPU itself ran at 2.0 TFLOPS and was busy
(2.9 s of waiting, 44 %); the processor spent 2.0 s on attention (30 %), 0.2 s on norms and RoPE, 1.1 s converting activations and adding partial sums.
An ablation of the array design (kernels with the data movement removed, the multiply repeated 1, 2 and 4 times) showed that the **bf16 matmul
kernel of the compute tile is the limit**: one k-step of a 32 x 64 x 64 tile (131 072 MACs) takes about 3 us, far below what the MAC array can do, and the
time grows linearly with repeats, while the DMAs alone finish in a quarter of the time. Tile shapes (16 to 64), 4 or 8 columns and bf16 with bfp16
emulation (`--emulate-bf16-mmul-with-bfp16`) did not change it by more than 15 %; an int8 design of the same array reaches 3.1 to 5.8 TOPS.

**Weights packed on the NPU.** `fllm_dq.cc` expands a 64 x 64 tile of 4-bit codes with bf16 group scales and minima to bf16 inside the compute tile
(it follows `q4nx_dequant.cc` of mlir-aie; the layout is `dqpack.rs` / `dq_pack.py`: 2560 bytes per tile, 0.625 byte per weight). The engine
repacks GGUF Q4_0, Q4_1 and Q4_K blocks into that stream once. Checked on the NPU against the host reference: relative error 5e-7 of the result
(the products are exact bf16 products), 1.6 to 2.3 TFLOPS, at the speed of the bf16 program with 3.2 times less memory (the 3B model needs 1.4 GiB
for its 4-bit tensors instead of 5.4 GiB of bf16). The unpack costs about 20 % of the run time (the same array with the unpack removed was 20 % faster).

**The native block-floating-point MAC.** mlir-aie's `mm_bfp_mixed.cc` multiplies a bf16 A tile (converted to bfp16 in the core) with a bfp16ebs8 B
tile on the bfp16 MAC of AIE2P, which is 2.5 times faster than the bf16 MAC on this array: 4.5 to 5.7 TFLOPS in a whole-array test with 64 x 64 x 64
tiles. `fllm_mm_bfp.cc` is that kernel with an f32 C tile (so the partial sums of the K reduction are not rounded to bf16 between the k-steps), and
`fllm_dq.cc` now also writes the expanded tile as bfp16ebs8 (`fllm_dq_q4_bfp`). Together: 4-bit weights in, native MAC, f32 out.

| One projection of 8192 x 3072 Q4_K (rows of `x`) | NPU | CPU (6 threads, AVX-512) |
|---|---|---|
| 512 | 4.55 ms = 5.7 TFLOPS | 21.5 ms = 1.2 TFLOPS |
| 256 | 2.20 ms = 5.9 TFLOPS | 11.9 ms |
| 128 | 1.95 ms | 6.7 ms |
| 64 | 1.27 ms | 4.1 ms |
| 16 | 1.24 ms | 2.6 ms |
| 8 / 4 | 1.22 ms | 2.5 / 2.7 ms |
| 3 / 2 / 1 | 1.22 ms | 1.0 / 0.94 / 0.91 ms |

(quiet machine, best of 10, resident weights; 3072 x 3072: 1.93 ms at 512 rows against 9.4 ms.) The 64-row program has a floor of about 1.2 ms for a
projection of 8192 outputs because every call computes 64 padded rows.

**Whole model, Llama-3.2-3B-Instruct Q4_K_M, prompt tokens per second of the whole request** (`scripts/npu_prompt_bench.py`, random-word prompts, one
request at a time, no other benchmark process seen in `ps`; two series each):

| prompt | processor | NPU, exact products (round 1 + 4-bit) | NPU, default (block floating point) | FastFlowLM |
|---|---|---|---|---|
| ~1180 tokens | 117-131 | 167-169 | **222-227** | 306 (other prompt) |
| ~400 tokens | 155-168 | 187-190 | **310-314** | 306 (other prompt) |
| ~160 tokens | 146-171 | 136 | **261-265** | |

For a prompt of a few hundred tokens the NPU path now reaches the 306 tokens/s of FastFlowLM (the figures are not for the same prompt, and
FastFlowLM's include its own attention on the NPU); for a 1180-token prompt it does not. The 4-bit tensors (1.4 GiB) plus the bf16 tiles of the
Q6_K tensors fit the default budget, so nothing is streamed.

**Where the time goes now** (`FLLM_RUST_PROFILE=1`, `FLLM_XDNA_STATS`; ms of one request):

| phase | 400 tokens | 1180 tokens | the same 1180 tokens on the processor |
|---|---|---|---|
| attention (processor) | 244 | 1850 | 2077 |
| MLP (gate, up, down) | 649 | 2032 | 4854 |
| q, k, v and output projections | 323 | 1018 | 1618 |
| norms, RoPE, KV write | 79 | 227 | 434 |
| total | 1304 | 5139 | 8998 |
| of the NPU phases: waiting for the NPU | about 660 | 1840 | |
| of the NPU phases: activation conversion + copy of the result tiles | about 270 | 840 | |

At 1180 tokens the **attention on the processor is now the largest single part (36 %)** and grows with the square of the prompt; the 306 tokens/s
target there (3.85 s) is out of reach without attention on the NPU or faster processor attention. mlir-aie has a flash-attention prefill kernel,
but only for head dimensions 256 and 512 (Gemma); Llama has 128, so it would need a kernel of its own, which was not written. Second: the NPU
phases cost 840 ms of processor work around 1840 ms of NPU time (converting x to bf16 for every projection, copying the f32 tiles into the
result); batching the q/k/v and gate/up projections of a layer into one submission would save the conversion but the host decoder calls them one
by one with the same input, so it was not done.

**Decode.** One row cannot use the array: a call computes at least 64 rows, 1.2 ms for an 8192 x 3072 projection against 0.9 ms on the
processor, and a 3072 x 3072 one is a tie (0.51 against 0.68 ms). From 4 rows up the NPU call is 2 times faster than the processor, but the whole
decode step did not profit: with the host decoder at `FLLM_XDNA_MIN_ROWS=4`, 4 simultaneous 48-token requests produced 14.3 tokens/s against 19.7
on the processor and 8 requests 21.2 against 23.1 (default settings: 16.0 and 26.2, within the noise of a machine that other jobs share). The
weight stream is not the limit (8192 x 3072 in 4-bit is 15.7 MB per 1.2 ms, 13 GB/s, far below the LPDDR); the padded rows and about 200 calls
per token are. A matrix-vector design that splits the output columns over all 32 tiles instead of the rows was not built. **Decode stays on the
processor; the target "decode at least as fast as the processor" was not reached, and with this design it cannot be.**

**What is not done:** attention on the NPU, batching projections into one submission, a decode matrix-vector design, 4-bit programs for Q6_K and
Q8_0 (they use the 2 TFLOPS bf16 program), the bf16 and exact 4-bit programs with 64-row tiles. Other agents' jobs shared the laptop during most
of the session; a run was only used when `ps` showed nothing else (the earlier figures of up to 30 % lower NPU throughput under load are not in
the tables).

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
