# Study: AMD XDNA (Ryzen AI NPU)

Research note, not user documentation. Test level of the accompanying software: **L0** (see [test-levels.md](../test-levels.md); it compiles; it has not run on any Ryzen AI machine).
Sources: Ryzen AI Software 1.8 documentation (ryzenai.docs.amd.com: Model Compilation and Deployment, Application Development, Linux
Installation Instructions), `onnxruntime_c_api.h` v1.22.0, the FastFlowLM README (github.com/ROCm/FastFlowLM). Statements that do not come from
these sources are marked "to the best of our knowledge" or "not verified".

## 1. Access from Rust

The route supported by AMD is **ONNX Runtime with the VitisAI execution provider** (VAI EP):

- Models are ONNX files (opset 17 recommended). When an `InferenceSession` is first loaded, the model is compiled for the NPU; the
  result lives in the VAI EP cache or as an ORT EP-context model. The EP splits the graph automatically into NPU and CPU subgraphs.
- The C API (`OrtGetApiBase` -> `OrtApi`) is the normal ONNX Runtime C API; the Ryzen AI `onnxruntime` is a drop-in. For the EP there is
  `SessionOptionsAppendExecutionProvider_VitisAI(options, keys, values, n)` (since ORT 1.18) and the generic
  `SessionOptionsAppendExecutionProvider(options, "VitisAI", ...)`.
- Provider options (from the AMD documentation): `config_file` (BF16 compiler options), `target` (`X1` = Phoenix/Hawk Point, `X2` = Strix/Krackan),
  `xclbin` (INT8 CNN on PHX/HPT only), `cache_dir`, `cache_key`, `enable_cache_file_io_in_mem`, `encryption_key`, `opt_level`.
- Device detection according to AMD is by PCI ID: vendor `0x1022`, device `0x1502` (Phoenix/Hawk Point, XDNA 1), device `0x17F0` with revision
  `0x00/0x10/0x11` (Strix, XDNA 2) and `0x20` (Krackan). On Linux this is found in `/sys/bus/pci/devices/*/{vendor,device,revision}`.
- Windows: Ryzen AI Software (MSI installer, NPU driver from 32.0.203.xxx; the documentation names minimum/maximum drivers per EP version).
  Linux: since Ryzen AI 1.8 for Strix and newer (Ubuntu 24.04, XRT packages + the `amdxdna` driver package, Python venv installation, then
  `source /opt/xilinx/xrt/setup.sh`). macOS: not available.
- On AMD's stack, LLMs run through `onnxruntime-genai` (OGA) with specially prepared models (NPU-only or hybrid with the iGPU) and their own
  custom ops (`ryzen_mm.dll`, `onnx_custom_ops.dll`); the models must be quantised and built with AMD's tools (Quark). GGUF is not
  read.

**In the scaffold** (`xdna.rs`, `xdna/`): `ort.rs` binds the ONNX Runtime C API at run time (`libloading`, nothing is linked). The slot numbers
of the `OrtApi` entries used (position in `struct OrtApi`, e.g. `CreateEnv` = 3, `Run` = 9, `SessionOptionsAppendExecutionProvider_VitisAI`
= 276) were counted with a script from the v1.22.0 header (318 entries). The search order is `$FLLM_XDNA_ORT_LIB`, then
`$RYZEN_AI_INSTALLATION_PATH/deployment/onnxruntime.dll` or `.../onnxruntime/lib/libonnxruntime.so`, then the default name. `list()` reports
a device if the library loads **and** `GetAvailableProviders` names the `VitisAIExecutionProvider` (an ordinary ORT build does not name
it); the name comes from the PCI detection (Linux only). `probe_matmul(m, k, n)` uses `onnx.rs` to build an ONNX model with a single
`MatMul` (a self-written protobuf emitter, field numbers checked against `onnx.proto3`), creates the session from memory
(`CreateSessionFromArray`), runs it once and compares with a CPU reference.

## 2. Data formats

| Path | Format | Remark |
|---|---|---|
| CNN, XINT8 / A8W8 / A16W8 | INT8 (QDQ-quantised ONNX models) | compiler "X2" (default) for STX/KRK, "X1" for PHX/HPT |
| CNN, NLP encoder | BF16 | STX/KRK or newer only; the deployment version of the EP does not compile BF16 at run time, the models must be shipped precompiled |
| LLM (OGA) | AWQ INT4 weights, BF16/INT activations following AMD's recipe | block sizes/layout are determined by AMD's model build, not documented |

fLLM's formats (Q4_0/Q4_1/Q5_x/Q8_0, K-quants, IQ4_NL, MXFP4, NVFP4, Q4Z, FP8M) have **no counterpart** on the ONNX/VitisAI route; a
linear layer would have to be re-exported as a `MatMul` with BF16 weights or as QDQ INT8. Q4Z/Q4ZP (AWQ/GPTQ) is closest in content to
AMD's own INT4 recipe, but the weight layout of the AMD custom ops is not publicly documented (not verified).

## 3. Limits

- **Static shapes, compile time.** Every session is compiled the first time (seconds to minutes per model; not measured). One
  model per weight matrix would mean hundreds of sessions; a whole model as one ONNX graph needs an export with a fixed
  context length.
- **No operator-by-operator access.** fLLM's backend trait calls individual ops with device buffers. Through ORT, data passes through ORT tensors
  (host memory) on every `Run`; there is no public way to keep NPU device buffers and pass them between sessions
  (not verified).
- **Memory.** XDNA shares the LPDDR memory with the CPU and iGPU. Decode (one row per step) is bandwidth bound; the NPU has no advantage over
  the CPU/iGPU there, except in energy.
- **Licence.** Ryzen AI Software is under AMD's own licence; DLLs/packages must not go into the fLLM repository. Therefore everything is bound with
  `dlopen`/`LoadLibrary`. The exact redistribution terms were not checked.
- **Availability.** Windows 11 (PHX to KRK) and Linux (from Strix). No macOS. For minimum driver levels see the AMD documentation; a cache must not be
  reused across EP/driver versions.

## 4. Expected benefit

- **Prefill:** compute bound; an NPU with tens of TOPS can help here - but only with BF16/INT8 weights in AMD's format, not with fLLM's
  GGUF blocks. Measuring the call overhead (session `Run` for a single `MatMul`) is exactly what `probe_matmul` is meant to deliver
  in a hardware test.
- **Decode:** bandwidth bound; an NPU offload per linear layer with a host round trip per layer is very likely slower than fLLM's
  AVX2/VNNI CPU path (assumption, not measured). The benefit would be energy, not speed.
- Independent reference: **FastFlowLM** (ROCm/FastFlowLM; orchestration MIT, the NPU kernels are binaries which, according to the README, are free to use, also
  commercially, with attribution) runs LLMs on XDNA 2 with its own kernels (IRON/AIE-MLIR), has Windows and, since 03/2026, Linux support,
  and an OpenAI-compatible server (port 52625). The speed/energy figures in its README are vendor figures and have not been verified here.

## 5. Recommendation (honest)

1. **Do not build an XDNA `Backend` on ONNX Runtime/VitisAI.** Neither the formats nor the call model fit fLLM, and the expected
   decode gain is zero to negative.
2. **If XDNA support is wanted:** (a) put fLLM in front of a running FastFlowLM/Lemonade server as a frontend (a different
   product feature from a backend, no compute path of its own), or (b) write own AIE kernels for Q4 matmul with IRON/mlir-aie - a
   project of its own, not a scaffolding topic.
3. **Hardware test (L1):** on a Ryzen AI machine call `probe_matmul(1, 4096, 4096)` and `probe_matmul(128, 4096, 4096)` (see
   Section 6), check with `XLNX_ONNX_EP_REPORT_FILE=report.json` whether the node landed on the NPU or the CPU, and note the times.
   Only then decide whether point 2 is worthwhile.

## 6. Verification instructions

Prerequisite: Ryzen AI Software installed (Windows: `RYZEN_AI_INSTALLATION_PATH` set; Linux: venv activated, XRT `setup.sh`
sourced, `LD_LIBRARY_PATH` pointing at `$RYZEN_AI_INSTALLATION_PATH/onnxruntime/lib`).

- Build with `--features npu-xdna`; `fllm-rs devices` must show "AMD XDNA NPU (...)" as a device of type `accel`. If it is missing, the program
  `create(0)` and the error message name every library path tried.
- Windows: absolute library paths are loaded with `LOAD_WITH_ALTERED_SEARCH_PATH` (`xdna/dl.rs`) so that the DLLs in the Ryzen AI folder
  `deployment` find their neighbours; if loading still fails with error 126, add the folder to `PATH`.
- Options for the EP through environment variables: `FLLM_XDNA_TARGET` (`X1`/`X2`), `FLLM_XDNA_CACHE_DIR`, `FLLM_XDNA_CONFIG`.
- `backend::npu::xdna::probe_matmul(m, k, n)` returns `ProbeReport { ort_version, api_version, providers, session_ms, run_ms, max_abs_err }`.
  Expected is `max_abs_err` in the range of float32 rounding if the node runs on the CPU, and in the range of BF16 rounding (about 1e-2
  relative) if the EP places it on the NPU. No caller (e.g. a self-test branch) is wired up yet: the XDNA probe is not part of `selftest.rs`.
- Known uncertainties: whether the EP hands a plain float32 `MatMul` to the NPU at all (the documentation names INT8 QDQ and BF16 compile flows);
  whether `SessionOptionsAppendExecutionProvider_VitisAI` without `config_file` suffices for BF16; whether all 18 bound `OrtApi` slots exist in the
  API version offered by the Ryzen AI build (versions 22 down to 17 are requested; slot 276 only from 18).
