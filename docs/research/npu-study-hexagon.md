# Study: Qualcomm Hexagon NPU (HTP) via QNN and Genie

Research note, not user documentation. Test level of the accompanying software: **L0** (see [test-levels.md](../test-levels.md); it compiles, and structure sizes are checked by unit tests against the values derived from the
C declarations; it has not run on any Snapdragon device).
Sources: the headers of QAIRT SDK 2.37.0.250724 (`QnnInterface.h`, `QnnTypes.h`, `QnnGraph.h`, `QnnDevice.h`, `QnnOpDef.h`,
`HTP/QnnHtpDevice.h`, `HTP/QnnHtpGraph.h`, `HTP/QnnHtpContext.h`, `HTP/QnnHtpMem.h`, `System/QnnSystemInterface.h`, `Genie/GenieCommon.h`,
`Genie/GenieDialog.h`; core API 2.27, Genie API 1.11), read from the publicly downloadable "Qualcomm AI Runtime Community" package
(`softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.37.0.250724/v2.37.0.250724.zip`; only the headers
were read, using HTTP range requests). Statements that do not come from these headers are marked "to the best of our knowledge" or "not
verified".

## 1. Access from Rust

- **Licence first:** the SDK headers carry "Confidential and Proprietary - Qualcomm Technologies, Inc.". They are therefore **not
  copied and not placed in the repository**; `hexagon/qnn.rs` reproduces only the binary interface (order and size of the fields,
  constants) in our own Rust structures. SDK libraries are not linked but loaded at run time.
- **QNN C API.** Every backend library (`libQnnHtp.so` on Linux/Android, `QnnHtp.dll` on Windows on Snapdragon) exports
  `QnnInterface_getProviders(const QnnInterface_t*** list, uint32_t* n)`. `QnnInterface_t` = `{ backendId, providerName, apiVersion, union
  of the versioned function tables }`; in header order the v2.x table has `propertyHasCapability, backendCreate, ..., graphExecute (21),
  tensorCreateGraphTensor (24), ..., deviceCreate (40), deviceFree (43), errorGetMessage (48), ...`. The scaffold declares the table up to
  `errorFreeVerboseMessage` (51 entries, offsets checked by a unit test) and requires core API 2.>=17 (assumption: all 51 entries exist from
  there; not checked against the changelog, current SDKs ship 2.27).
- **Flow for one model** (as `hexagon/graph.rs` executes it): `backendCreate` -> `deviceCreate` -> `contextCreate` -> `graphCreate`
  (with HTP config: precision FP16) -> `tensorCreateGraphTensor` per tensor (input `APP_WRITE`, weight `STATIC` with a client buffer, output
  `APP_READ`) -> `graphAddNode` (`qti.aisw`/`MatMul`, parameters `transpose_in0/1` as `BOOL_8`) -> `graphFinalize` (compiled for the HTP here) -> `graphExecute`. Contexts can be saved as a binary file (`contextGetBinary`) and loaded again
  (`contextCreateFromBinary`): this is how LLM models generated offline with the QAIRT tools on an x86 PC work.
- **Device data.** `deviceGetPlatformInfo` returns, per HTP, `deviceInfoExtension` with `arch` (V68, V69, V73, V75, V79, V81), `socModel`,
  `vtcmSize` and support for Signed PD/DLBC. `list()` shows "Qualcomm Hexagon NPU (HTP v73, SoC model ...)" from this.
- **Genie** (`Genie.dll`/`libGenie.so`, API 1.11) is the LLM layer on top of QNN: `GenieDialogConfig_createFromJson` + `GenieDialog_create` +
  `GenieDialog_query(dialog, prompt, sentenceCode, callback, userData)` (streaming callback). Configuration: JSON with tokenizer, sampler and
  the model's precompiled context binaries. `hexagon/genie.rs` wraps exactly these calls (`Genie::query(json, prompt)`).
- An alternative that was not built: the QNN execution provider of ONNX Runtime (`QNNExecutionProvider`, backend type "htp").
  It would have the same limits as the XDNA route (see [npu-study-xdna.md](npu-study-xdna.md)).

## 2. Data formats (from `QnnTypes.h`)

- Tensor data types: `INT_8/16/32/64`, `UINT_8/16/32/64`, `FLOAT_16/32/64`, `SFIXED_POINT_4/8/16/32`, `UFIXED_POINT_4/8/16/32`, `BOOL_8`.
- Quantisation kinds: `SCALE_OFFSET`, `AXIS_SCALE_OFFSET` (per channel), `BW_SCALE_OFFSET`, `BW_AXIS_SCALE_OFFSET` (bit width + axis),
  `BLOCK`, **`BLOCKWISE_EXPANSION`** (axis, blocks per axis, scales as 8- or 16-bit storage, bit width), `VECTOR`. Tensor data formats:
  `DENSE`, `SPARSE`, `CODEBOOK`, `MX`, `HMX_WEIGHT_LAYOUT`.
- In content, **Q4_0** (blocks of 32, FP16 scale, symmetric) fits best onto `SFIXED_POINT_4` with `BLOCKWISE_EXPANSION`; Q4_1/Q5_x/K-quants/
  IQ4_NL/MXFP4/NVFP4 would have no direct counterpart and would have to be converted (loss of accuracy, not verified). Which combinations
  the HTP actually accepts as `MatMul`/`FullyConnected` per architecture is not stated in the headers (op definition documentation, not read).
- HTP graph options (`QnnHtpGraph.h`): precision (`QNN_PRECISION_FLOAT16`), VTCM size, number of HVX threads, `WEIGHTS_PACKING`,
  `SHARE_IO_BUFFER`, parallel graph execution. Context options (`QnnHtpContext.h`): `WEIGHT_SHARING_ENABLED`, `REGISTER_MULTI_CONTEXTS`,
  `FILE_READ_MEMORY_BUDGET`, `PREPARE_ONLY`, `SHARE_RESOURCES`. Shared buffers (`QnnHtpMem.h`): `SHARED_BUFFER` (fd + offset), `WEIGHTS_BUFFER`,
  `SHARED_SPILLFILL_BUFFER`.

## 3. Limits

- **Static graphs.** An HTP graph has fixed shapes; LLMs are therefore stored as several graphs (e.g. prefill with a fixed chunk length and
  decode with one token) in one context, with the KV cache as input/output tensors (to the best of our knowledge; hidden inside Genie).
- **Compile time.** `graphFinalize` compiles on the device: short for the one-layer example, not practical for a whole model; for LLMs
  the context binaries are produced offline. The weights live in the binary file (weight format = the HTP's own layout).
- **Memory.** The address space of the Hexagon DSP is limited (to the best of our knowledge 32 bit, about 4 GiB per process domain); large models are split into
  several contexts with shared weights (`WEIGHT_SHARING_ENABLED`, `REGISTER_MULTI_CONTEXTS`). The host must hand buffers
  to the DSP through `rpcmem`/shared buffers, otherwise they are copied.
- **Licence.** The SDK is proprietary (see above); it must not be redistributed, the user installs it themselves. The terms of the
  community package were not read.
- **Availability.** Windows on Snapdragon (X Elite/X Plus, driver + `QnnHtp.dll` from the SDK), Android, some Linux boards (aarch64).
  Not on x86 Linux/Windows without a Qualcomm chip (there, to the best of our knowledge, only the offline compiler libraries exist; `list()` then finds no
  device) and not on macOS.

## 4. Expected benefit

- **Prefill:** compute bound; the HTP is built for it. With offline-generated Qualcomm models (Genie) this is the use case.
- **Decode:** bandwidth bound; the advantage is mainly energy. A fLLM-native decode that issues one `graphExecute` with a
  host round trip per linear layer is very likely slower than the NEON CPU path because of the call overhead (RPC to the DSP; order of magnitude not measured).
- No measurements exist; Qualcomm's vendor figures for Genie on Snapdragon X were not verified.

## 5. Recommendation (honest)

1. **Do not build a Hexagon `Backend` from fLLM's individual ops** (static graphs, offline compilation, weights in the HTP layout, call overhead).
2. **If Hexagon is wanted: Genie as an "external generator".** `Genie::query` shows that the route takes only a few FFI calls; the
   models, however, must be produced with Qualcomm's tools (QAIRT/AI Hub) for the particular SoC, not from GGUF. That is a different
   product feature (a second engine behind the OpenAI server), not backend work.
3. A dedicated HTP pipeline (GGUF -> QNN graph with `BLOCKWISE_EXPANSION` INT4 weights -> context binary) is a large standalone project with
   a doubtful gain over Genie. Decide only after the measurements in Section 6.

## 6. Verification instructions

Prerequisite: QAIRT SDK installed; `QNN_SDK_ROOT` set (or `FLLM_QNN_HTP_LIB` pointing at `libQnnHtp.so`/`QnnHtp.dll`); on Android/Linux
the `libQnnHtpV*Stub.so`/`...Skel.so` matching the SoC architecture must be findable through `ADSP_LIBRARY_PATH` (to the best of our knowledge).

- Windows: absolute library paths are loaded with `LOAD_WITH_ALTERED_SEARCH_PATH` (`xdna/dl.rs`); on error 126 add the SDK's `lib` folder to `PATH`.
- Build with `--features npu-qnn`; `fllm-rs devices` must show "Qualcomm Hexagon NPU (HTP v...)". On errors `create(0)` shows every
  path tried, or the QNN error code together with the `errorGetMessage` text.
- `backend::npu::hexagon::probe_matmul_f16(m, k, n)` builds/compiles/runs `y = x · wᵀ` in FP16 and returns `ProbeReport { provider,
  api_version, finalize_ms, run_ms, max_abs_err }`. Expected: `max_abs_err` in the range of FP16 rounding (about 1e-2 at k = 4096 and inputs
  in [-1, 1]). Uncertain: whether the HTP accepts an FP16 `MatMul` with a static weight and `transpose_in1 = true` (otherwise an error code from
  `graphAddNode`/`graphFinalize`), whether `deviceCreate` with a null configuration is enough, whether the client buffer of the static tensor is copied at `tensorCreate`
  (the scaffold keeps it alive until the end of the function) and whether version 1 suffices for `Qnn_Tensor_t`.
- `backend::npu::hexagon::genie::genie()?.query(json, prompt)` for an existing Genie model (JSON configuration according to the Genie documentation).
