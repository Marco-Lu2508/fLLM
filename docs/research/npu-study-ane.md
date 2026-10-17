# Study: Apple Neural Engine (ANE) via Core ML

Research note, not user documentation. Test level of the accompanying software: **L0** (see [test-levels.md](../test-levels.md); it compiles and is type-checked for `aarch64-apple-darwin`; the pure Rust
parts (MIL text, `.mlpackage` writer) are checked at byte level by unit tests; it has never run against Core ML).
Sources: the `.proto` files of `apple/coremltools` (BSD-3-Clause: `Model.proto`, `MIL.proto`, `FeatureTypes.proto`) and
`MILBlob/Blob/StorageFormat.hpp`/`BlobDataType.hpp`, `modelpackage/src/ModelPackage.cpp`. Statements about Core ML itself (behaviour, limits,
version levels) are, where they do not come from these files, "to the best of our knowledge" and have not been checked against Apple's documentation.

## 1. Access from Rust

- **There is no public ANE API.** The only supported access is Core ML: the developer chooses only a preference
  (`MLModelConfiguration.computeUnits`: `cpuOnly = 0`, `cpuAndGPU = 1`, `all = 2`, `cpuAndNeuralEngine = 3`), Core ML decides per
  operation whether it runs on the ANE, and falls back to the CPU/GPU for unsupported ops. There is no way to read the placement result
  from the API (Xcode Instruments or `powermetrics` show it). Private frameworks with direct ANE access exist
  (to the best of our knowledge), are undocumented, not App Store compatible, and are deliberately not used here.
- **Rust binding.** In `Cargo.toml`, `npu-ane` depends on nothing (`npu-ane = []`); the `objc2` crates belong to `metal`. Therefore
  `ane/objc.rs` talks to the Objective-C runtime directly (`objc_getClass`, `sel_registerName`, `objc_msgSend` through function pointers with the
  exact signature per message), opens Foundation and CoreML with `dlopen`, and knows only the roughly twenty messages it needs
  (`NSString`, `NSURL`, `NSArray`, `NSNumber`, `MLModel`, `MLModelConfiguration`, `MLMultiArray`, `MLFeatureValue`,
  `MLDictionaryFeatureProvider`). Switching to `objc2` + `objc2-core-ml` is mechanical once the dependencies
  are listed in `Cargo.toml`.
- **Creating a model.** An ML Program (`mlProgram`, specification version >= 6; here 7 = iOS 16/macOS 13) consists of
  `Model` > `MILSpec.Program` > `Function "main"` > `Block` > `Operation`s (`const`, `linear`, ...). `ane/mlpackage.rs` writes exactly that as
  protobuf (own encoder `xdna/pb.rs`, field numbers taken from the `.proto` files and cross-checked), plus the weights in the MIL blob format
  (version 2: 64-byte header, 64 bytes of metadata per blob with sentinel `0xDEADBEEF`, 64-byte alignment, data type `Float16 = 1`), plus
  `Manifest.json` with `itemInfoEntries` (`model.mlmodel` and `weights`, author `com.apple.CoreML`) and `rootModelIdentifier`.
  `ane/mil.rs` prints the same program as MIL text (`program(1.0) { func main<ios16>(tensor<fp16, [m, k]> x) { ... linear(x = x, weight = weight) ... } -> (y); }`)
  for reading and comparing with `coremltools`; Core ML cannot compile this text itself.
- **Flow in `ane/coreml.rs`:** write the `.mlpackage` to a temp directory -> `+[MLModel compileModelAtURL:error:]` -> configuration with
  `computeUnits = cpuAndNeuralEngine` -> `+[MLModel modelWithContentsOfURL:configuration:error:]` -> fill an `MLMultiArray` (Float16) ->
  `-predictionFromFeatures:error:` -> read the output through `strides` (the ANE can return padded outputs) -> compare with a CPU reference.
  `detect()` recognises Apple silicon (aarch64), reads the chip name and memory with `sysctlbyname` and, if available (macOS 14), the core count
  through `MLAllComputeDevices()`/`MLNeuralEngineComputeDevice`.

## 2. Data formats

- To the best of our knowledge the ANE computes in FP16 (newer chips additionally support INT8 activations); inputs/outputs are FP16 `MLMultiArray`s.
- Weights: FP16 as a `const` blob (here), compressed through the `constexpr_*` ops of the newer opsets (INT8/INT4 with scale/zero point, block-wise
  quantisation, palettisation/LUT, sparsity; iOS 17/18 opsets, **to the best of our knowledge**, not verified). These ops are unpacked into dense
  weights at load time or read directly by the ANE - Core ML decides.
- fLLM's formats: Q4_0/Q4_1/Q8_0 could be mapped block-wise to INT4/INT8 with scale; K-quants, IQ4_NL, MXFP4, NVFP4, Q4Z and FP8M could not without
  conversion. There is no way to hand raw GGUF blocks to the ANE.

## 3. Limits

- **Static shapes.** Flexible shapes (enumerated or ranged) are possible but, to the best of our knowledge, often cost the ANE placement.
  An LLM needs a prefill and a decode program (or stateful models, `MLState`, from iOS 18/macOS 15, for the KV cache).
- **Compile time.** `compileModelAtURL` and the first load with ANE specialisation take long for large models (minutes, to the best of
  our knowledge; not measured). A whole 7B model would have to be rewritten from GGUF as an ML Program with all weights at load time and
  compiled.
- **Memory.** Unified memory; Core ML keeps its own copies of the weights (double occupancy compared with the GGUF mmap).
- **Licence.** Core ML is part of the system, no redistribution is needed. The `.proto` files of coremltools are BSD-3-Clause (notice in
  `THIRD_PARTY.md`).
- **Availability.** Apple silicon only (`aarch64-apple-darwin`, macOS 13+ for `cpuAndNeuralEngine`). Linux/Windows: no. Intel Macs: no ANE.

## 4. Expected benefit

- **Prefill:** compute bound, FP16 matmul - the ANE is built for it; a hybrid setup (prefill on the ANE, decode on the Metal GPU) is
  conceivable in principle. Whether that is faster than the Metal GPU of the M chips (to the best of our knowledge often not in throughput, but in energy)
  is open and not measured.
- **Decode:** bandwidth bound; memory is shared with the CPU/GPU. No speed advantage expected, only energy.

## 5. Recommendation (honest)

1. **No ANE `Backend`.** Core ML accepts only models, not individual ops with device buffers; placement cannot be controlled; fLLM's
   weight formats do not fit. The Metal path is the main Apple route.
2. If an experiment is wanted later: a **prefill accelerator** through an ML Program generated at load time (linear layers as `linear`
   with block-quantised weights, iOS 18 opset), using the emitter present here as a starting point. Only the measurement from Section 6
   (compile/load time, `predict_ms` for M = 1 and M = 128+) shows whether that pays off.
3. The emitter is useful even without an ANE: a generated `.mlpackage` can be opened and inspected with `coremltools` or Xcode.

## 6. Verification instructions

Prerequisite: an Apple-silicon Mac, macOS 13 or newer, Xcode command-line tools.

- Build with `--features npu-ane` on macOS (default target platform); `fllm-rs devices` must show "Apple Neural Engine (Apple M...)".
- `backend::npu::ane::probe_linear(m, k, n, ComputeUnits::CpuAndNeuralEngine)` returns `ProbeReport { compile_ms, load_ms, predict_ms, max_abs_err, mil }`.
  Expected: `max_abs_err` in the range of FP16 rounding (about 1e-2 at k = 4096, inputs in [-1, 1]). Then do the same with `CpuOnly` and `All`; the
  time difference and `powermetrics --samplers ane_power` show whether the ANE took part.
- First sources of error that cannot be ruled out: protobuf field numbers/names that `compileModelAtURL` rejects (the error message
  arrives through `NSError.localizedDescription` in the error text); the opset name `CoreML6`; the shape of the `name` attributes; whether `Manifest.json` without
  further fields is enough. Cross-check: produce the same program with `coremltools` (`ct.convert(..., convert_to="mlprogram")`), decompose it with `protoc
  --decode` and compare with the output of `linear_model()`.
