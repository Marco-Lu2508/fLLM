# Adding a backend

A backend runs the operations of a decoder on an accelerator: CUDA-like GPUs, Vulkan, Metal, SYCL, NPUs, TPUs. This page describes
the contract and the steps; read [architecture.md](architecture.md) first. The existing backends under
`rust/engine/src/backend/` (`cuda.rs`, `vulkan/`, `metal/`, `sycl/`, `hip/`, `npu/`, `tpu/`, `wgpu_backend.rs`) are the best examples,
and `backend/stub.rs` is the scaffold that reports every operation as unsupported.

## The contract

Two traits in `rust/engine/src/backend/mod.rs`:

* `Backend` (device level): `name`, `device_id`, `capabilities`, `create_stream`, `allocate_buffer`/`free_buffer`, `upload_weight`
  (the quantized bytes of a tensor, its type, shape and optional bias; the result is a `DeviceWeight` that stays resident),
  `upload_weight_scaled`, the pointer-table helpers for mixture-of-experts, and `allocate_kv_cache` (the paged cache pool).
* `Stream` (an ordered queue of work): `synchronize`, `upload`/`download`/`copy_buffer`, and the operations: `embedding_lookup`,
  `rms_norm`, `layer_norm`, `rope`, `quantized_linear`, activations (`swiglu`, `geglu`, ...), `attention_paged`, `kv_cache_write`,
  `residual_add`, the MoE operations, and the layer-level operations for MLA, Mamba 1/2, short convolution and DeltaNet. Read the
  current trait for the full list; it grows.

Rules that make a backend correct and incremental:

1. **Everything is optional.** Every `Stream` method outside the minimum has a default returning `BackendError::UnsupportedOperation`.
   The decoder falls back to the host for that layer. Implement the smallest useful set first (`quantized_linear` for a few formats,
   norms, RoPE, attention, KV write) and report exactly what you support in `BackendCapabilities` (formats and `supports_*` flags).
   Never return a wrong result for an unsupported case.
2. **Weights are uploaded in their native quantized form.** Convert at upload only when the device has no kernel for the format, and log
   the conversion. Document which formats are native and which are converted.
3. **The KV cache layout is shared.** `kv_format.rs` defines the block layout and format numbers; a backend either implements the
   formats it advertises or rejects the others at `allocate_kv_cache`.
4. **Asynchronous by design.** Operations enqueue; `synchronize` and `download` wait. Do not block per operation.
5. **Numerics are checked against the CPU.** Kernels may round differently from the CPU path, but they must pass the self test
   tolerances, and an "exact" environment switch (for example `FLLM_<BACKEND>_EXACT=1`) should select the most CPU-like kernels for
   diagnosis.

## Steps

1. **Cargo feature.** Add a feature to `rust/engine/Cargo.toml` (and the dependencies it needs as optional dependencies). Keep it out of
   the default set unless every platform can build it. If kernels are compiled from source, extend `rust/engine/build.rs` so that the
   feature compiles them and fails with a clear message when the toolchain is missing.
2. **Module.** Create `rust/engine/src/backend/<name>/` and declare it in `backend/mod.rs` behind `#[cfg(feature = "<name>")]` (and the
   target `cfg` if it is platform specific).
3. **Registration.** Wire the name into the native-backend functions in `backend/mod.rs`: the list of backend names, the device listing
   used by `fllm-rs devices` (name, memory, capabilities, the `--device` value to select it), and `create_native_backend`. Decide
   whether `--device auto` may pick it (the order is in `devices.rs`); an experimental backend is only used with an explicit `--device`.
4. **Capabilities and first kernels.** Start from the stub. Implement `quantized_linear` for one format, then the rest of a dense
   transformer layer, then more formats, then MoE and the recurrent layers.
5. **Self test.** `fllm-rs selftest-gpu --device <name>` already runs every operation of the `Stream` trait on random data against the
   CPU backend and an f64 reference and groups the checks (`--list` shows the groups). Add cases in `backend/selftest*.rs` for every
   new operation or format. Unsupported operations show as `SKIPPED (unsupported)`. Test levels: the self test on the device is L1
   ([test-levels.md](test-levels.md)). Keep hardware-only unit tests `#[ignore]`d with a reason so that CI stays green.
6. **Text check.** Run a real model and compare the greedy text with the CPU (L2). The `scripts/verify-*.sh` and `.ps1` scripts share
   `verify-common.sh` / `verify-common.ps1` (build, device list, self test, text comparison, benchmark, JSON report); copy the closest
   one, set the backend variables at its top and add backend-specific pre-steps. [backends-verify.md](backends-verify.md) documents the
   procedure.
7. **CI.** Add the feature to the compile checks: `platforms.yml` builds feature sets on the hosted runners, `kernels.yml` compiles GPU
   kernels in vendor containers, and `conformance.yml` runs `devices` and `selftest-gpu` on a self-hosted runner with the hardware: add
   your backend to its `backend` choices and the feature mapping. Pin new third-party actions to a commit SHA and run `actionlint`.
   See [ci.md](ci.md).
8. **Documentation.** Add a section to [backends.md](backends.md) and [gpu-backends.md](gpu-backends.md) with the build requirements,
   `--device` value, environment switches (also in [configuration.md](configuration.md)), the per-format and per-operation status with
   an honest test level, and known limits. Add the backend to the README Status table only with a level you reached on hardware.
   Third-party code goes into [THIRD_PARTY.md](../THIRD_PARTY.md).
9. **Pull request.** State exactly what ran where ([CONTRIBUTING.md](../CONTRIBUTING.md)).

## Checklist for reviewers

* Does `capabilities()` match what the stream implements (no advertised format without a kernel)?
* Are failures reported as errors, not panics or silent wrong results? Are `unsafe` blocks commented?
* Does the build still work on the platforms without the feature (`cfg` guards, no unconditional dependencies)?
* Are the environment switches documented, with defaults?
* Is the claimed test level supported by a record of the command and device?
