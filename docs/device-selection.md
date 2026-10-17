# Choosing a device: `--device`, `fllm-rs devices`, `auto`, the self test and the tuner

Level of verification of everything in this file: **L0** for the GPU backends (compiles; no GPU backend other than CUDA on Blackwell, Ada and
Turing has run on its hardware), the selection logic and the self test harness are covered by unit tests and by runs on a machine without a GPU.

## 1. Two numbers per device

`fllm-rs devices` lists every device of every backend that is compiled into the binary. Each entry has two numbers, and only one of them goes
on the command line:

```text
  [1] NVIDIA GeForce RTX 4090 (CUDA, sm_89, 128 SMs)
      Backend:      cuda (native)
      Select:       --device cuda:0
      ...
  [3] AMD Radeon RX 7900 XTX (gfx1100) (native, Vulkan 1.3.275 ...)
      Backend:      vulkan (native)
      Select:       --device vulkan:0
```

* `[n]`: the position in the list of *all* devices. It only identifies a line of that output.
* `--device <backend>:<N>`: `N` counts the devices of *that backend* from 0. `vulkan:1` is the second Vulkan device, `cuda:1` the second CUDA
  device, whatever their `[n]` is. The `Select` line of each device prints the value to use.

A number that the machine does not have is refused before anything is loaded, with the list of valid ones:

```text
--device vulkan:7: this machine has 1 vulkan device(s) (vulkan:0 = Intel(R) Iris(R) Xe Graphics (RPL-P)); N counts the devices of one backend, ...
```

Backend names: `cpu`, `cuda`, `hip` (`rocm`), `metal`, `vulkan`, `dx12`, `sycl` (`xpu`), `npu` (`openvino`), `tpu`, `gpu` (`wgpu`: any wgpu adapter),
and `auto`. `vulkan` and `metal` use the native backend where the build has one and the device is usable, and the wgpu path of the same API
otherwise. To force one of the two: `vulkan-native` / `metal-native` (native only, an error if it cannot be created) and `wgpu-vulkan`,
`wgpu-metal`, `wgpu-dx12` (wgpu only). With both present, `fllm-rs devices` lists the wgpu adapters of that API as `wgpu-vulkan:N` etc., so
the two implementations of one GPU can be compared (`selftest-gpu --device vulkan:0` against `--device wgpu-vulkan:0`).

## 2. `--device auto`

`auto` (the default of `serve`, `run` and `selftest-gpu`) takes the first backend of

```text
CUDA > ROCm (hip) > Metal > Vulkan (native, else wgpu) > DX12 > CPU
```

that has a hardware GPU, and there the best device: a discrete GPU before an integrated one, then the larger memory. NPUs and TPUs are never
chosen automatically (`--device npu` / `--device tpu`), and SYCL only when `FLLM_AUTO_ORDER` names it (`--device sycl`; an Intel GPU is reached through Vulkan by default). The chain is printed by `fllm-rs devices` ("Auto selection") and at server start.

If the chosen device cannot be created, or cannot run the model (a feature the backend lacks), `serve` says why and tries the next device of
the chain, ending on the CPU. An *explicit* device is never replaced silently: if `--device cuda:0` cannot be created that is an error. (`vulkan` and
`vulkan:N` still fall back from the native to the wgpu implementation of the same API when the native one cannot be created, with a message, as before;
`vulkan-native` does not.)

| Variable | Effect |
| --- | --- |
| `FLLM_AUTO_ORDER=cuda,vulkan,cpu` | order of the backends `auto` tries; `cpu` ends the chain (backends after it are ignored) |
| `FLLM_AUTO_IGPU=0` | integrated GPUs other than Apple's are left to an explicit `--device`. Default: they take part. On x86 machines the CPU kernels can be the faster choice than the integrated GPU, and the inventory cannot tell; if `auto` picks an iGPU that is slower than `--device cpu` for you, set this |
| `FLLM_ALLOW_SOFTWARE_GPU=1` | software adapters (lavapipe, WARP) count as GPUs |

## 3. Capabilities in `fllm-rs devices`

Every device shows `FP16`, `BF16`, `Int8 dot` (a hardware integer dot product: dp4a, VNNI, dotprod, DP4A on Intel, `int-dot` of Vulkan), `Matrix cores`
(tensor cores, MFMA, WMMA, XMX, simdgroup_matrix, cooperative matrix), `Unified memory` and `Bandwidth`, plus free and total memory.

```sh
fllm-rs devices            # text
fllm-rs devices --json     # the same as JSON: inventory number, backend, device spec, memory, capabilities
fllm-rs devices --measure  # additionally measure the host memory bandwidth (about a second)
```

`?` means "not known". The values come from the device description the backend gives its devices (architecture name such as `sm_89`,
`gfx1100`, `Apple8`; feature words such as `fp16`, `int-dot`, `coop-matrix`, `XMX`, `unified-memory`; a `~273 GB/s` figure), they are **inferred, not
queried again**; `notes` in the JSON says where a value is a guess. The bandwidth of a GPU is shown only when its backend reports it (the
Metal backend does for known Apple chips); the CPU's can be measured with `--measure`. A backend that wants a value shown puts it into the
description in that form: `devices::device_caps(&DeviceInfo)` reads it from there.

## 4. The self test: `fllm-rs selftest-gpu`

Every backend runs the same checks against the CPU backend and an f64 reference (`rust/engine/src/backend/selftest.rs`). The output is grouped by
operation:

| Group | Checks |
| --- | --- |
| `delta` | `delta_layer` (Gated DeltaNet) |
| `linear` | `quantized_linear` for every weight format, exact and fast kernels, batches, bias, shared activations |
| `embedding` | `embedding_lookup` |
| `norm`, `rope`, `activation` | `rms_norm`; `rope`; `swiglu`, `geglu`, `residual_add` |
| `kv`, `attention` | `kv_cache_write` (f16, q8_0); `attention_paged` (decode, prefill, window, chunks, soft cap, sinks, ALiBi) |
| `sampling` | `argmax_rows` |
| `host` | `host_linear` (weights streamed from host memory) |
| `moe` | `moe_topk`, `moe_linear`, `moe_combine`, gather / scatter rows |

```sh
fllm-rs selftest-gpu                                   # --device auto: the best GPU
fllm-rs selftest-gpu --device vulkan:1 --json out.json
fllm-rs selftest-gpu --device hip --only linear --quants q4_k,q8_0,q4z
fllm-rs selftest-gpu --list                            # groups and weight formats
fllm-rs selftest-gpu --device cpu                      # the CPU backend against itself: checks the test, not a device
```

* `--only GROUPS` limits the groups; `--quants LIST` the weight formats (`q4z` = all Q4Z group sizes, `q4zp` the act-order ones). `--quants` alone
  limits the run to the groups that depend on a format (`linear`, `embedding`, `moe`, `host`). With a filter every group draws its random data
  from a seed of its own, so a filtered run reproduces the data of the same check in a run with the same filter.
* A check ends as `ok`, `FAIL` or **`SKIPPED (unsupported)`**. The last one is a backend answering `UnsupportedOperation` / `UnsupportedQuant`
  for an operation or weight format that it does not list as supported: a missing feature, not a wrong result, and it does not fail the run.
  A backend that *lists* a weight format in its capabilities and then answers "unsupported" fails the check, and so does the CPU reference
  saying "unsupported" (a broken check must not hide as a skip). `--strict` turns skips into failures.
* The exit status is non-zero when a check fails. `--json FILE` writes backend, device, the counts per group and every check (group, operation,
  shape, status, device and CPU error, tolerance, device time, message) next to the keys of the earlier format (`checks`, `failed`, `ok`, `error`).
* `cuda` and `hip` run the exact kernels first and then the fast ones (fp16 / int8 activations) against the looser bound those allow. The groups `sampling`,
  `host` and `moe` use operations of the fast pass: on `cuda` and `hip` they run there, on every other backend in its only pass (and show up as `SKIPPED` where the
  backend does not implement `argmax_rows`, `host_linear`, `moe_*`, ...). The CPU backend against itself, in full, takes about a second:
  `220 checks, 0 failed, 35 skipped (unsupported), 185 ok`.

The CI workflow `conformance` (see [ci.md](ci.md)) runs this on self-hosted runners with the hardware and keeps the JSON files.

## 5. The tuner

`rust/engine/src/autotune.rs` is the scaffold for per-device tuning (tile sizes, work-group shapes, split factors): a backend gives
`Autotuner::best(key, candidates, bench)` a list of candidate configurations (the first is the default) and a function that times one run; the
fastest wins and is stored in `<cache dir>/autotune-<device>-<hash>.json` (`$FLLM_AUTOTUNE_DIR`, else `$XDG_CACHE_HOME/fllm`, else
`~/.cache/fllm`, `%LOCALAPPDATA%\fllm` on Windows). A cache made for another device, driver or engine version is ignored. `FLLM_AUTOTUNE=off`
returns the first candidate without touching anything, `FLLM_AUTOTUNE=retune` measures again. **Nothing calls the tuner yet and nothing was
measured**; the cache logic is covered by unit tests.
