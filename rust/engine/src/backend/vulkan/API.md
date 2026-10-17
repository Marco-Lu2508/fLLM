# Vulkan core API

Level of verification: **L0** (compiles with `--features vulkan`, clippy clean, 47 hardware-free unit tests pass; nothing here has
ever run on a driver). Smoke tests for the first hardware run are in `instance/smoke.rs` (`--ignored`).

The core wraps `ash` 0.38 and knows nothing about models. The integration layer (`INTEGRATION.md`) builds `Backend`/`Stream`, the weight/KV layouts
and the operations on top of it; the shader crates deliver SPIR-V (`shaders::spirv(name)`) that is turned into pipelines here.

## Objects at a glance

| Type | File | What it is | Threads |
|---|---|---|---|
| `VkContext` | `instance.rs` | One device: caps, allocator, pipeline cache, timeline, queue. `Arc<VkContext>` | `Send + Sync` |
| `VkCaps` | `instance/caps.rs` | Plain data about the device (limits, features, extensions, coop-matrix shapes) | `Send + Sync`, `Clone` |
| `Allocator` | `memory.rs` | Block suballocator (256 MiB blocks, free list), dedicated allocations, host-pointer import | `Send + Sync`, all `&self` |
| `Buffer` | `memory.rs` | View `(VkBuffer, offset, size)` of an allocation; `Clone` = another view of the same allocation | `Send + Sync` |
| `PipelineCache` | `pipeline.rs` | Deduplicated `ComputePipeline`s + `VkPipelineCache` on disk | `Send + Sync` |
| `ComputePipeline` | `pipeline.rs` | SPIR-V + layouts; held as `Arc` | `Send + Sync`, immutable |
| `Timeline` | `commands.rs` | The timeline semaphore; `submit` is internal, `wait`/`completed` are public | `Send + Sync` |
| `Recorder` | `commands.rs` | Records and submits compute work (`&mut self`) | `Send`, not `Sync` |
| `Transfer` | `commands/transfer.rs` | Synchronous `upload` / `download` / `zero` (staging, double buffered) | `Send` (own `Recorder`) |
| `VkError` | `instance/error.rs` | One error enum; converts to `String` and to `BackendError` | `Clone` |

Everything holds an `Arc<DeviceCore>` (the logical device), so the device outlives every buffer, pipeline and recorder. Dropping the last
`Arc<VkContext>` waits for the queue first.

## Start-up

```rust
use crate::backend::vulkan::instance::{VkContext, ComponentType};
use crate::backend::vulkan::memory::BufferUsage;
use crate::backend::vulkan::pipeline::{PipelineSpec, SpecConstant};

let ctx = VkContext::new(device_index)?;           // Result<Arc<VkContext>, VkError>; `?` converts to String / BackendError
let caps = ctx.caps();                             // &VkCaps
let coop = caps.best_coop_matrix(ComponentType::F16, ComponentType::F16, ComponentType::F32, ComponentType::F32);

// pipelines: resolve ONCE and keep the Arc (the lookup hashes the SPIR-V)
let spirv = crate::backend::vulkan::shaders::spirv("rms_norm").unwrap();
let constants = [SpecConstant::u32(0, 256)];
let spec = PipelineSpec::new("rms_norm", &spirv, /*storage buffers*/ 3, /*push constant bytes*/ 16).with_spec_constants(&constants);
let rms_norm = ctx.pipelines().get_or_create(&spec)?;          // Arc<ComputePipeline>
// many at once, in parallel: ctx.pipelines().create_many(&specs)
ctx.pipelines().flush_to_disk()?;                               // after warm-up; also done on drop

// memory
let weights = ctx.allocator().alloc_device(bytes, BufferUsage::COMPUTE)?;
let mut transfer = Transfer::new(&ctx)?;
transfer.upload(&weights, 0, &host_bytes)?;                    // synchronous

// one step
let mut rec = ctx.recorder()?;                                  // keep one per stream
rec.bind_pipeline(&rms_norm)?;
rec.set_buffers(&[&x, &w, &out])?;                              // exactly `bindings` buffers
rec.push_words(&[n as u32, eps.to_bits(), 0, 0])?;              // or push_constants(&[u8])
rec.dispatch(rows, 1, 1)?;                                      // automatic barrier against earlier writes
rec.dispatch_independent(...)?;                                 // no barrier before it
let value = rec.submit()?;                                      // one vkQueueSubmit, returns the timeline value
rec.wait(value)?;                                               // before the host reads results
```

`VkContext::new(index)`: `index` counts the *usable* Vulkan devices in driver order, exactly the numbering of `device_info::list`
(entry `i` of `list(first)` has `index == first + i`, the argument is `i`). Software renderers (llvmpipe, lavapipe, SwiftShader, WARP,
device type CPU/Other) are skipped unless `FLLM_ALLOW_SOFTWARE_GPU` is set (not empty, not `0`). Errors: `Loader` (no `libvulkan`, or older
than 1.1), `NoDevice`, `Unsupported` (Vulkan 1.2 / timeline semaphores missing).

## Capabilities (`VkCaps`)

All fields are public. The layer *enables* everything it knows and the device offers, so the flags are what a shader may use.

- Identity: `device_name`, `vendor_id`, `device_id`, `vendor: Vendor`, `device_type`, `api_version` (capped to 1.3), `driver_name`,
  `driver_info`, UUIDs, `describe()` (one line).
- Memory: `vram_bytes`, `unified_memory`, `bandwidth_hint_gbps` (table of known cards, `None` otherwise), `max_buffer_size`,
  `max_storage_buffer_range` (**binding size limit**, often 4 GiB - 1, 128 MiB on some old/software drivers),
  `min_storage_buffer_offset_alignment`, `max_memory_allocation_size`.
- Compute: `max_shared_memory`, `max_workgroup_invocations`, `max_workgroup_size`, `max_workgroup_count`, `max_push_constants` (>= 128).
- Subgroups: `subgroup_size`, `min/max_subgroup_size`, `subgroup_ops` (basic, vote, arithmetic, ballot, shuffle, shuffle_relative, clustered,
  quad - for the compute stage), `subgroup_size_control`, `compute_full_subgroups`, `subgroup_extended_types`.
- Types: `fp16` (shaderFloat16), `int8`, `int16`, `int64`, `storage16`, `storage8`, `scalar_block_layout`.
- `int_dot`: `IntDotCaps { enabled, packed_4x8_signed/unsigned/mixed, vec_8bit_*, acc_sat_packed_4x8_signed }`; `int8_accelerated()` is the
  switch for dp4a-style Q8 kernels.
- `coop_matrix_supported`, `coop_matrix: Vec<CoopMatrixShape { m, n, k, a, b, c, result, saturating, scope }>`; helpers
  `coop_matrix_for(a, b, c, result)` and `best_coop_matrix(...)` (subgroup scope, largest m*n*k). bf16 is reported as `ComponentType::Bf16`
  when the driver lists it, but the layer does not enable `VK_KHR_shader_bfloat16` (not in this ash version).
- Extensions: `push_descriptor`, `external_memory_host` (+ `min_imported_host_pointer_alignment`), `memory_budget`, `maintenance4`,
  `extensions` (all names the device offers), `has_extension(name)`.

Requirements for a device to be used: Vulkan >= 1.2, `timelineSemaphore`, a compute queue. Everything else is optional.

## Memory (`Allocator`, `Buffer`)

```rust
alloc_device(size, usage)        // GPU memory. On unified-memory devices (caps.unified_memory) it is host visible and mapped.
alloc_device_mapped(size, usage) // device-local + host-visible (ReBAR/UMA) if the device has such a type, else staging memory (check Buffer::kind())
alloc_staging(size)              // host, coherent, upload side (not in the small BAR heap when avoidable)
alloc_readback(size)             // host, cached, download side
alloc(kind, size, usage)
unsafe import_host_ptr(ptr, len) // wrap pinned host memory (an mmap of the model file) without copying; VK_EXT_external_memory_host
can_import_host_ptr(), import_alignment(), is_unified(), block_size(), alignment()
free_memory(), total_memory(), stats()      // AllocStats { live_bytes, live_buffers, reserved_bytes, device_reserved_bytes, allocations, blocks }
```

- `BufferUsage`: `STORAGE | TRANSFER_SRC | TRANSFER_DST | UNIFORM | INDIRECT`; `BufferUsage::COMPUTE` = everything a compute backend needs.
  Pools are keyed by `(memory type, usage)`, so use few distinct usages.
- Allocation size must be > 0 and <= `max_buffer_size`. Every allocation is 256-byte aligned (at least `minStorageBufferOffsetAlignment`
  and `nonCoherentAtomSize`), so a whole `Buffer` can always be bound as a storage buffer. Memory is **not cleared**.
- Blocks are at most 256 MiB (less on small heaps); a request larger than a quarter block gets its own memory. The driver allows only a
  limited number of live memory allocations (often 4096, see `caps.max_memory_allocation_count`); for many large tensors allocate one
  arena and cut it with `Buffer::slice` (tensors of 100+ MiB are dedicated allocations).
- `free_memory()` uses `VK_EXT_memory_budget` (usage of all processes) when present, else heap size minus what this allocator reserved.
- `Buffer`: `size()`, `offset()` (offset in the `VkBuffer`), `handle()`, `id()`, `kind()`, `slice(off, len)`, `aligned_view(off, len, align)
  -> (Buffer, skew)`, `mapped_ptr()`, `is_host_visible()`, `write(off, data)`, `read(off, out)`, `flush`, `invalidate`, `descriptor_info()`.
  `write`/`read` are plain copies into/out of the mapping (flush/invalidate for non-coherent memory is done); the caller must not touch a
  range the GPU is using (wait for the timeline value).
- **Binding rules** (checked by `Recorder::set_buffers`, error otherwise): view offset a multiple of
  `min_storage_buffer_offset_alignment`, view size <= `max_storage_buffer_range`. A tensor at an arbitrary file offset: use
  `aligned_view(offset, len, caps.min_storage_buffer_offset_alignment)`; it returns a view starting at the aligned-down offset and the
  `skew` in bytes the data starts at inside that view - pass the skew to the shader (push constant). Cut weight tensors that exceed
  `max_storage_buffer_range` into several bindings or rows ranges.
- `DeviceBuffer` mapping for the integration layer: keep `Arc<Buffer>` (or `Buffer`, it is cheap) in `DeviceBuffer.handle`
  (`Arc::new(buffer) as Arc<dyn Any + Send + Sync>`), `DeviceBuffer.size/offset` as the view, `slice()` of `DeviceBuffer` -> `Buffer::slice`.
- Lifetime: dropping the last clone returns the memory immediately. The `Recorder` keeps every buffer it recorded alive until the GPU
  finished the submission, so dropping a `Buffer` right after `submit()` is safe. Outside the recorder (your own `ash` calls) keep the
  buffer alive until the timeline value is reached.
- Host-pointer import: the `VkBuffer` starts at `ptr` rounded down to `import_alignment()` (often 4096); `Buffer::offset()` is the
  distance. The whole rounded range must stay mapped. On a discrete GPU every read crosses PCIe: use it for unified memory (iGPU, APU) or
  for layers kept on the host. Size is limited by `max_buffer_size`: import big files in chunks.

## Pipelines (`PipelineCache`, `PipelineSpec`)

- `PipelineSpec { name, spirv: &[u32], bindings, push_constant_bytes, spec_constants, required_subgroup_size, require_full_subgroups }`,
  built with `PipelineSpec::new(name, spirv, bindings, push_bytes)` and `.with_spec_constants(..)`, `.with_subgroup_size(n)`,
  `.with_full_subgroups()`. Entry point is always `main`. Interface: set 0, bindings `0..bindings` are storage buffers; one push constant
  block of `push_constant_bytes` (multiple of 4, <= `caps.max_push_constants`). At most `MAX_BINDINGS` = 32 storage buffers.
- Specialization constants are 4 bytes each (`u32/i32/f32/bool`); the key of the pipeline is the SPIR-V hash + bindings + push size + constant
  values + subgroup settings (not the name). `get_or_create` is thread safe and deduplicates; `create_many` compiles in parallel (<= 8 threads).
- Descriptors: push descriptors (`VK_KHR_push_descriptor`) when the device has them and `bindings <= maxPushDescriptors`, otherwise
  per-recorder descriptor pools that are reset together with the command pool. Nothing to do on the caller side.
- Disk cache: `$FLLM_VK_CACHE_DIR`, else `$XDG_CACHE_HOME/fllm`, else `~/.cache/fllm` (`%LOCALAPPDATA%\fllm` on Windows); file
  `vk-pipeline-v1-<vendor>-<device>-<pipelineCacheUUID>.bin` (the UUID changes with the driver version). Container with length and checksum
  and a check of the Vulkan header against the live device; damaged or foreign files are ignored. Written atomically on `flush_to_disk()`
  and on drop. `FLLM_VK_PIPELINE_CACHE=0` disables it.

## Recording (`Recorder`)

One recorder per stream/thread. Typical decode step: many `bind_pipeline` / `set_buffers` / `push_*` / `dispatch*`, then one `submit()`.

- `begin()` is implicit; call it to start a command buffer early. `submit()` ends it, adds a barrier to the host and submits; it returns the
  timeline value (`u64`, increasing; `0` = nothing submitted). `submit()` with nothing recorded returns the previous value.
- Synchronisation: automatic and conservative. A global barrier is inserted before a `dispatch`, `dispatch_indirect`, `copy_buffer`, `fill`
  or `update_buffer` when any of those wrote since the last barrier. `dispatch_independent` skips the barrier *before* it (it must not read
  what the preceding unbarriered commands wrote). `memory_barrier()` forces one. Every command buffer starts with a barrier against all
  earlier submissions, so work submitted by different recorders / `Transfer` objects on the device is ordered in submission order.
- State rules (errors, not undefined behaviour): `set_buffers` needs a bound pipeline and exactly `bindings` buffers; `dispatch` needs
  `set_buffers` after the last `bind_pipeline`; push-constant size must not exceed the pipeline's. Workgroup counts above the device limit are
  rejected; zero counts are a no-op.
- Other commands: `copy_buffer(src, src_off, dst, dst_off, size)`, `fill(dst, off, size, u32)`, `update_buffer(dst, off, data <= 64 KiB)`,
  `dispatch_indirect(args, off)`.
- Reuse: the command pool and descriptor pools are reset at the next `begin()` once everything this recorder submitted has completed;
  otherwise more command buffers are allocated (no blocking). `stats()` -> `RecorderStats { dispatches, barriers, submits, copies,
  descriptor_sets }` to verify "one submit per step".
- `wait(value)` waits on the device timeline (any recorder's value). `Timeline::{completed, is_complete, wait_timeout}` are available via
  `ctx.timeline()`. Waiting for a value that was never submitted is an error (not a hang).

`Transfer` (own recorder, blocking): `upload(dst, off, &[u8])` (direct `memcpy` when `dst` is host visible; else two 32 MiB staging buffers,
the CPU fills one while the GPU copies the other), `download(src, off, &mut [u8])`, `zero(dst, off, size)`.

## Errors

`VkError { Loader, NoDevice, Unsupported, OutOfMemory { requested }, DeviceLost, Vulkan { call, result }, InvalidArgument, Io }` with `Display`.
`VkError -> String` and `VkError -> BackendError` exist (`OutOfMemory` -> `BackendError::OutOfMemory`, loader/device -> `DeviceNotFound`,
bad argument -> `BufferError`, rest -> `DriverError`). `DeviceLost` is final: drop the context and create a new one.

## Environment variables

| Variable | Effect |
|---|---|
| `FLLM_ALLOW_SOFTWARE_GPU` | list/use software Vulkan (lavapipe, llvmpipe, SwiftShader, WARP) |
| `FLLM_VULKAN_VALIDATE=1` | enable `VK_LAYER_KHRONOS_validation` if installed (messages go to stdout); use it in the first hardware run |
| `FLLM_VULKAN_VERBOSE` | print why the device probe found nothing |
| `FLLM_VK_CACHE_DIR` / `FLLM_VK_PIPELINE_CACHE=0` | pipeline cache directory / disable |

## Known limits and untested assumptions

- One queue (a compute family without graphics when available) for compute and transfers; no async transfer queue, no multi-queue overlap.
- Not implemented: sparse memory, buffer device address, descriptor indexing, debug-utils names, multi-GPU, `VK_NV_cooperative_matrix`
  (older NVIDIA drivers without the KHR extension report no coop-matrix shapes), FP8/bf16 shader extensions.
- The memory-type policy (`memory/select.rs`) is tested with synthetic layouts of NVIDIA / AMD / Intel; real drivers may expose types
  that make another choice better. MoltenVK portability is enabled but untried.
- All Vulkan calls follow the specification and the ash 0.38 signatures (checked by the compiler), but none has run: expect driver-specific
  findings in the first hardware round. Run `FLLM_VULKAN_VALIDATE=1 cargo test ... smoke -- --ignored` first.
