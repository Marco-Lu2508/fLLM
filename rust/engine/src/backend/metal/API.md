# Metal core: API for the shader and integration layers

Status: **L0** (see [test levels](../../../../../docs/test-levels.md)) - compiles for `aarch64-apple-darwin` (`--features metal`), clippy-clean, the pure parts
have unit tests on a Linux host. None of it has run on a Mac yet. A self test for the first hardware run
is built in (section 9).

The module `backend::metal` is only compiled under `cfg(all(feature = "metal", target_vendor = "apple"))`.
The four core files form the Metal core; `ops.rs`, `weights.rs`, `shaders.rs` (kernels) and `backend.rs`, `stream.rs`
(integration) build on it.

| File | Contents |
| --- | --- |
| `device_info.rs` (+ `device_info/{platform,config,error}.rs`) | Device list, `MetalDevice`, `MetalDeviceInfo`, `MetalConfig` (environment variables), `MetalError`, chip detection |
| `buffers.rs` (+ `buffers/plan.rs`) | `MetalBuffer`, `BufferRef`, `MemoryManager`, `MappedRegion`/`MappedRegistry` (no-copy GGUF), `AddressTable`, `Arena` |
| `pipelines.rs` (+ `pipelines/{key,source}.rs`) | `PipelineCache`, `Pipeline`, `FunctionConstant`, MSL source preparation |
| `commands.rs` (+ `commands/{hazard,geometry,residency,smoke}.rs`) | `CommandQueue`, `Recorder`, `DispatchBuilder`, `MetalContext`, hazard tracking, self test |

## 1. Getting started

```rust
use crate::backend::metal::commands::{Access, MetalContext};

let ctx = MetalContext::new(index)?;            // device, memory manager, pipeline cache, queue, residency
let mut rec = ctx.recorder();                   // one recorder per stream
```

`MetalContext` has the public fields `device`, `config`, `memory` (`Arc<MemoryManager>`), `pipelines`
(`Arc<PipelineCache>`), `queue`, `mapped` (`Arc<MappedRegistry>`), `residency` (`Option<Arc<ResidencyManager>>`).

**Device index.** `device_info::list(first_index)` reports the devices with global numbers starting at `first_index` and remembers
`first_index`. `device_info::open(n)` and `MetalContext::new(n)` read `n` as a global number if it lies in
`[first_index, first_index + count)`, otherwise as a position among the Metal devices (0 = default device). On
Macs with a single GPU both are the same device. `--device auto` passes on the global number of the *first*
entry with "metal" in its description; if wgpu and native Metal are built at the same time, the
wgpu adapter comes first - the device enumeration in `devices.rs` resolves that.

`MetalDeviceInfo` (from `ctx.device.info()`): name, `chip: Option<AppleChip>` (M1..M5, A series, Pro/Max/Ultra - name
patterns only), `apple_family: Option<u8>` (highest `MTLGPUFamilyAppleN`), `metal3`, `metal4`, `mac_family2`,
`has_unified_memory`, `recommended_max_working_set_bytes`, `max_buffer_length_bytes`, `max_threadgroup_memory_bytes`,
`max_threads_per_threadgroup`, `argument_buffers_tier2`, `simdgroup_matrix`, `simdgroup_reduction`, `bfloat`
(the last three are rules of thumb from Apple's feature-set tables, not measured), `msl_version`, `page_size`.

## 2. Environment variables (`MetalConfig::from_env()`)

| Variable | Effect |
| --- | --- |
| `FLLM_METAL_SERIAL=1` | serial encoder (the driver orders every dispatch); **first measure for wrong results**, rules out barrier errors |
| `FLLM_METAL_CB_DISPATCHES=N` | commit the command buffer after N dispatches (default 160, 0 = only on `flush`) |
| `FLLM_METAL_MEM_LIMIT_MB=N` | memory budget (default `recommendedMaxWorkingSetSize`) |
| `FLLM_METAL_LIB=path` | precompiled `.metallib` instead of the embedded sources |
| `FLLM_METAL_SINGLE_LIB=1` | all sources in *one* library (default: one per file, lazy) |
| `FLLM_METAL_FAST_MATH=0\|1` | compiler fast math (default 1) |
| `FLLM_METAL_NO_NOCOPY=1` | never bind GGUF without a copy |
| `FLLM_METAL_RESIDENCY=0\|1` | use a residency set (macOS 15) (default 1) |
| `FLLM_METAL_DEBUG=1` | labels and debug groups for GPU captures |
| `FLLM_METAL_MSL=3.1` | force the MSL version (default: the newest the OS supports, at most 3.2; `4.0` only this way) |

## 3. Memory (`buffers.rs`)

All buffers are `MTLStorageModeShared` (unified memory: upload = `memcpy`, download = read, no staging).

```rust
let buf: Arc<MetalBuffer> = ctx.memory.alloc(bytes, BufferUsage::Activations)?;   // contents undefined
let zero = ctx.memory.alloc_zeroed(bytes, BufferUsage::KvCache)?;
let w    = ctx.memory.from_bytes(&host, BufferUsage::Weights)?;                    // copy
let r    = BufferRef::whole(buf.clone());                                           // binding unit
let part = r.slice(offset, len)?;
buf.write(off, &bytes)?; buf.read(off, &mut out)?; buf.fill(off, len, 0)?;
```

* `BufferUsage::Weights` is *untracked* (no hazard-tracking cost): weights must not be written by any kernel after
  the upload. `Activations`/`KvCache`/`Staging` are tracked.
* `write`/`read`/`fill` access the memory directly: only after `synchronize()` and before the next dispatch that
  uses the range.
* `MetalBuffer` is `Send + Sync` and fits as `DeviceBuffer.handle` (`Arc<dyn Any + Send + Sync>`):
  `Arc::downcast::<MetalBuffer>(handle.clone())`. `DeviceBuffer { offset, size }` corresponds to `BufferRef { offset, len }`.
* **Alignment.** `Arena` and `AddressTable` align to 256 bytes. Bound offsets should be divisible by 4
  (Apple GPUs), 256 for `constant` arguments on Intel/AMD Macs. Kernels therefore bind weights as
  `device const ...*`, not as `constant`.
* Metal reserves at least one page per `MTLBuffer` (16 KiB on Apple silicon). Cut small intermediate buffers from an `Arena`
  (`Arena::new(&ctx.memory, capacity, usage)`, `alloc(len) -> BufferRef`, `reset()` per step)
  or the existing `BufferPool` instead of allocating them individually.
* Exceeding `maxBufferLength` or the budget yields `MetalError::OutOfMemory { requested, limit }`.

### GGUF without a copy

```rust
unsafe { ctx.register_host_mapping(owner /* Arc<Mmap> or similar */, base_ptr, len, largest_tensor_in_bytes)?; }
// when uploading a weight:
let weight: BufferRef = ctx.memory.buffer_for_host_slice(&ctx.mapped, raw_bytes)?;   // window or copy
```

`register_host_mapping` splits the range into page-aligned, overlapping windows (<= `maxBufferLength`) so that
every object up to `max_item_len` lies entirely in one window (`buffers/plan.rs`, with tests). `buffer_for_host_slice`
finds the slice by its address; if it is not in a registered range (or no-copy is
disabled), it is copied into a new `Weights` buffer. Requirement for the integration layer: the decoder must hold the
GGUF file via `mmap` and hand the same memory to `upload_weight` (if the loader reads into `Vec`s
the copy path is used, without error). The `owner` keeps the mapping alive. The mapping
is also added to the residency set if the OS has one.

### GPU address table ("argument buffer")

`AddressTable::build(&ctx.memory, &[BufferRef])` creates a buffer with the 64-bit GPU addresses (for
`upload_pointer_table` of the MoE experts). In the kernel: `device const ulong *table`, `(device const T *)table[i]`.
At dispatch time `builder.table(index, &table, Access::Read)` binds the table, makes the targets
resident with `useResource` and counts the access as "everything" for the barrier logic. Needs argument-buffers tier 2
(`argument_buffers_tier2`, all Apple GPUs).

## 4. Pipelines (`pipelines.rs`)

```rust
let p = ctx.pipelines.get("rms_norm_f32", &[])?;                                   // Arc<Pipeline>
let q = ctx.pipelines.get("gemv_q4_0", &[FunctionConstant::u32(0, 4), FunctionConstant::bool(1, true)])?;
ctx.pipelines.prewarm_libraries()?;               // compile all libraries in parallel (otherwise lazy on first use)
ctx.pipelines.prewarm(&[("rms_norm_f32", &[]), ...])?;
p.max_threads(); p.simd_width(); p.static_threadgroup_memory();
```

* `get` hashes the key on every call: resolve pipelines **once at start-up** and keep the `Arc` in the op.
* Function constants: `[[function_constant(i)]]` in MSL; `FunctionConstant::{bool,i32,u32,f32}(index, value)`.
  Every set of values is its own pipeline. Wrong types or indices report `MetalError::Pipeline` with the Metal text.

### Layout of the shader sources (binding for kernel authors)

`build.rs` embeds `rust/engine/metal/*.metal` (`common.metal` first, the rest alphabetically).

* Files whose name begins with `common` or `_`, **and files in which no kernel is found** (pure helper files),
  are *shared code*: they are prepended to every other file (in source order).
* Every other file (with at least one kernel) is its own library, compiled lazily. It may therefore only use
  helper code from shared files and its own text. If a kernel needs code from another
  kernel file, move it to a `common_*.metal` (or set `FLLM_METAL_SINGLE_LIB=1`).
* A small scanner finds kernel names: `kernel void NAME(`, `[[kernel]] void NAME(` and
  `[[host_name("NAME")]]` on template instantiations. It does not see macro-generated names (`host_name("a_" #T)`);
  the cache then queries all libraries (slower first access, but works).
* No `#include "..."` of project files (there are none at run time); only `<metal_stdlib>` and the like.
* Every unit starts with `#define` lines (always 0/1 or numbers): `FLLM_METAL`, `FLLM_APPLE_GPU_FAMILY` (0 = no
  Apple GPU), `FLLM_HAS_SIMDGROUP_MATRIX`, `FLLM_HAS_SIMDGROUP_REDUCTION`, `FLLM_HAS_BFLOAT`, `FLLM_UNIFIED_MEMORY`,
  `FLLM_SIMD_WIDTH_HINT`, `FLLM_MAX_THREADGROUP_MEMORY`. `#line` directives keep compiler errors pointing at the real
  file names and lines.
* Language version: the newest the OS supports (macOS 13 -> 3.0, 14 -> 3.1, 15+ -> 3.2); `bfloat` needs 3.1.
* Precompiled (optional, on the Mac): `xcrun -sdk macosx metal -c -I... file.metal -o file.air` for each file
  with the shared files prepended (`cat common.metal file.metal > ...`), then `xcrun -sdk macosx metallib *.air -o
  fllm.metallib` and `FLLM_METAL_LIB=fllm.metallib`. The `FLLM_*` defines are then missing; set them with `-D` if needed.

## 5. Commands (`commands.rs`)

```rust
let mut rec = ctx.recorder();
rec.begin(&pipeline)?
    .read(0, &x)               // BufferRef, Access::Read
    .write(1, &y)              // Access::Write
    .read_write(2, &acc)       // Access::ReadWrite
    .bytes(3, &params)         // #[repr(C)] struct without padding, <= 4096 bytes, setBytes
    .threadgroup_memory(0, 4096)
    .threadgroups([groups, 1, 1], [threads, 1, 1])?;      // or .threads([total,1,1], [per_group,1,1])?
rec.synchronize()?;            // commit and wait for the GPU; CPU access to the buffers is allowed afterwards
```

* **Barriers.** The encoder is `MTLDispatchTypeConcurrent`. Every binding declares how the kernel uses the range;
  before a dispatch the `HazardTracker` inserts a `memoryBarrierWithScope(Buffers)` if it reads something that
  an earlier dispatch of the same window wrote (or writes something that was read or written). Overlaps
  are checked byte-exactly (sub-ranges of the same buffer do not interfere). **Declaring wrongly causes
  races**: a buffer bound as `read` that the kernel writes is an error. If necessary use `rec.barrier()?`.
* `FLLM_METAL_SERIAL=1` switches this off (no tracking, the driver serialises).
* **Command buffers.** After `FLLM_METAL_CB_DISPATCHES` dispatches the buffer is committed and encoding continues in a new
  one (the GPU starts early). `flush()` commits without waiting. `synchronize()` is the only place that waits for the
  GPU (except with more than 40 buffers in flight). Errors of a buffer are returned as `MetalError::Gpu`.
* `copy_buffer(&src, &dst, len)` and `fill_zero(&dst)` are blits (they end the compute encoder, a new one opens when
  needed); they only work on tracked buffers without further fences, because Metal orders those across encoders.
* `Recorder` is `Send` (moves with the stream) but not `Sync`; all methods take `&mut self`.
* Helpers in `commands::geometry`: `elementwise_threads(n, simd, max)`, `ceil_div`, `validate_dispatch` (checked
  internally: group size 0, or above the pipeline or device limit -> `MetalError::InvalidDispatch` instead of an abort in the driver).
* Binding slots `0..=30` (Metal allows 31). Constants passed through `bytes` also occupy a slot.

## 6. Mapping for `backend.rs` / `stream.rs`

| `Backend`/`Stream` | Metal |
| --- | --- |
| `Backend::create_stream` | put `ctx.recorder()` into a `MetalStream` |
| `allocate_buffer(size)` | `ctx.memory.alloc(size, Activations)`; `DeviceBuffer::new(id, size, device_id, arc as Arc<dyn Any + Send + Sync>)` |
| `Stream::upload(dst, src)` | `BufferRef.write` (after `synchronize` if the GPU uses the buffer) |
| `Stream::download(dst, src)` | `synchronize()`, then `BufferRef.read` |
| `Stream::synchronize` | `Recorder::synchronize` |
| `copy_buffer` | `Recorder::copy_buffer` |
| `zero_buffer` | `Recorder::fill_zero` |
| `upload_weight` | `buffer_for_host_slice` (no-copy) or, after repacking, `from_bytes(.., Weights)` |
| `upload_pointer_table` | `AddressTable` |
| `allocate_kv_cache` | `alloc_zeroed(.., KvCache)` (access only through `DeviceKvCache`) |
| `BackendError` | `impl From<MetalError> for BackendError` exists |

## 7. Guarantees and limits

* Metal objects that Apple documents as thread-safe (device, queue, buffers, pipelines, libraries) are
  `Send + Sync`; command buffers and encoders live only in the `Recorder`.
* Every call that creates Metal objects runs inside an `autoreleasepool` (Rust threads have none).
* Not included: textures, indirect command buffers, MTL4 (Metal 4: command allocators, tensor API for M5). The
  newest capabilities (`metal4`, Apple10) are reported but not used.
* Several recorders on the same queue at once are possible; dependencies *between* recorders are governed only by the
  tracking mode of the buffers (activations/KV tracked -> Metal orders across buffers). There are no explicit
  events/fences.

## 8. Unverified assumptions (to check on hardware)

1. `newBufferWithBytesNoCopy` accepts windows from a `PROT_READ` `mmap` and a page end that extends beyond the end of
   the file (the page padding of the mapping).
2. Untracked weight buffers written by the CPU: the first kernel sees the data without extra synchronisation because
   a `commit` follows.
3. `memoryBarrierWithScope(Buffers)` suffices as ordering between compute dispatches in the same encoder (MLX uses it
   the same way).
4. Blits between compute encoders on tracked buffers need no fence.
5. `useResource` on the targets of an address table is enough for the GPU to be allowed to read them.
6. The residency-set API (macOS 15) is detected through `respondsToSelector`; on older systems it is skipped.
7. SIMD-group capabilities are derived from the family (Apple7+/Mac2); `Pipeline::simd_width` is the real value.

## 9. Self test for the first hardware run

```
cargo test --release -p fllm-engine --features metal metal_core_smoke -- --nocapture
```

runs `commands::smoke::run` (the function can also be called directly: `smoke::run_default(0)`): device info, buffers and
budget, function constants (two specialisations, cache hit), dispatch chain with barriers (expects >= 2
barriers in concurrent mode), independent dispatches (expects 0 barriers), threadgroup-memory reduction,
blit copy/fill, address table and no-copy binding of a page-aligned heap block. The kernels are in the
test itself (`SMOKE_MSL`), independent of the backend's shaders. Each line names the part and the finding; if one fails,
first try `FLLM_METAL_SERIAL=1` and `FLLM_METAL_NO_NOCOPY=1` to narrow down the cause.
