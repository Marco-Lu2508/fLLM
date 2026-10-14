//! Hardware smoke tests of the Vulkan core. All are `#[ignore]`d: they need a Vulkan device (any, also a software renderer with
//! `FLLM_ALLOW_SOFTWARE_GPU=1`) and are run by hand in the test round:
//!
//! ```text
//! cargo test --release -p fllm-engine --features vulkan --lib backend::vulkan::instance::smoke -- --ignored --nocapture --test-threads=1
//! ```
//!
//! They exercise context creation, the allocator (device, staging, readback, mapped, import), the transfer helper, pipeline creation with a
//! specialization constant and push constants, descriptor binding, dispatch, barriers, the timeline and the recorder statistics.
//! Level of verification: L0 (never run).

use super::VkContext;
use crate::backend::vulkan::commands::{Recorder, Transfer};
use crate::backend::vulkan::memory::{BufferUsage, MemoryKind};
use crate::backend::vulkan::pipeline::{PipelineSpec, SpecConstant};
use std::sync::Arc;

/// SPIR-V of `b[i] = a[i] + ADD` (binding 0 = `a` read-only, binding 1 = `b` write-only, push constant `n`, local size 64, spec constant
/// 0 = ADD, default 1). GLSL source (compiled with `glslc --target-env=vulkan1.2 -O`):
///
/// ```glsl
/// #version 450
/// layout(local_size_x = 64) in;
/// layout(constant_id = 0) const uint ADD = 1;
/// layout(std430, binding = 0) readonly buffer A { uint a[]; };
/// layout(std430, binding = 1) writeonly buffer B { uint b[]; };
/// layout(push_constant) uniform Push { uint n; } pc;
/// void main() { uint i = gl_GlobalInvocationID.x; if (i < pc.n) { b[i] = a[i] + ADD; } }
/// ```
const ADD_SPIRV: [u32; 272] = [
    0x07230203, 0x00010500, 0x000d000b, 0x00000030, 0x00000000, 0x00020011, 0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e,
    0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0009000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000b,
    0x00000013, 0x00000020, 0x00000025, 0x00060010, 0x00000004, 0x00000011, 0x00000040, 0x00000001, 0x00000001, 0x00040047, 0x0000000b,
    0x0000000b, 0x0000001c, 0x00050048, 0x00000011, 0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x00000011, 0x00000002, 0x00040047,
    0x0000001d, 0x00000006, 0x00000004, 0x00040048, 0x0000001e, 0x00000000, 0x00000019, 0x00050048, 0x0000001e, 0x00000000, 0x00000023,
    0x00000000, 0x00030047, 0x0000001e, 0x00000002, 0x00040047, 0x00000020, 0x00000022, 0x00000000, 0x00040047, 0x00000020, 0x00000021,
    0x00000001, 0x00040047, 0x00000022, 0x00000006, 0x00000004, 0x00040048, 0x00000023, 0x00000000, 0x00000018, 0x00050048, 0x00000023,
    0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x00000023, 0x00000002, 0x00040047, 0x00000025, 0x00000022, 0x00000000, 0x00040047,
    0x00000025, 0x00000021, 0x00000000, 0x00040047, 0x0000002a, 0x00000001, 0x00000000, 0x00040047, 0x0000002f, 0x0000000b, 0x00000019,
    0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000020, 0x00000000, 0x00040017, 0x00000009,
    0x00000006, 0x00000003, 0x00040020, 0x0000000a, 0x00000001, 0x00000009, 0x0004003b, 0x0000000a, 0x0000000b, 0x00000001, 0x0004002b,
    0x00000006, 0x0000000c, 0x00000000, 0x00040020, 0x0000000d, 0x00000001, 0x00000006, 0x0003001e, 0x00000011, 0x00000006, 0x00040020,
    0x00000012, 0x00000009, 0x00000011, 0x0004003b, 0x00000012, 0x00000013, 0x00000009, 0x00040015, 0x00000014, 0x00000020, 0x00000001,
    0x0004002b, 0x00000014, 0x00000015, 0x00000000, 0x00040020, 0x00000016, 0x00000009, 0x00000006, 0x00020014, 0x00000019, 0x0003001d,
    0x0000001d, 0x00000006, 0x0003001e, 0x0000001e, 0x0000001d, 0x00040020, 0x0000001f, 0x0000000c, 0x0000001e, 0x0004003b, 0x0000001f,
    0x00000020, 0x0000000c, 0x0003001d, 0x00000022, 0x00000006, 0x0003001e, 0x00000023, 0x00000022, 0x00040020, 0x00000024, 0x0000000c,
    0x00000023, 0x0004003b, 0x00000024, 0x00000025, 0x0000000c, 0x00040020, 0x00000027, 0x0000000c, 0x00000006, 0x00040032, 0x00000006,
    0x0000002a, 0x00000001, 0x0004002b, 0x00000006, 0x0000002d, 0x00000040, 0x0004002b, 0x00000006, 0x0000002e, 0x00000001, 0x0006002c,
    0x00000009, 0x0000002f, 0x0000002d, 0x0000002e, 0x0000002e, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8,
    0x00000005, 0x00050041, 0x0000000d, 0x0000000e, 0x0000000b, 0x0000000c, 0x0004003d, 0x00000006, 0x0000000f, 0x0000000e, 0x00050041,
    0x00000016, 0x00000017, 0x00000013, 0x00000015, 0x0004003d, 0x00000006, 0x00000018, 0x00000017, 0x000500b0, 0x00000019, 0x0000001a,
    0x0000000f, 0x00000018, 0x000300f7, 0x0000001c, 0x00000000, 0x000400fa, 0x0000001a, 0x0000001b, 0x0000001c, 0x000200f8, 0x0000001b,
    0x00060041, 0x00000027, 0x00000028, 0x00000025, 0x00000015, 0x0000000f, 0x0004003d, 0x00000006, 0x00000029, 0x00000028, 0x00050080,
    0x00000006, 0x0000002b, 0x00000029, 0x0000002a, 0x00060041, 0x00000027, 0x0000002c, 0x00000020, 0x00000015, 0x0000000f, 0x0003003e,
    0x0000002c, 0x0000002b, 0x000200f9, 0x0000001c, 0x000200f8, 0x0000001c, 0x000100fd, 0x00010038,
];

fn context() -> Arc<VkContext> {
    VkContext::new(0).expect("a usable Vulkan device (set FLLM_ALLOW_SOFTWARE_GPU=1 for lavapipe)")
}

fn pattern(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| i.wrapping_mul(2_654_435_761).rotate_left(7)).collect()
}

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect()
}

#[test]
#[ignore = "needs a Vulkan device"]
fn context_reports_capabilities() {
    let ctx = context();
    let caps = ctx.caps();
    eprintln!("{} | {}", caps.device_name, caps.describe());
    eprintln!("coop matrix shapes: {:?}", caps.coop_matrix);
    assert!(caps.api_version >= ash::vk::API_VERSION_1_2);
    assert!(caps.subgroup_size >= 1);
    assert!(caps.max_workgroup_invocations >= 128);
    assert!(caps.vram_bytes > 0);
    assert!(ctx.allocator().free_memory() > 0);
}

#[test]
#[ignore = "needs a Vulkan device"]
fn upload_download_round_trip() {
    let ctx = context();
    let mut transfer = Transfer::new(&ctx).unwrap();
    // 40 MiB: more than one staging chunk when the buffer is not mapped
    let words = pattern(10 << 20);
    let bytes = words_to_bytes(&words);
    let device = ctx.allocator().alloc_device(bytes.len() as u64, BufferUsage::COMPUTE).unwrap();
    transfer.upload(&device, 0, &bytes).unwrap();
    let mut back = vec![0u8; bytes.len()];
    transfer.download(&device, 0, &mut back).unwrap();
    assert_eq!(bytes_to_words(&back), words);

    // sub-range of a slice
    let slice = device.slice(4096, 1024).unwrap();
    let mut part = vec![0u8; 1024];
    transfer.download(&slice, 0, &mut part).unwrap();
    assert_eq!(part, bytes[4096..5120]);

    // zero fill
    transfer.zero(&device, 0, 4096).unwrap();
    let mut head = vec![1u8; 4096];
    transfer.download(&device, 0, &mut head).unwrap();
    assert!(head.iter().all(|&b| b == 0));
}

#[test]
#[ignore = "needs a Vulkan device"]
fn dispatch_with_specialization_and_push_constants() {
    let ctx = context();
    let n = 100_003usize; // not a multiple of the workgroup size
    let input = pattern(n);
    for add in [1u32, 40_000] {
        let constants = [SpecConstant::u32(0, add)];
        let spec = PipelineSpec::new("smoke_add", &ADD_SPIRV, 2, 4).with_spec_constants(&constants);
        let pipeline = ctx.pipelines().get_or_create(&spec).unwrap();
        // the same description gives the same pipeline
        assert!(Arc::ptr_eq(&pipeline, &ctx.pipelines().get_or_create(&spec).unwrap()));

        let a = ctx.allocator().alloc_device(4 * n as u64, BufferUsage::COMPUTE).unwrap();
        let b = ctx.allocator().alloc_device(4 * n as u64, BufferUsage::COMPUTE).unwrap();
        let mut transfer = Transfer::new(&ctx).unwrap();
        transfer.upload(&a, 0, &words_to_bytes(&input)).unwrap();

        let mut rec = Recorder::new(&ctx).unwrap();
        rec.bind_pipeline(&pipeline).unwrap();
        rec.set_buffers(&[&a, &b]).unwrap();
        rec.push_words(&[n as u32]).unwrap();
        rec.dispatch((n as u32).div_ceil(64), 1, 1).unwrap();
        let value = rec.submit().unwrap();
        rec.wait(value).unwrap();
        assert_eq!(rec.stats().submits, 1);
        assert_eq!(rec.stats().dispatches, 1);

        let mut out = vec![0u8; 4 * n];
        transfer.download(&b, 0, &mut out).unwrap();
        let expected: Vec<u32> = input.iter().map(|v| v.wrapping_add(add)).collect();
        assert_eq!(bytes_to_words(&out), expected, "ADD = {add}");
    }
}

#[test]
#[ignore = "needs a Vulkan device"]
fn chained_dispatches_and_recorder_reuse() {
    // c = a + 1 (dispatch 1), d = c + 1 (dependent dispatch: needs the automatic barrier), repeated over several steps with the same
    // recorder (pool reset) and a single submit per step
    let ctx = context();
    let n = 4096usize;
    let input = pattern(n);
    let spec = PipelineSpec::new("smoke_add", &ADD_SPIRV, 2, 4);
    let pipeline = ctx.pipelines().get_or_create(&spec).unwrap();
    let usage = BufferUsage::COMPUTE;
    let a = ctx.allocator().alloc_device(4 * n as u64, usage).unwrap();
    let c = ctx.allocator().alloc_device(4 * n as u64, usage).unwrap();
    let d = ctx.allocator().alloc_device(4 * n as u64, usage).unwrap();
    let mut transfer = Transfer::new(&ctx).unwrap();
    transfer.upload(&a, 0, &words_to_bytes(&input)).unwrap();
    let mut rec = Recorder::new(&ctx).unwrap();
    for step in 0..8 {
        rec.begin().unwrap();
        for (src, dst) in [(&a, &c), (&c, &d)] {
            rec.bind_pipeline(&pipeline).unwrap();
            rec.set_buffers(&[src, dst]).unwrap();
            rec.push_words(&[n as u32]).unwrap();
            rec.dispatch(n as u32 / 64, 1, 1).unwrap();
        }
        let value = rec.submit().unwrap();
        rec.wait(value).unwrap();
        let mut out = vec![0u8; 4 * n];
        transfer.download(&d, 0, &mut out).unwrap();
        let expected: Vec<u32> = input.iter().map(|v| v.wrapping_add(2)).collect();
        assert_eq!(bytes_to_words(&out), expected, "step {step}");
    }
    assert_eq!(rec.stats().submits, 8);
    assert!(rec.stats().barriers >= 8 * 2, "{:?}", rec.stats());
}

#[test]
#[ignore = "needs a Vulkan device"]
fn allocator_reuses_and_accounts_memory() {
    let ctx = context();
    let alloc = ctx.allocator();
    let before = alloc.stats();
    let mut buffers = Vec::new();
    for i in 0..200u64 {
        buffers.push(alloc.alloc_device(1000 + i * 37, BufferUsage::COMPUTE).unwrap());
    }
    let live = alloc.stats();
    assert_eq!(live.live_buffers, before.live_buffers + 200);
    // distinct, aligned, non-overlapping ranges inside shared blocks
    for pair in buffers.windows(2) {
        assert!(pair[0].offset().is_multiple_of(alloc.alignment()));
        if pair[0].handle() == pair[1].handle() {
            assert!(pair[0].offset() + pair[0].size() <= pair[1].offset());
        }
    }
    buffers.truncate(100);
    drop(buffers);
    assert_eq!(alloc.stats().live_buffers, before.live_buffers);
    // a dedicated allocation
    let big = alloc.alloc_device(alloc.block_size(), BufferUsage::COMPUTE).unwrap();
    drop(big);
    // staging / readback / mapped kinds are host visible
    assert!(alloc.alloc_staging(1 << 20).unwrap().is_host_visible());
    assert!(alloc.alloc_readback(1 << 20).unwrap().is_host_visible());
    let mapped = alloc.alloc_device_mapped(4096, BufferUsage::COMPUTE).unwrap();
    assert!(mapped.is_host_visible());
    assert!(matches!(mapped.kind(), MemoryKind::DeviceMapped | MemoryKind::Staging));
    mapped.write(0, &[1, 2, 3, 4]).unwrap();
    let mut four = [0u8; 4];
    mapped.read(0, &mut four).unwrap();
    assert_eq!(four, [1, 2, 3, 4]);
}

#[test]
#[ignore = "needs a Vulkan device with VK_EXT_external_memory_host"]
fn host_pointer_import() {
    let ctx = context();
    let alloc = ctx.allocator();
    if !alloc.can_import_host_ptr() {
        eprintln!("VK_EXT_external_memory_host not available: skipped");
        return;
    }
    let align = alloc.import_alignment() as usize;
    let n = 64 * 1024usize;
    let layout = std::alloc::Layout::from_size_align(4 * n + 2 * align, align).unwrap();
    // SAFETY: a fresh allocation of non-zero size; freed at the end of the test after every GPU use has finished.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    let input = pattern(n);
    // data starts 64 bytes into the allocation: not aligned to the import alignment, not to 256 either
    let data = unsafe { base.add(64) };
    // SAFETY: the allocation holds 4 * n + 2 * align bytes starting at `base`.
    unsafe { std::ptr::copy_nonoverlapping(words_to_bytes(&input).as_ptr(), data, 4 * n) };
    // SAFETY: the range stays allocated and unmodified until `imported` and all recorded uses are dropped (below).
    let imported = unsafe { alloc.import_host_ptr(data, 4 * n) }.unwrap();
    assert_eq!(imported.size(), 4 * n as u64);

    let spec = PipelineSpec::new("smoke_add", &ADD_SPIRV, 2, 4);
    let pipeline = ctx.pipelines().get_or_create(&spec).unwrap();
    let out = alloc.alloc_device(4 * n as u64, BufferUsage::COMPUTE).unwrap();
    let (view, skew) = imported
        .aligned_view(0, 4 * n as u64, ctx.caps().min_storage_buffer_offset_alignment)
        .unwrap();
    assert_eq!(skew % 4, 0);
    // the shader of this smoke test has no skew parameter, so only run it when the data is aligned for a direct binding
    if skew == 0 {
        let mut rec = Recorder::new(&ctx).unwrap();
        rec.bind_pipeline(&pipeline).unwrap();
        rec.set_buffers(&[&view, &out]).unwrap();
        rec.push_words(&[n as u32]).unwrap();
        rec.dispatch((n as u32).div_ceil(64), 1, 1).unwrap();
        rec.submit_and_wait().unwrap();
        let mut transfer = Transfer::new(&ctx).unwrap();
        let mut bytes = vec![0u8; 4 * n];
        transfer.download(&out, 0, &mut bytes).unwrap();
        let expected: Vec<u32> = input.iter().map(|v| v.wrapping_add(1)).collect();
        assert_eq!(bytes_to_words(&bytes), expected);
    } else {
        eprintln!("import skew {skew}: binding needs a shader with an offset parameter, dispatch skipped");
    }
    drop(view);
    drop(imported);
    ctx.wait_idle().unwrap();
    // SAFETY: same layout as the allocation; no GPU use remains.
    unsafe { std::alloc::dealloc(base, layout) };
}

#[test]
#[ignore = "needs a Vulkan device"]
fn pipeline_cache_file_is_written() {
    let dir = std::env::temp_dir().join(format!("fllm-vk-cache-test-{}", std::process::id()));
    std::env::set_var("FLLM_VK_CACHE_DIR", &dir);
    let ctx = context();
    let spec = PipelineSpec::new("smoke_add", &ADD_SPIRV, 2, 4);
    ctx.pipelines().get_or_create(&spec).unwrap();
    let written = ctx.pipelines().flush_to_disk().unwrap();
    eprintln!("cache file written: {written} at {:?}", ctx.pipelines().disk_path());
    if written {
        let path = ctx.pipelines().disk_path().unwrap().to_owned();
        assert!(path.starts_with(&dir));
        assert!(std::fs::metadata(path).unwrap().len() > 24);
    }
    drop(ctx);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn types_are_thread_safe() {
    fn send_sync<T: Send + Sync>() {}
    fn send<T: Send>() {}
    send_sync::<VkContext>();
    send_sync::<crate::backend::vulkan::memory::Buffer>();
    send_sync::<crate::backend::vulkan::memory::Allocator>();
    send_sync::<crate::backend::vulkan::pipeline::PipelineCache>();
    send_sync::<crate::backend::vulkan::pipeline::ComputePipeline>();
    send_sync::<crate::backend::vulkan::commands::Timeline>();
    send::<Recorder>();
    send::<Transfer>();
}
