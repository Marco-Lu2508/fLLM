//! Self-check of the Metal core on real hardware: device, buffers, function constants, dispatch chains with barriers,
//! threadgroup memory, blits, GPU-address tables and no-copy host mappings, against kernels that are compiled from
//! the small source embedded below (independent of the shaders of the backend).
//!
//! Nothing here has run yet. On a Mac, `cargo test --release -p fllm-engine --features metal metal_core_smoke -- --nocapture`
//! runs it and prints one line per check. A failing check names the part of the core that is wrong.

use super::super::buffers::{
    AddressTable, BufferKind, BufferRef, BufferUsage, MappedRegion, MappedRegistry,
};
use super::super::device_info::MetalError;
use super::super::pipelines::{FunctionConstant, PipelineCache};
use super::geometry::elementwise_threads;
use super::{Access, MetalContext};
use std::alloc::Layout;
use std::sync::Arc;

/// Kernels of the self-check. One shared function-constant pair, a chain stage, a gather through a GPU-address table
/// and a threadgroup-memory reduction.
const SMOKE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint ADD [[function_constant(0)]];
constant bool DOUBLE [[function_constant(1)]];

kernel void smoke_iota(device uint *out [[buffer(0)]], constant uint &n [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = i + ADD;
}

kernel void smoke_scale(device const uint *in [[buffer(0)]], device uint *out [[buffer(1)]],
                        constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = DOUBLE ? in[i] * 2u : in[i];
}


kernel void smoke_tg_sum(device const uint *in [[buffer(0)]], device uint *out [[buffer(1)]],
                         threadgroup uint *scratch [[threadgroup(0)]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint group [[threadgroup_position_in_grid]],
                         uint size [[threads_per_threadgroup]]) {
    scratch[tid] = in[group * size + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = size / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[group] = scratch[0];
}
"#;
/// The kernel that follows GPU addresses sits in a unit of its own: if the pointer cast does not compile on some
/// OS release, only the address-table check fails and the others still run.
const SMOKE_TABLE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void smoke_gather(device const ulong *table [[buffer(0)]], device uint *out [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
    device const uint *p = reinterpret_cast<device const uint *>(table[i]);
    out[i] = p[0];
}
"#;

/// Result of one check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    Passed(String),
    Skipped(String),
    Failed(String),
}

#[derive(Clone, Debug)]
pub struct SmokeCheck {
    pub name: &'static str,
    pub outcome: Outcome,
}

/// All checks of one run.
#[derive(Clone, Debug)]
pub struct SmokeReport {
    pub device: String,
    pub checks: Vec<SmokeCheck>,
}

impl SmokeReport {
    /// No check failed (skips do not count as failures).
    pub fn passed(&self) -> bool {
        self.checks
            .iter()
            .all(|c| !matches!(c.outcome, Outcome::Failed(_)))
    }

    /// One line per check.
    pub fn render(&self) -> String {
        let mut text = format!("Metal core self-check on {}\n", self.device);
        for check in &self.checks {
            let (tag, detail) = match &check.outcome {
                Outcome::Passed(d) => ("ok  ", d),
                Outcome::Skipped(d) => ("skip", d),
                Outcome::Failed(d) => ("FAIL", d),
            };
            text.push_str(&format!("  [{tag}] {}: {detail}\n", check.name));
        }
        text
    }
}

/// Create the context of device `index` and run all checks.
pub fn run_default(index: usize) -> Result<SmokeReport, MetalError> {
    Ok(run(&MetalContext::new(index)?))
}

/// Run all checks on `context`. Never panics on a failing check; the failure is in the report.
pub fn run(context: &MetalContext) -> SmokeReport {
    let mut report = SmokeReport {
        device: context.device.info().summary(),
        checks: Vec::new(),
    };
    let pipelines = PipelineCache::with_sources(
        &context.device,
        &context.config,
        &[
            ("smoke.metal", SMOKE_MSL),
            ("smoke_table.metal", SMOKE_TABLE_MSL),
        ],
    );
    let mut record = |name: &'static str, result: Result<Outcome, MetalError>| {
        report.checks.push(SmokeCheck {
            name,
            outcome: result.unwrap_or_else(|e| Outcome::Failed(e.to_string())),
        });
    };
    record("device", check_device(context));
    record("buffers", check_buffers(context));
    record("function-constants", check_constants(context, &pipelines));
    record("barrier-chain", check_chain(context, &pipelines));
    record(
        "independent-dispatches",
        check_independent(context, &pipelines),
    );
    record("threadgroup-memory", check_threadgroup(context, &pipelines));
    record("blit", check_blit(context));
    record("address-table", check_table(context, &pipelines));
    record("no-copy-mapping", check_mapping(context, &pipelines));
    report
}

fn u32s_to_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

// chunks_exact stays: s_chunks needs a newer Rust than the engine's minimum
#[allow(clippy::chunks_exact_to_as_chunks)]
fn read_u32s(buffer: &BufferRef, count: usize) -> Result<Vec<u32>, MetalError> {
    let mut bytes = vec![0u8; count * 4];
    buffer.read(0, &mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn new_buffer(context: &MetalContext, count: usize) -> Result<BufferRef, MetalError> {
    Ok(BufferRef::whole(
        context
            .memory
            .alloc_zeroed(count * 4, BufferUsage::Activations)?,
    ))
}

fn expect_equal(what: &str, got: &[u32], want: impl Fn(usize) -> u32) -> Result<(), MetalError> {
    for (i, value) in got.iter().enumerate() {
        if *value != want(i) {
            return Err(MetalError::Gpu(format!(
                "{what}: element {i} is {value}, expected {}",
                want(i)
            )));
        }
    }
    Ok(())
}

fn check_device(context: &MetalContext) -> Result<Outcome, MetalError> {
    let info = context.device.info();
    if info.max_buffer_length_bytes == 0 || info.max_threads_per_threadgroup[0] == 0 {
        return Err(MetalError::NoDevice(
            "the device reports zero limits".to_owned(),
        ));
    }
    Ok(Outcome::Passed(format!(
        "{}; page {} KiB, max buffer {} MiB, {} threads/group, MSL {}.{}, serial={}",
        info.name,
        info.page_size / 1024,
        info.max_buffer_length_bytes >> 20,
        info.max_threads_per_threadgroup[0],
        info.msl_version.0,
        info.msl_version.1,
        context.config.serial_dispatch
    )))
}

fn check_buffers(context: &MetalContext) -> Result<Outcome, MetalError> {
    let used_before = context.memory.budget().used();
    let pattern: Vec<u8> = (0..4096u32).map(|i| (i * 7 + 3) as u8).collect();
    {
        let buffer = context
            .memory
            .from_bytes(&pattern, BufferUsage::Activations)?;
        let mut back = vec![0u8; pattern.len()];
        buffer.read(0, &mut back)?;
        if back != pattern {
            return Err(MetalError::Buffer(
                "read-back differs from the written bytes".to_owned(),
            ));
        }
        buffer.fill(100, 50, 0xAB)?;
        let mut part = [0u8; 50];
        buffer.read(100, &mut part)?;
        if part.iter().any(|b| *b != 0xAB) {
            return Err(MetalError::Buffer("fill did not set the range".to_owned()));
        }
        if buffer.write(4090, &[0u8; 16]).is_ok() {
            return Err(MetalError::Buffer(
                "out-of-range write was accepted".to_owned(),
            ));
        }
        if context.memory.budget().used() <= used_before {
            return Err(MetalError::Buffer(
                "budget did not grow with an allocation".to_owned(),
            ));
        }
    }
    if context.memory.budget().used() != used_before {
        return Err(MetalError::Buffer(
            "budget did not return the bytes of a dropped buffer".to_owned(),
        ));
    }
    Ok(Outcome::Passed(
        "write, read, fill, bounds, budget".to_owned(),
    ))
}

fn check_constants(
    context: &MetalContext,
    pipelines: &PipelineCache,
) -> Result<Outcome, MetalError> {
    const N: usize = 1000;
    let n = N as u32;
    let plus7 = pipelines.get("smoke_iota", &[FunctionConstant::u32(0, 7)])?;
    let plus9 = pipelines.get("smoke_iota", &[FunctionConstant::u32(0, 9)])?;
    if plus7.id() == plus9.id() {
        return Err(MetalError::Pipeline {
            function: "smoke_iota".to_owned(),
            message: "two constant sets share one pipeline".to_owned(),
        });
    }
    let again = pipelines.get("smoke_iota", &[FunctionConstant::u32(0, 7)])?;
    if again.id() != plus7.id() {
        return Err(MetalError::Pipeline {
            function: "smoke_iota".to_owned(),
            message: "the cache built a pipeline twice".to_owned(),
        });
    }
    let out7 = new_buffer(context, N)?;
    let out9 = new_buffer(context, N)?;
    let mut recorder = context.recorder();
    for (pipeline, out) in [(&plus7, &out7), (&plus9, &out9)] {
        let threads = elementwise_threads(N, pipeline.simd_width(), pipeline.max_threads());
        recorder
            .begin(pipeline)?
            .write(0, out)
            .bytes(1, &n)
            .threads([N, 1, 1], [threads, 1, 1])?;
    }
    recorder.synchronize()?;
    expect_equal("iota+7", &read_u32s(&out7, N)?, |i| i as u32 + 7)?;
    expect_equal("iota+9", &read_u32s(&out9, N)?, |i| i as u32 + 9)?;
    Ok(Outcome::Passed(format!(
        "two specialisations, simd width {}, max threads {}",
        plus7.simd_width(),
        plus7.max_threads()
    )))
}

fn check_chain(context: &MetalContext, pipelines: &PipelineCache) -> Result<Outcome, MetalError> {
    const N: usize = 4096;
    let n = N as u32;
    let iota = pipelines.get("smoke_iota", &[FunctionConstant::u32(0, 5)])?;
    let double = pipelines.get("smoke_scale", &[FunctionConstant::bool(1, true)])?;
    let (a, b, c) = (
        new_buffer(context, N)?,
        new_buffer(context, N)?,
        new_buffer(context, N)?,
    );
    let mut recorder = context.recorder();
    let threads = elementwise_threads(N, iota.simd_width(), iota.max_threads());
    let grid = [N, 1, 1];
    let group = [threads, 1, 1];
    recorder
        .begin(&iota)?
        .write(0, &a)
        .bytes(1, &n)
        .threads(grid, group)?;
    recorder
        .begin(&double)?
        .read(0, &a)
        .write(1, &b)
        .bytes(2, &n)
        .threads(grid, group)?;
    recorder
        .begin(&double)?
        .read(0, &b)
        .write(1, &c)
        .bytes(2, &n)
        .threads(grid, group)?;
    let barriers = recorder.stats().barriers;
    recorder.synchronize()?;
    expect_equal("iota -> x2 -> x2", &read_u32s(&c, N)?, |i| {
        (i as u32 + 5) * 4
    })?;
    if !context.config.serial_dispatch && barriers < 2 {
        return Err(MetalError::Gpu(format!(
            "two dependent dispatches produced only {barriers} barriers"
        )));
    }
    Ok(Outcome::Passed(format!(
        "3 dependent dispatches, {barriers} barriers"
    )))
}

fn check_independent(
    context: &MetalContext,
    pipelines: &PipelineCache,
) -> Result<Outcome, MetalError> {
    const N: usize = 512;
    let n = N as u32;
    let iota = pipelines.get("smoke_iota", &[FunctionConstant::u32(0, 1)])?;
    let (a, b) = (new_buffer(context, N)?, new_buffer(context, N)?);
    let mut recorder = context.recorder();
    let threads = elementwise_threads(N, iota.simd_width(), iota.max_threads());
    for out in [&a, &b] {
        recorder
            .begin(&iota)?
            .write(0, out)
            .bytes(1, &n)
            .threads([N, 1, 1], [threads, 1, 1])?;
    }
    let barriers = recorder.stats().barriers;
    recorder.synchronize()?;
    if barriers != 0 {
        return Err(MetalError::Gpu(format!(
            "two independent dispatches got {barriers} barriers"
        )));
    }
    expect_equal("a", &read_u32s(&a, N)?, |i| i as u32 + 1)?;
    expect_equal("b", &read_u32s(&b, N)?, |i| i as u32 + 1)?;
    Ok(Outcome::Passed(
        "no barrier between independent writes".to_owned(),
    ))
}

fn check_threadgroup(
    context: &MetalContext,
    pipelines: &PipelineCache,
) -> Result<Outcome, MetalError> {
    const GROUP: usize = 64;
    const GROUPS: usize = 6;
    let sum = pipelines.get_plain("smoke_tg_sum")?;
    if sum.max_threads() < GROUP {
        return Ok(Outcome::Skipped(format!(
            "pipeline allows only {} threads per group",
            sum.max_threads()
        )));
    }
    let input = BufferRef::whole(context.memory.from_bytes(
        &u32s_to_bytes(&vec![1u32; GROUP * GROUPS]),
        BufferUsage::Activations,
    )?);
    let out = new_buffer(context, GROUPS)?;
    let mut recorder = context.recorder();
    recorder
        .begin(&sum)?
        .read(0, &input)
        .write(1, &out)
        .threadgroup_memory(0, GROUP * 4)
        .threadgroups([GROUPS, 1, 1], [GROUP, 1, 1])?;
    recorder.synchronize()?;
    expect_equal("group sums", &read_u32s(&out, GROUPS)?, |_| GROUP as u32)?;
    Ok(Outcome::Passed(format!(
        "{GROUPS} groups of {GROUP} threads reduced in threadgroup memory"
    )))
}

fn check_blit(context: &MetalContext) -> Result<Outcome, MetalError> {
    const N: usize = 256;
    let source = BufferRef::whole(context.memory.from_bytes(
        &u32s_to_bytes(&(0..N as u32).collect::<Vec<_>>()),
        BufferUsage::Activations,
    )?);
    let copy = new_buffer(context, N)?;
    let cleared = BufferRef::whole(context.memory.from_bytes(
        &u32s_to_bytes(&vec![0xFFFF_FFFF; N]),
        BufferUsage::Activations,
    )?);
    let mut recorder = context.recorder();
    recorder.copy_buffer(&source, &copy, N * 4)?;
    recorder.fill_zero(&cleared)?;
    recorder.synchronize()?;
    expect_equal("copy", &read_u32s(&copy, N)?, |i| i as u32)?;
    expect_equal("fill_zero", &read_u32s(&cleared, N)?, |_| 0)?;
    Ok(Outcome::Passed("copy_buffer and fill_zero".to_owned()))
}

fn check_table(context: &MetalContext, pipelines: &PipelineCache) -> Result<Outcome, MetalError> {
    const COUNT: usize = 5;
    if !context.device.info().argument_buffers_tier2 {
        return Ok(Outcome::Skipped(
            "argument buffers tier 2 not available".to_owned(),
        ));
    }
    let gather = pipelines.get_plain("smoke_gather")?;
    let mut targets = Vec::new();
    for k in 0..COUNT as u32 {
        targets.push(BufferRef::whole(context.memory.from_bytes(
            &u32s_to_bytes(&[1000 + k, 0, 0, 0]),
            BufferUsage::Weights,
        )?));
    }
    let table = AddressTable::build(&context.memory, &targets)?;
    let out = new_buffer(context, COUNT)?;
    let mut recorder = context.recorder();
    recorder
        .begin(&gather)?
        .table(0, &table, Access::Read)
        .write(1, &out)
        .threads([COUNT, 1, 1], [COUNT, 1, 1])?;
    recorder.synchronize()?;
    expect_equal("gathered values", &read_u32s(&out, COUNT)?, |i| {
        1000 + i as u32
    })?;
    Ok(Outcome::Passed(format!(
        "{COUNT} buffers reached through one binding"
    )))
}

/// Page-aligned heap block standing in for a memory-mapped file.
struct AlignedBlock {
    ptr: *mut u8,
    layout: Layout,
}

// SAFETY: the block is plain memory owned by this value; nothing mutates it after the checks wrapped it.
unsafe impl Send for AlignedBlock {}
unsafe impl Sync for AlignedBlock {}

impl Drop for AlignedBlock {
    fn drop(&mut self) {
        // SAFETY: allocated with exactly this layout in `check_mapping`.
        unsafe { std::alloc::dealloc(self.ptr, self.layout) };
    }
}

#[allow(clippy::chunks_exact_to_as_chunks)] // see read_u32s
fn check_mapping(context: &MetalContext, pipelines: &PipelineCache) -> Result<Outcome, MetalError> {
    const N: usize = 1024;
    let n = N as u32;
    let page = context.device.info().page_size;
    let len = page * 3;
    let layout = Layout::from_size_align(len, page)
        .map_err(|e| MetalError::Buffer(format!("layout: {e}")))?;
    // SAFETY: the layout has a non-zero size.
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        return Err(MetalError::OutOfMemory {
            requested: len,
            limit: 0,
        });
    }
    let block = Arc::new(AlignedBlock { ptr, layout });
    // SAFETY: `ptr` is valid for `len` bytes, exclusively ours until the block is shared below.
    let host = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
    for (i, word) in host.chunks_exact_mut(4).enumerate() {
        word.copy_from_slice(&(i as u32).wrapping_mul(3).to_le_bytes());
    }
    let owner: Arc<dyn std::any::Any + Send + Sync> = block.clone();
    // SAFETY: the memory stays valid and unchanged while `block` (kept by the region and its buffers) lives.
    let region = match unsafe { MappedRegion::new(&context.memory, owner, ptr, len, 8192) } {
        Ok(region) => region,
        Err(MetalError::Unsupported(why)) => return Ok(Outcome::Skipped(why)),
        Err(error) => return Err(error),
    };
    let registry = MappedRegistry::new();
    let region = registry.register(region);
    // a window of N words that starts 256 bytes into the block, like a tensor inside a file
    let tensor = &host[256..256 + N * 4];
    let view = context.memory.buffer_for_host_slice(&registry, tensor)?;
    if view.buffer.kind() != BufferKind::Mapped {
        return Err(MetalError::Buffer(
            "the tensor was copied instead of wrapped".to_owned(),
        ));
    }
    let double = pipelines.get("smoke_scale", &[FunctionConstant::bool(1, true)])?;
    let out = new_buffer(context, N)?;
    let mut recorder = context.recorder();
    let threads = elementwise_threads(N, double.simd_width(), double.max_threads());
    recorder
        .begin(&double)?
        .read(0, &view)
        .write(1, &out)
        .bytes(2, &n)
        .threads([N, 1, 1], [threads, 1, 1])?;
    recorder.synchronize()?;
    expect_equal("doubled host words", &read_u32s(&out, N)?, |i| {
        ((256 / 4 + i) as u32).wrapping_mul(3).wrapping_mul(2)
    })?;
    Ok(Outcome::Passed(format!(
        "{} bytes wrapped in {} window(s), read in place",
        region.covered_bytes(),
        region.buffers().len()
    )))
}

#[cfg(test)]
mod tests {
    use super::super::super::device_info::all_devices;

    #[test]
    fn metal_core_smoke() {
        if all_devices().is_empty() {
            eprintln!("no Metal device on this machine; self-check skipped");
            return;
        }
        let report = super::run_default(0).expect("Metal context");
        eprintln!("{}", report.render());
        assert!(report.passed(), "{}", report.render());
    }
}
