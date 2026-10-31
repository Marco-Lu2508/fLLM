//! `fllm-rs selftest-gpu`: run every backend operation on random data on an accelerator
//! backend and on the CPU backend, and compare both with an f64 reference.
//!
//! Each line reports the largest error of the device and of the CPU backend, each divided by
//! the error scale of its output element, the tolerance in the same units, the device time
//! (enqueue to synchronize) and ok/FAIL. Tolerances are worst-case rounding bounds for f32
//! arithmetic in any summation order (`gamma(n) = n u / (1 - n u)`, `u = 2^-24`) plus the
//! accuracy WGSL guarantees for `exp`/`sqrt`/division and f16 conversions, so they hold on
//! every conforming device; an indexing, stride, scale or layout defect exceeds them by orders
//! of magnitude. Shapes include odd row counts (13), batches of 1, 2 and 5, odd column counts,
//! more than 65535 output rows and KV layers that start on an odd 16-bit word.

use super::cpu::CpuBackend;
use super::*;
use crate::cpu_selftest::{format_name, random_weights, Lcg};
use half::{bf16, f16};
use std::time::Instant;

mod blocksparse;
mod dsa;
mod inkling;
mod hc;
mod v4;
mod kv;
mod mlx;
mod moe;
mod rwkv;
#[cfg(feature = "cuda")]
mod prefill;
#[cfg(feature = "cuda")]
mod smallm;
#[cfg(feature = "cuda")]
mod actdtype;
#[cfg(feature = "cuda")]
mod fp4sm100;
#[cfg(feature = "cuda")]
mod w8a8;
mod fp8blk;
#[cfg(feature = "cuda")]
mod hgemm;
#[cfg(feature = "cuda")]
mod hattn;
#[cfg(feature = "cuda")]
mod hmla;
use kv::layout_of;

pub const HELP: &str = "usage: fllm-rs selftest-gpu [--device auto|vulkan|metal|dx12|hip|sycl|npu|tpu|gpu|cpu|BACKEND:N] [--only GROUPS] [--quants LIST]
                           [--json FILE] [--strict] [--failures-only] [--expect-skips FILE] [--list]
Runs every backend operation on random data on the chosen backend and on the CPU backend, compares both with an f64
reference and prints the largest error of each check relative to its error scale, the tolerance and the device time.
The output is grouped by operation (linear, embedding, norm, rope, activation, kv, attention, moe, delta, sampling,
host; --list says what each covers).
  --device D     auto (default): the best GPU of this machine (CUDA > ROCm > Metal > Vulkan > DX12); a backend name;
                 BACKEND:N for device N of that backend (vulkan:1; the number counts the devices of one backend, see
                 `fllm-rs devices`); gpu = any wgpu adapter; wgpu-vulkan, wgpu-metal, wgpu-dx12 = the wgpu path of that API;
                 cpu = the CPU backend against itself (checks the test, not a device)
  --only G,...   run only these groups
  --quants Q,... run only these weight formats (q4_k,q8_0,iq4_nl,...; q4z = all Q4Z group sizes). Without --only this
                 limits the run to the groups that depend on a format (linear, embedding, moe, host)
  --json FILE    write every check and the summary as JSON
  --strict       count checks the backend reports as unsupported as failures
  --failures-only print only the failing checks (and the summary), not the passing and skipped ones
  --expect-skips FILE  compare the skipped checks (per group and op) with the `expected-skips` block of FILE (the backend's
                 EXPECTED_SKIPS.md) and fail on any difference, in either direction
  --list         print the groups and formats and exit
Verdict per check: ok; FAIL (wrong result, or an error); SKIPPED (unsupported): the backend reports the operation or the
format as unsupported (BackendError::UnsupportedOperation / UnsupportedQuant) and does not list the format in its
capabilities. A skipped check is not a failure; a format that the backend lists as supported but cannot run is one.
Exit status: non-zero when a check fails (or, with --strict, is skipped). Software adapters (lavapipe) need
FLLM_ALLOW_SOFTWARE_GPU=1.";

/// Operation groups of the self test, in the order they run: (name, what the group covers).
pub const GROUPS: &[(&str, &str)] = &[
    ("delta", "delta_layer (Gated DeltaNet): prefill, decode and mixed calls"),
    ("kda", "kda_layer (Kimi Delta Attention): prefill, decode and mixed calls"),
    ("rwkv", "RWKV-6 / RWKV-7: token shift, interpolation, activations and the two WKV recurrences, prompts and decode steps of several sequences"),
    ("arch", "LayerNorm, activations beyond SiLU, clamp, MLA data movement, Mamba-1/2, ShortConv, softplus gate, per-head gate, clamped GLU, hyper-connections, lightning indexer and sparse attention, MiniMax-M3 block-sparse attention, Inkling convolution and relative-position attention"),
    ("linear", "quantized_linear for every weight format (exact and fast kernels, batches, bias, shared activations)"),
    ("embedding", "embedding_lookup for the formats that can hold an embedding table"),
    ("norm", "rms_norm"),
    ("rope", "rope (neox and pair layout, partial rotary dimensions)"),
    ("activation", "swiglu, geglu, residual_add"),
    ("kv", "kv_cache_write for the f16 and q8_0 caches"),
    ("attention", "attention_paged: decode, prefill, window, chunks, softcap, sinks, ALiBi, f16 and q8_0 caches"),
    ("sampling", "argmax_rows (greedy sampling on the device) and sample_rows (penalties, bias, grammar mask, temperature, top-k / min-p / typical-p / top-p, seeded draws, log-probabilities)"),
    ("host", "host_linear (weights streamed from host memory)"),
    ("moe", "moe_topk, moe_linear, moe_combine, gather_rows / scatter_add_rows; device routing variants, expert-major sort, grouped expert GEMM per format, whole layers (moe_route_ex, moe_align, moe_grouped_linear, moe_combine_ex, moe_gate_scale)"),
    ("moebench", "timing of the device MoE operations in microseconds per call at the shapes of real models (only when asked for: --only moebench)"),
    ("fuse", "fused decode kernels against the unfused sequence: residual add + norm (+ activation conversion), rotary + cache write, gated activation + int8 quantization"),
    ("encoder", "bidirectional variable-length encoder attention, ModernBERT local/global window, pooling"),
    ("prefill", "CUDA prompt speed-ups: per-step attention cache, norm and attention FP16 fusions, Marlin M tail (each variant against the plain one)"),
    ("actdtype", "CUDA narrow activations (FLLM_ACT_DTYPE f16/bf16): norm, residual, GLU, rope, KV write, Marlin GEMM, FFN chain, attention against the FP32 pipeline"),
    ("smallm", "CUDA Marlin small-M GEMV (1 to 4 rows) against the Marlin GEMM and the reference; prints us per call of both"),
    ("w8a8", "CUDA FP8 and int8 W8A8 tensor-core GEMMs with activation quantization (FLLM_SELFTEST_BENCH=1 adds timings against the W8A16 paths)"),
    ("fp8blk", "block-wise FP8 weights (FP8B) on CUDA and Vulkan: decode GEMV and prompt GEMM that keep the weights 8-bit, tile shapes, odd shapes, 8-bit memory check; on CUDA also the FP8 tensor-core path against its emulation"),
    ("hattn", "CUDA Hopper and FlashInfer attention: FA3 prefill, sliding window / soft cap / sinks, ragged batches, tensor-core decode, cascade"),
    ("hmla", "CUDA latent-cache MLA (DeepSeek-V2/V3): absorbed up-projections, 576-value cache, tensor cores"),
    ("mlx", "MLX affine group-quantized weights (2 to 8 bits, groups of 32 to 128, f16 and bf16 scales): quantized_linear against the f64 reference; on CUDA every device kernel (GEMV, FP32 tile, tensor cores) on its own"),
    ("hgemm", "CUDA cuBLASLt GEMMs (runtime-loaded) and Hopper CUTLASS WGMMA FP8"),
    ("fp4sm100", "CUDA NVFP4 block-scaled GEMM on datacenter Blackwell (sm_100 tcgen05)")
];

/// Groups whose checks depend on a weight format (`--quants` limits them).
const QUANT_GROUPS: &[&str] = &["linear", "embedding", "moe", "host"];

/// Weight formats the self test knows (GGUF type ids; `format_name` gives the names).
const KINDS: &[u32] = &[
    0, 1, 30, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 29, 20, 23, 39, 40, 0x8000_0010, 0x8000_0011, 0x8000_0012, 0x8000_0014,
    0x8000_0015, 0x8000_0016, 0x8000_0020, 0x8000_0021, 0x8000_0030, 0x8000_0031, fllm_gguf_probe::bnb::NF4_B64, fllm_gguf_probe::bnb::NF4_B128,
    fllm_gguf_probe::bnb::FP4_B64, fllm_gguf_probe::bnb::FP4_B128, fllm_gguf_probe::bnb::INT8_ROW,
];

/// The bitsandbytes weight formats (`fllm_gguf_probe::bnb`).
const BNB: [u32; 5] = [
    fllm_gguf_probe::bnb::NF4_B64,
    fllm_gguf_probe::bnb::NF4_B128,
    fllm_gguf_probe::bnb::FP4_B64,
    fllm_gguf_probe::bnb::FP4_B128,
    fllm_gguf_probe::bnb::INT8_ROW,
];

/// Lower case without separators: `Q4_K` and `q4-k` are the same name.
fn normalize(name: &str) -> String {
    name.chars().filter(char::is_ascii_alphanumeric).map(|c| c.to_ascii_lowercase()).collect()
}

/// The weight formats named in a `--quants` list (`q4_k,q8_0`, a GGUF type number, `q4z` for all Q4Z group sizes, `q4zp` for the act-order ones).
pub fn parse_quants(list: &str) -> Result<Vec<u32>, String> {
    let known = || KINDS.iter().map(|&k| format_name(k)).collect::<Vec<_>>().join(", ");
    let mut kinds = Vec::new();
    for token in list.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let wanted = normalize(token);
        let exact: Vec<u32> = KINDS.iter().copied().filter(|&k| normalize(format_name(k)) == wanted).collect();
        let found = if !exact.is_empty() {
            exact
        } else if let Some(kind) = token.parse::<u32>().ok().filter(|k| KINDS.contains(k)) {
            vec![kind]
        } else {
            // a family: q4z -> q4z32 q4z64 q4z128 (not the act-order q4zp), iq -> every iq format
            KINDS
                .iter()
                .copied()
                .filter(|&k| {
                    let name = normalize(format_name(k));
                    (name.starts_with(&wanted) && !(wanted == "q4z" && name.starts_with("q4zp"))) || (wanted == "bnb" && fllm_gguf_probe::bnb::is_bnb(k))
                })
                .collect()
        };
        if found.is_empty() {
            return Err(format!("unknown weight format '{token}' (known: {})", known()));
        }
        for kind in found {
            if !kinds.contains(&kind) {
                kinds.push(kind);
            }
        }
    }
    if kinds.is_empty() {
        return Err("--quants needs at least one weight format".into());
    }
    Ok(kinds)
}

/// Which checks run: all of them, or the groups and weight formats of `--only` / `--quants`.
#[derive(Clone, Debug, Default)]
pub struct Filter {
    groups: Option<Vec<String>>,
    kinds: Option<Vec<u32>>,
}

impl Filter {
    /// Every check.
    pub fn all() -> Self {
        Self::default()
    }

    /// Parse the values of `--only` and `--quants`.
    pub fn parse(only: Option<&str>, quants: Option<&str>) -> Result<Self, String> {
        let groups = match only {
            None => None,
            Some(list) => {
                let mut groups = Vec::new();
                for name in list.split(',').map(|g| g.trim().to_lowercase()).filter(|g| !g.is_empty()) {
                    if name == "all" {
                        groups.extend(GROUPS.iter().filter(|g| g.0 != "moebench").map(|g| g.0.to_owned()));
                    } else if GROUPS.iter().any(|g| g.0 == name) {
                        groups.push(name);
                    } else {
                        let names: Vec<&str> = GROUPS.iter().map(|g| g.0).collect();
                        return Err(format!("unknown group '{name}' (groups: {})", names.join(", ")));
                    }
                }
                if groups.is_empty() {
                    return Err("--only needs at least one group".into());
                }
                Some(groups)
            }
        };
        let kinds = quants.map(parse_quants).transpose()?;
        Ok(Filter { groups, kinds })
    }

    /// Does any option restrict the run?
    pub fn is_active(&self) -> bool {
        self.groups.is_some() || self.kinds.is_some()
    }

    /// Does the group run? `--quants` alone limits the run to the groups that depend on a weight format.
    pub fn wants_group(&self, group: &str) -> bool {
        match (&self.groups, &self.kinds) {
            (Some(groups), _) => groups.iter().any(|g| g == group),
            (None, Some(_)) => QUANT_GROUPS.contains(&group),
            (None, None) => true,
        }
    }

    /// Was the group named in `--only`? (`moebench` runs only then.)
    pub fn asked_for(&self, group: &str) -> bool {
        self.groups.as_ref().is_some_and(|groups| groups.iter().any(|g| g == group))
    }

    /// Does the weight format run?
    pub fn wants_kind(&self, kind: u32) -> bool {
        self.kinds.as_ref().is_none_or(|kinds| kinds.contains(&kind))
    }

    /// One line for the report: `groups norm,rope; formats q4_k`.
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        if let Some(groups) = &self.groups {
            parts.push(format!("groups {}", groups.join(",")));
        }
        if let Some(kinds) = &self.kinds {
            parts.push(format!("formats {}", kinds.iter().map(|&k| format_name(k)).collect::<Vec<_>>().join(",")));
        }
        if parts.is_empty() {
            "all checks".to_owned()
        } else {
            parts.join("; ")
        }
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "groups": self.groups,
            "formats": self.kinds.as_ref().map(|kinds| kinds.iter().map(|&k| format_name(k)).collect::<Vec<_>>()),
        })
    }
}

/// Unit roundoff of f32.
const U: f64 = 1.0 / 16_777_216.0;

fn gamma(n: usize) -> f64 {
    let nu = n as f64 * U;
    nu / (1.0 - nu)
}

/// What `selftest-gpu` was asked to do.
#[derive(Debug)]
struct Options {
    device: String,
    json: Option<String>,
    filter: Filter,
    strict: bool,
    failures_only: bool,
    expect_skips: Option<String>,
}

#[derive(Debug)]
enum Command {
    Help,
    List,
    Run(Options),
}

fn parse_args(args: &[String]) -> Result<Command, String> {
    let mut device = "auto".to_string();
    let (mut json, mut only, mut quants, mut expect_skips) = (None, None, None, None);
    let (mut strict, mut list, mut failures_only) = (false, false, false);
    let mut i = 0;
    let value = |i: &mut usize, flag: &str| -> Result<String, String> {
        *i += 1;
        args.get(*i).cloned().ok_or_else(|| format!("missing value for {flag}"))
    };
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => return Ok(Command::Help),
            "--device" => device = value(&mut i, "--device")?,
            "--json" => json = Some(value(&mut i, "--json")?),
            "--only" => only = Some(value(&mut i, "--only")?),
            "--quants" => quants = Some(value(&mut i, "--quants")?),
            "--strict" => strict = true,
            "--failures-only" => failures_only = true,
            "--expect-skips" => expect_skips = Some(value(&mut i, "--expect-skips")?),
            "--list" => list = true,
            other => return Err(format!("unknown argument {other}\n{HELP}")),
        }
        i += 1;
    }
    if list {
        return Ok(Command::List);
    }
    Ok(Command::Run(Options { device, json, filter: Filter::parse(only.as_deref(), quants.as_deref())?, strict, failures_only, expect_skips }))
}

/// `--list`: the groups and the weight formats.
fn print_list() {
    println!("groups (--only):");
    for (name, what) in GROUPS {
        println!("  {name:<11} {what}{}", if QUANT_GROUPS.contains(name) { " [depends on weight formats]" } else { "" });
    }
    println!("weight formats (--quants):");
    println!("  {}", KINDS.iter().map(|&k| format_name(k)).collect::<Vec<_>>().join(" "));
}

/// Entry point of `fllm-rs selftest-gpu`.
pub fn run(args: &[String]) -> Result<(), String> {
    let options = match parse_args(args)? {
        Command::Help => {
            println!("{HELP}");
            return Ok(());
        }
        Command::List => {
            print_list();
            return Ok(());
        }
        Command::Run(options) => options,
    };
    clear_records();
    FAILURES_ONLY.store(options.failures_only, std::sync::atomic::Ordering::Relaxed);
    let mut ran = (options.device.clone(), 0usize);
    let mut verdict = run_selected(&options, &mut ran);
    if verdict.is_ok() && options.strict {
        let skipped = records().iter().filter(|r| r.status == Status::Skipped).count();
        if skipped > 0 {
            verdict = Err(format!("{skipped} checks were skipped as unsupported and --strict is set"));
        }
    }
    if verdict.is_ok() {
        if let Some(path) = options.expect_skips.as_deref() {
            verdict = check_expected_skips(path, &records());
        }
    }
    write_json(&options, &ran, &verdict)?;
    verdict
}

/// Skipped checks per (group, op): the report of `skipped_ops` and the quantity `--expect-skips` compares.
fn skipped_ops(records: &[CheckRecord]) -> std::collections::BTreeMap<(String, String), (usize, String)> {
    let mut map = std::collections::BTreeMap::new();
    for r in records.iter().filter(|r| r.status == Status::Skipped) {
        let entry = map.entry((r.group.to_owned(), r.op.clone())).or_insert((0usize, r.message.clone().unwrap_or_default()));
        entry.0 += 1;
    }
    map
}

/// The lines `group | op | count` of the fenced block ```` ```expected-skips ```` of `text` (`#` starts a comment).
fn parse_expected_skips(text: &str) -> Result<std::collections::BTreeMap<(String, String), usize>, String> {
    let mut map = std::collections::BTreeMap::new();
    let mut inside = false;
    for (number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.starts_with("```") {
            inside = line.trim_start_matches('`').trim() == "expected-skips" && !inside;
            continue;
        }
        let line = line.split('#').next().unwrap_or("").trim();
        if !inside || line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split('|').map(str::trim).collect();
        let count = parts.get(2).and_then(|c| c.parse::<usize>().ok());
        let (Some(group), Some(op), Some(count)) = (parts.first(), parts.get(1), count) else {
            return Err(format!("line {}: expected `group | op | count`, found `{line}`", number + 1));
        };
        map.insert(((*group).to_owned(), (*op).to_owned()), count);
    }
    Ok(map)
}

/// `--expect-skips FILE`: the skipped checks of this run must be exactly the expected ones.
fn check_expected_skips(path: &str, records: &[CheckRecord]) -> Result<(), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let expected = parse_expected_skips(&text).map_err(|e| format!("{path}: {e}"))?;
    let actual: std::collections::BTreeMap<(String, String), usize> = skipped_ops(records).into_iter().map(|(k, v)| (k, v.0)).collect();
    let mut problems = Vec::new();
    for (key, &want) in &expected {
        match actual.get(key) {
            None => problems.push(format!("{} / {}: expected {want} skipped, none skipped", key.0, key.1)),
            Some(&have) if have != want => problems.push(format!("{} / {}: expected {want} skipped, found {have}", key.0, key.1)),
            _ => {}
        }
    }
    for (key, have) in &actual {
        if !expected.contains_key(key) {
            problems.push(format!("{} / {}: {have} skipped, not listed in {path}", key.0, key.1));
        }
    }
    if problems.is_empty() {
        println!("skipped checks match {path} ({} entries)", expected.len());
        Ok(())
    } else {
        Err(format!("skipped checks differ from {path}:\n  {}", problems.join("\n  ")))
    }
}

/// Run the checks on the device `--device` names. `auto` tries the devices of the chain in turn (see `devices::device_candidates`).
/// `ran` receives the backend and number of the device the checks ran on.
fn run_selected(options: &Options, ran: &mut (String, usize)) -> Result<(), String> {
    let automatic = crate::devices::is_auto(&options.device);
    let mut last = String::from("no GPU device found");
    for (name, index) in crate::devices::device_candidates(&options.device)? {
        if name == "cpu" {
            if automatic {
                return Err(format!(
                    "{last}; `--device cpu` runs the CPU backend against itself, which checks the test but no device"
                ));
            }
            println!("note: the CPU backend is checked against itself; this tests the self test, not a device");
            *ran = (name, index);
            return run_backend_filtered(&CpuBackend::new(), &options.filter);
        }
        *ran = (name.clone(), index);
        // the outer error: the device could not be created (the next one of an `auto` chain is tried); the inner one: checks failed
        match run_device(&name, index, &options.filter) {
            Ok(verdict) => return verdict,
            Err(error) if automatic => {
                eprintln!("[selftest] --device auto: {} is not usable ({error}); trying the next device", crate::devices::describe_choice(&name, index));
                last = error;
                // like the decoder: where the native vulkan / metal backend cannot be created, the wgpu path of the same API may be
                if matches!(name.as_str(), "vulkan" | "metal") {
                    let wgpu_name = format!("wgpu-{name}");
                    match run_device(&wgpu_name, index, &options.filter) {
                        Ok(verdict) => {
                            *ran = (wgpu_name, index);
                            return verdict;
                        }
                        Err(wgpu_error) => last = format!("{last}; {wgpu_name}: {wgpu_error}"),
                    }
                }
            }
            Err(error) => return Err(error),
        }
    }
    Err(last)
}

/// Create the backend `name`:`index` and run the checks on it. `Err`: the backend could not be created.
fn run_device(name: &str, index: usize, filter: &Filter) -> Result<Result<(), String>, String> {
    // `wgpu-<api>`: the wgpu implementation of that API even where a native backend exists
    let wgpu_only = name.strip_prefix("wgpu-");
    // native backends (vulkan, metal, hip, sycl, npu, tpu): the same checks against the CPU backend
    if wgpu_only.is_none() {
        if let Some(backend) = super::create_native_backend(name, index)? {
            // hip drives the HIP build of the CUDA kernel library: like `cuda`, check the exact FP32 kernels first and the fast ones
            // against the error their approximations allow
            #[cfg(feature = "hip")]
            if matches!(name, "hip" | "rocm") {
                super::cuda::set_exact_math(true);
                let exact = run_backend_pass(&*backend, filter, true);
                super::cuda::set_exact_math(false);
                println!();
                let fast = run_backend_fast_filtered(&*backend, filter);
                return Ok(exact.and(fast));
            }
            // vulkan: like cuda, the exact FP32 shaders against the FP32 error bound, then the fast variants (cooperative-matrix and
            // integer dot-product kernels, fp16 / int8 activations) against the error their approximations allow
            #[cfg(feature = "vulkan")]
            if matches!(name, "vulkan" | "vulkan-native") {
                super::vulkan::backend::set_exact_math(true);
                println!("kernel variant: exact (FP32 shaders only; tolerance = FP32 worst-case bound; FLLM_VULKAN_EXACT is overridden)");
                let exact = run_backend_pass(&*backend, filter, true);
                super::vulkan::backend::set_exact_math(false);
                println!();
                println!("kernel variant: fast (the shaders a normal run picks for this device: cooperative matrices / int8 dot products / f16; tolerance = FAST_TOL)");
                let fast = run_backend_fast_filtered(&*backend, filter);
                return Ok(exact.and(fast));
            }
            // npu: the projections of 64 or more rows go through the NPU (fp16 compute), so the second pass checks those batch sizes
            if name == "npu" {
                // the native AMD XDNA engine computes in bf16: the exact pass runs with its offload paused, the fast pass with it on
                // (set FLLM_XDNA_MIN_ROWS=1 FLLM_XDNA_MIN_ELEMS=1 FLLM_XDNA_MIN_EFF=0 to send the small test shapes to the NPU)
                #[cfg(feature = "npu-xdna-native")]
                super::npu::xdna::native::pause_offload(true);
                let exact = run_backend_pass(&*backend, filter, true);
                #[cfg(feature = "npu-xdna-native")]
                super::npu::xdna::native::pause_offload(false);
                println!();
                let fast = run_backend_fast_filtered(&*backend, filter);
                return Ok(exact.and(fast));
            }
            return Ok(run_backend_filtered(&*backend, filter));
        }
    }
    #[cfg(feature = "cuda")]
    if name == "cuda" {
        // The self test holds every operation to the FP32 error bound, so it runs the exact kernels; the FP16 tensor-core kernels are
        // covered by the parity runs against the CPU.
        let backend = super::cuda::CudaBackend::new(index).map_err(|e| e.to_string())?;
        super::cuda::set_exact_math(true);
        let exact = run_backend_pass(&backend, filter, true);
        // second pass: the fast kernels, held to the error their approximations allow (fp16 / int8 activations)
        super::cuda::set_exact_math(false);
        println!();
        let fast = run_backend_fast_filtered(&backend, filter);
        return Ok(exact.and(fast));
    }
    let wgpu_api = wgpu_only.unwrap_or(name);
    if matches!(wgpu_api, "vulkan" | "metal" | "dx12" | "opengl" | "gl" | "gpu" | "wgpu") {
        #[cfg(feature = "rust-gpu")]
        {
            let backend = super::wgpu_backend::WgpuBackend::new(wgpu_api, index).map_err(|e| e.to_string())?;
            return Ok(run_backend_filtered(&backend, filter));
        }
        #[cfg(not(feature = "rust-gpu"))]
        {
            let _ = (index, filter);
            return Err(format!("the {wgpu_api} backend needs a build with --features rust-gpu (wgpu) or --features vulkan / metal (native)"));
        }
    }
    let _ = (index, filter);
    Err(format!("the {name} backend is not compiled into this binary (compiled: {})", crate::devices::compiled_backends().join(", ")))
}

/// Verdict of one check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Ok,
    /// Wrong result, or an error other than "unsupported".
    Failed,
    /// The backend reports the operation or format as unsupported (and does not list the format as supported).
    Skipped,
}

impl Status {
    fn as_str(self) -> &'static str {
        match self {
            Status::Ok => "ok",
            Status::Failed => "failed",
            Status::Skipped => "skipped",
        }
    }
}

/// One check as it went into the report (`--json`).
#[derive(Clone, Debug)]
pub struct CheckRecord {
    /// `exact` or `fast` (the second pass of `cuda` and `hip`).
    pub pass: &'static str,
    pub group: &'static str,
    pub op: String,
    pub shape: String,
    pub status: Status,
    pub device_error: Option<f64>,
    pub cpu_error: Option<f64>,
    pub tolerance: Option<f64>,
    pub cpu_tolerance: Option<f64>,
    pub device_ms: Option<f64>,
    pub message: Option<String>,
}

/// The records of the checks since the last [`clear_records`] (all passes of one `selftest-gpu` run) and the name of the device.
static RECORDS: std::sync::Mutex<Vec<CheckRecord>> = std::sync::Mutex::new(Vec::new());
static LAST_DEVICE: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

fn records() -> Vec<CheckRecord> {
    RECORDS.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

fn clear_records() {
    RECORDS.lock().unwrap_or_else(|e| e.into_inner()).clear();
}

/// A number for JSON: infinite and NaN errors (a wrong or missing result) become text, since JSON has no such numbers.
fn json_number(value: Option<f64>) -> serde_json::Value {
    match value {
        None => serde_json::Value::Null,
        Some(v) if v.is_finite() => serde_json::json!(v),
        Some(v) if v.is_nan() => serde_json::json!("nan"),
        Some(v) => serde_json::json!(if v > 0.0 { "inf" } else { "-inf" }),
    }
}

/// Checks, ok / failed / skipped per group and in total.
fn tally(records: &[CheckRecord]) -> Vec<(&'static str, [usize; 3])> {
    let mut groups: Vec<(&'static str, [usize; 3])> = GROUPS.iter().map(|g| (g.0, [0; 3])).collect();
    for record in records {
        if let Some((_, counts)) = groups.iter_mut().find(|(name, _)| *name == record.group) {
            counts[match record.status {
                Status::Ok => 0,
                Status::Failed => 1,
                Status::Skipped => 2,
            }] += 1;
        }
    }
    groups.retain(|(_, counts)| counts.iter().sum::<usize>() > 0);
    groups
}

/// `--json FILE`: the backend, device, summary, the counts per group and every check, as a JSON document (`checks` and `failed` keep
/// their meaning of the earlier format: the numbers of checks and failures).
fn write_json(options: &Options, ran: &(String, usize), verdict: &Result<(), String>) -> Result<(), String> {
    let Some(path) = options.json.as_deref() else {
        return Ok(());
    };
    let all = records();
    let count = |status: Status| all.iter().filter(|r| r.status == status).count();
    let groups: serde_json::Map<String, serde_json::Value> = tally(&all)
        .into_iter()
        .map(|(name, c)| (name.to_owned(), serde_json::json!({ "checks": c.iter().sum::<usize>(), "ok": c[0], "failed": c[1], "skipped": c[2] })))
        .collect();
    let results: Vec<serde_json::Value> = all
        .iter()
        .map(|r| {
            serde_json::json!({
                "pass": r.pass,
                "group": r.group,
                "op": r.op,
                "shape": r.shape,
                "status": r.status.as_str(),
                "device_error": json_number(r.device_error),
                "cpu_error": json_number(r.cpu_error),
                "tolerance": json_number(r.tolerance),
                "cpu_tolerance": json_number(r.cpu_tolerance),
                "device_ms": json_number(r.device_ms),
                "message": r.message,
            })
        })
        .collect();
    let mut passes: Vec<&str> = Vec::new();
    for r in &all {
        if !passes.contains(&r.pass) {
            passes.push(r.pass);
        }
    }
    let doc = serde_json::json!({
        "tool": "fllm-rs selftest-gpu",
        "requested_device": options.device,
        "backend": ran.0,
        "device_index": ran.1,
        "device_name": LAST_DEVICE.lock().unwrap_or_else(|e| e.into_inner()).clone(),
        "filter": options.filter.to_json(),
        "strict": options.strict,
        "checks": all.len(),
        "ok_count": count(Status::Ok),
        "failed": count(Status::Failed),
        "skipped": count(Status::Skipped),
        "passes": passes,
        "skipped_ops": skipped_ops(&all)
            .into_iter()
            .map(|((group, op), (count, reason))| serde_json::json!({ "group": group, "op": op, "count": count, "reason": reason }))
            .collect::<Vec<_>>(),
        "groups": groups,
        "ok": verdict.is_ok(),
        "error": verdict.as_ref().err(),
        "results": results,
        "version": env!("CARGO_PKG_VERSION"),
        "commit": env!("FLLM_GIT_COMMIT"),
    });
    std::fs::write(path, serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())? + "\n").map_err(|e| format!("{path}: {e}"))
}

/// Run every check on `device` and the CPU backend. Usable for any [`Backend`].
pub fn run_backend(device: &dyn Backend) -> Result<(), String> {
    run_backend_filtered(device, &Filter::all())
}

/// [`run_backend`] for the checks `filter` selects.
pub fn run_backend_filtered(device: &dyn Backend, filter: &Filter) -> Result<(), String> {
    run_backend_pass(device, filter, false)
}

/// The exact pass. `fast_pass_follows`: a second pass ([`run_backend_fast_filtered`]) runs afterwards (`cuda`, `hip`), and the groups that
/// need the fast kernels (`sampling`, `host`, `moe`) belong to that one; without it they run here, so that every backend meets them.
pub fn run_backend_pass(device: &dyn Backend, filter: &Filter, fast_pass_follows: bool) -> Result<(), String> {
    println!("fllm-rs selftest-gpu: {} against the CPU backend ({})", device.name(), filter.describe());
    let mut t = Tester::new(device, filter, "exact");
    t.extra_groups = !fast_pass_follows;
    t.print_header();
    t.run_all();
    t.finish("GPU backend")
}

/// The second pass of `selftest-gpu --device cuda`: the fast kernels (tensor-core GEMMs with fp16 activations, integer
/// dot products with int8 activations) over the batch sizes that select their different tile shapes.
pub fn run_backend_fast(device: &dyn Backend) -> Result<(), String> {
    run_backend_fast_filtered(device, &Filter::all())
}

/// [`run_backend_fast`] for the checks `filter` selects.
pub fn run_backend_fast_filtered(device: &dyn Backend, filter: &Filter) -> Result<(), String> {
    #[cfg(feature = "cuda")]
    super::cuda::set_marlin_q4k_enabled(true);
    println!("fllm-rs selftest-gpu: {} fast kernels against the CPU backend ({})", device.name(), filter.describe());
    let mut t = Tester::new(device, filter, "fast");
    t.print_header();
    t.run_fast();
    t.finish("fast-kernel")
}

/// Verdict of a check from the answer of the backend. `claimed`: the backend lists the check's weight format as supported, so an
/// "unsupported" answer is a defect and not a skip.
fn classify(outcome: &Result<Outcome, BackendError>, claimed: bool) -> (Status, Option<String>) {
    match outcome {
        Ok(o) => {
            let ok = o.device <= o.tol && o.cpu <= o.cpu_tol;
            (if ok { Status::Ok } else { Status::Failed }, None)
        }
        Err(BackendError::UnsupportedQuant(quant)) if !claimed => (Status::Skipped, Some(format!("format {quant:?}"))),
        Err(BackendError::UnsupportedOperation(what)) if !claimed => (Status::Skipped, Some(what.clone())),
        Err(error @ (BackendError::UnsupportedQuant(_) | BackendError::UnsupportedOperation(_))) => {
            (Status::Failed, Some(format!("{error} (the backend lists this weight format as supported)")))
        }
        Err(error) => (Status::Failed, Some(error.to_string())),
    }
}

/// Errors of one check, each relative to the error scale of its element.
struct Outcome {
    device: f64,
    cpu: f64,
    tol: f64,
    cpu_tol: f64,
    ms: f64,
}

type Op<'a> =
    dyn Fn(&dyn Backend, &mut dyn Stream, &mut f64) -> Result<Vec<f32>, BackendError> + 'a;

struct Tester<'a> {
    device: &'a dyn Backend,
    cpu: CpuBackend,
    rng: Lcg,
    /// Attention checks of the fast FP16 prompt kernels use a looser bound (set around those checks).
    fast: bool,
    filter: &'a Filter,
    /// `exact` or `fast`.
    pass: &'static str,
    /// Group of the checks that run now (see [`GROUPS`]).
    group: &'static str,
    printed_group: &'static str,
    /// Weight format of the check that runs now; `record` uses it to tell "unsupported" from "claimed but unsupported".
    claim: Option<u32>,
    /// Run the `sampling`, `host` and `moe` groups in the exact pass too (set when no fast pass follows, see [`run_backend_pass`]).
    extra_groups: bool,
    records: Vec<CheckRecord>,
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn put(
    backend: &dyn Backend,
    stream: &mut dyn Stream,
    values: &[f32],
) -> Result<DeviceBuffer, BackendError> {
    let mut buf = backend.allocate_buffer(values.len() * 4)?;
    stream.upload(&mut buf, &f32_bytes(values))?;
    Ok(buf)
}

fn get(
    stream: &mut dyn Stream,
    buf: &DeviceBuffer,
    count: usize,
) -> Result<Vec<f32>, BackendError> {
    let mut bytes = vec![0u8; count * 4];
    stream.download(&mut bytes, buf)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

/// Run `f`, wait for the device and store the elapsed milliseconds.
fn timed(
    ms: &mut f64,
    stream: &mut dyn Stream,
    f: impl FnOnce(&mut dyn Stream) -> Result<(), BackendError>,
) -> Result<(), BackendError> {
    let start = Instant::now();
    f(stream)?;
    stream.synchronize()?;
    *ms = start.elapsed().as_secs_f64() * 1e3;
    Ok(())
}

/// Largest `|have - want| / scale` (scale floored at 1e-30). NaN or a length mismatch is
/// infinite.
fn worst(have: &[f32], want: &[f64], scale: &[f64]) -> f64 {
    let mut at = WorstAt { err: 0.0, index: 0, have: 0.0, want: 0.0, scale: 0.0, len: have.len(), want_len: want.len() };
    if have.len() != want.len() {
        at.err = f64::INFINITY;
        return log_worst(at);
    }
    for (i, ((&h, &w), &s)) in have.iter().zip(want).zip(scale).enumerate() {
        let e = (f64::from(h) - w).abs() / s.max(1e-30);
        if e.is_nan() {
            return log_worst(WorstAt { err: f64::INFINITY, index: i, have: h, want: w, scale: s, ..at });
        }
        if e > at.err {
            at = WorstAt { err: e, index: i, have: h, want: w, scale: s, ..at };
        }
    }
    log_worst(at)
}

/// Where the largest error of a `worst` call sits. `record` names it in the message of a check that fails the tolerance, so that no failure is
/// reported without the element, its values and the error scale.
#[derive(Clone, Copy)]
struct WorstAt {
    err: f64,
    index: usize,
    have: f32,
    want: f64,
    scale: f64,
    len: usize,
    want_len: usize,
}

/// Set by `--failures-only`: `Tester::record` prints only the failing checks.
static FAILURES_ONLY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

thread_local! {
    static WORST_LOG: std::cell::RefCell<Vec<WorstAt>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn log_worst(at: WorstAt) -> f64 {
    WORST_LOG.with(|log| {
        let mut log = log.borrow_mut();
        if log.len() >= 32 {
            log.remove(0);
        }
        log.push(at);
    });
    at.err
}

/// For checks that compute their own error: records the element of the largest error so that `describe_failure` can name it. `have` and
/// `want` are the two values compared at `index` of `len`, `scale` the error scale; returns `err`.
fn note_worst(err: f64, index: usize, len: usize, have: f32, want: f64, scale: f64) -> f64 {
    log_worst(WorstAt { err, index, have, want, scale, len, want_len: len })
}

/// The message of a check whose error is beyond its tolerance: for the device and for the CPU backend the error, the tolerance and the worst element.
fn describe_failure(o: &Outcome) -> String {
    let log: Vec<WorstAt> = WORST_LOG.with(|l| l.borrow().clone());
    let mut parts = Vec::new();
    for (label, err, tol) in [("device", o.device, o.tol), ("cpu", o.cpu, o.cpu_tol)] {
        if err <= tol {
            continue;
        }
        let found = log.iter().rev().find(|w| w.err.to_bits() == err.to_bits() || (w.err.is_infinite() && err.is_infinite()));
        parts.push(match found {
            Some(w) if w.len != w.want_len => format!("{label}: {} results for {} expected elements", w.len, w.want_len),
            Some(w) if err.is_infinite() => {
                format!("{label}: NaN at element {} of {}: got {}, expected {:.6e}, error scale {:.3e}", w.index, w.len, w.have, w.want, w.scale)
            }
            Some(w) => format!(
                "{label}: max error {err:.3e} (relative to the error scale) > tolerance {tol:.1e} at element {} of {}: got {}, expected {:.6e}, error scale {:.3e}",
                w.index, w.len, w.have, w.want, w.scale
            ),
            None => format!("{label}: error {err:.3e} > tolerance {tol:.1e} (the position of the worst element was not recorded)"),
        });
    }
    if parts.is_empty() {
        format!("error {:.3e} / {:.3e} beyond tolerance {:.1e} / {:.1e}", o.device, o.cpu, o.tol, o.cpu_tol)
    } else {
        parts.join("; ")
    }
}

/// Offset of the 16-byte trailer (`nvfp4_trailer` in cuda.rs) of an NVFP4 device weight of `rows` x `cols`: behind the nibbles (padded to 256) and the
/// swizzled scales, at the next multiple of 256.
fn nvfp4_trailer_offset(rows: usize, cols: usize) -> usize {
    let nibbles = (rows * cols / 2).div_ceil(256) * 256;
    (nibbles + rows.div_ceil(128) * (cols / 64) * 512).div_ceil(256) * 256
}

/// The E2M1 magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6.
const E2M1_GRID: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Value of an unsigned E4M3 code (the UE4M3 scale of an NVFP4 block).
fn ue4m3_value(code: u8) -> f64 {
    let (e, m) = (i32::from(code >> 3) & 15, f64::from(code & 7));
    if e == 0 { m * 0.125 * 2f64.powi(-6) } else { (1.0 + m * 0.125) * 2f64.powi(e - 7) }
}

/// Nearest UE4M3 code of `x` (ties to the even code), saturating at 448; 0 for x <= 0.
fn ue4m3_code(x: f64) -> u8 {
    if x.is_nan() || x <= 0.0 {
        return 0;
    }
    if x >= 448.0 {
        return 0x7E;
    }
    let mut best = 0u8;
    for code in 1..=0x7Eu8 {
        let (d, db) = ((ue4m3_value(code) - x).abs(), (ue4m3_value(best) - x).abs());
        if d < db || (d == db && code % 2 == 0) {
            best = code;
        }
    }
    best
}

/// Index into `E2M1_GRID` of the magnitude nearest to `a`, ties to the even code (cvt.rn.satfinite.e2m1x2.f32): the midpoints 0.25, 1.25, 2.5 and 5 go down,
/// 0.75, 1.75 and 3.5 go up.
fn e2m1_index(a: f64) -> usize {
    let mut best = 0usize;
    for (i, &g) in E2M1_GRID.iter().enumerate().skip(1) {
        let (d, db) = ((g - a).abs(), (E2M1_GRID[best] - a).abs());
        if d < db || (d == db && i % 2 == 0) {
            best = i;
        }
    }
    best
}

/// A row of activations as the NVFP4 GEMM sees it after vLLM's `scaled_fp4_quant` / ModelOpt: per 16 values a UE4M3 scale
/// `round_e4m3(gs * amax / 6)` and E2M1 values `round_rne(x * gs / scale)`. Returns `q * scale` for every value; the GEMM multiplies the dot product
/// of these with the weights by `weight_scale_2 / gs`. The arithmetic is f32 where the GPU kernel is f32, so the two agree on which grid point a value takes.
fn nvfp4_activation_reference(x: &[f32], gs: f32) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len());
    for block in x.chunks(16) {
        let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let code = ue4m3_code(f64::from(amax * gs * (1.0 / 6.0)));
        let sd = ue4m3_value(code) as f32;
        let inv = if sd > 0.0 { gs / sd } else { 0.0 };
        for &v in block {
            let t = v * inv;
            let q = E2M1_GRID[e2m1_index(f64::from(t.abs()))] * if t < 0.0 { -1.0 } else { 1.0 };
            out.push(q * f64::from(sd));
        }
    }
    out
}

/// The weights of GGUF type 40 (llama.cpp's NVFP4 block: 4 UE4M3 scales + 32 nibble bytes per 64 values, sub-block s byte j holds elements 16 s + j and
/// 16 s + j + 8) as values without the per-tensor factor.
fn nvfp4_weight_values(raw: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    let blocks = cols / 64;
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for b in 0..blocks {
            let block = &raw[(r * blocks + b) * 36..][..36];
            for sub in 0..4 {
                let scale = ue4m3_value(block[sub]);
                for i in 0..16 {
                    let byte = block[4 + sub * 8 + i % 8];
                    let nib = if i < 8 { byte & 15 } else { byte >> 4 };
                    let v = E2M1_GRID[usize::from(nib & 7)] * if nib & 8 != 0 { -1.0 } else { 1.0 };
                    out[r * cols + b * 64 + sub * 16 + i] = (v * scale) as f32;
                }
            }
        }
    }
    out
}

/// Random weights of GGUF type `kind`: uniform [-1, 1] floats, or valid quant blocks with
/// moderate scales (`cpu_selftest::random_weights`).
fn random_kind(rng: &mut Lcg, kind: u32, rows: usize, cols: usize) -> Vec<u8> {
    if let Some(m) = fllm_gguf_probe::mlx_affine(kind) {
        return crate::cpu_mlx::random_weights(m, rows, cols, rng);
    }
    if kind == 12 {
        if let Ok(spec) = std::env::var("FLLM_SELFTEST_TENSOR") {
            if let Some((path, name)) = spec.split_once(':') {
                let m = fllm_gguf_probe::MappedGguf::open(path).expect("open gguf");
                let b = m.tensor_bytes(name).expect("tensor");
                if b.len() == rows * cols / 256 * 144 {
                    return b.to_vec();
                }
            }
        }
    }
    match kind {
        0 => (0..rows * cols)
            .flat_map(|_| rng.unit().to_le_bytes())
            .collect(),
        1 => (0..rows * cols)
            .flat_map(|_| f16::from_f32(rng.unit()).to_le_bytes())
            .collect(),
        30 => (0..rows * cols)
            .flat_map(|_| bf16::from_f32(rng.unit()).to_le_bytes())
            .collect(),
        _ => random_weights(kind, rows, cols, rng),
    }
}

/// One value of a KV pool in the layout documented on [`DeviceKvCache`].
struct PoolView<'a> {
    words: &'a [u16],
    layer_words: usize,
    slots: usize,
    kv: usize,
    group: usize,
    q8: bool,
    /// Set for the formats of `crate::kv_format` other than f16 and q8_0.
    layout: Option<crate::kv_format::KvLayout>,
}

impl PoolView<'_> {
    fn value(&self, layer: usize, slot: usize, c: usize) -> f64 {
        let base = layer * self.layer_words;
        if let Some(layout) = &self.layout {
            return f64::from(layout.value(&self.words[base..base + self.layer_words], slot, c));
        }
        let index = slot * self.kv + c;
        if !self.q8 {
            return f64::from(f16::from_bits(self.words[base + index]).to_f32());
        }
        let q = (self.words[base + index / 2] >> (index % 2 * 8)) as u8 as i8;
        f64::from(q) * self.scale(layer, slot, c / self.group)
    }

    fn scale(&self, layer: usize, slot: usize, g: usize) -> f64 {
        let at = layer * self.layer_words
            + (self.slots * self.kv).div_ceil(2)
            + slot * (self.kv / self.group)
            + g;
        f64::from(f16::from_bits(self.words[at]).to_f32())
    }

    /// Whether every word of `layer` that belongs to no slot in `written` is zero.
    fn untouched_is_zero(&self, layer: usize, written: &[usize]) -> bool {
        if self.layout.is_some() {
            return true;
        }
        let base = layer * self.layer_words;
        let words = &self.words[base..base + self.layer_words];
        let owner_q = |byte: usize| written.contains(&(byte / self.kv));
        if !self.q8 {
            return words.iter().enumerate().all(|(i, &w)| w == 0 || owner_q(i));
        }
        let q_words = (self.slots * self.kv).div_ceil(2);
        let groups = self.kv / self.group;
        words.iter().enumerate().all(|(i, &w)| {
            if i < q_words {
                let low_ok = (w & 0xff) == 0 || owner_q(2 * i);
                let high_ok = (w >> 8) == 0 || owner_q(2 * i + 1);
                low_ok && high_ok
            } else {
                w == 0 || written.contains(&((i - q_words) / groups))
            }
        })
    }
}

fn words_of(values: &[f32]) -> Vec<u16> {
    values.iter().map(|&v| v as u16).collect()
}

/// Sequences with random block tables, flattened into rows of a paged-attention batch.
struct Batch {
    lengths: Vec<i64>,
    tables: Vec<i64>,
    /// (slot, sequence, position) of every cached position.
    cached: Vec<(usize, usize, usize)>,
}

impl<'a> Tester<'a> {
    const SEED: u64 = 0x5eed_f00d;

    fn new(device: &'a dyn Backend, filter: &'a Filter, pass: &'static str) -> Self {
        *LAST_DEVICE.lock().unwrap_or_else(|e| e.into_inner()) = device.name().to_owned();
        Tester {
            device,
            cpu: CpuBackend::new(),
            rng: Lcg(Self::SEED),
            fast: false,
            filter,
            pass,
            group: "",
            printed_group: "",
            claim: None,
            extra_groups: false,
            records: Vec::new(),
        }
    }

    fn print_header(&self) {
        println!(
            "{:<18} {:<40} {:>9} {:>9} {:>9} {:>10}",
            "op", "shape", "device", "cpu", "tol", "device ms"
        );
    }

    /// Start of a group of checks: `false` when the filter leaves it out. With a filter active every group draws its random data from a
    /// seed of its own, so that `--only` reproduces the data a check had in the full run of the same filter; without a filter the
    /// data stream is the one of the single seed (as it always was).
    fn enter(&mut self, group: &'static str) -> bool {
        if !self.filter.wants_group(group) {
            return false;
        }
        self.group = group;
        if self.filter.is_active() {
            let salt = group.bytes().fold(0u64, |h, b| h.wrapping_mul(131).wrapping_add(u64::from(b)));
            self.rng = Lcg(Self::SEED ^ salt.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        }
        true
    }

    /// The members of `list` the filter lets through.
    fn kinds(&self, list: &[u32]) -> Vec<u32> {
        list.iter().copied().filter(|&k| self.filter.wants_kind(k)).collect()
    }

    /// Record the verdict of a check and print its line. An "unsupported" answer is a skip (`SKIPPED (unsupported)`), unless the backend
    /// lists the check's weight format as supported; any other error and a result beyond the tolerance is a failure.
    fn record(&mut self, op: &str, shape: &str, outcome: Result<Outcome, BackendError>) {
        let claimed = self
            .claim
            .take()
            .is_some_and(|kind| self.device.capabilities().supports_quant(QuantType::from_gguf_type_id(kind)));
        // `--device cpu` checks the test with the CPU backend as the device: its kernels are held to the CPU tolerance, which allows for
        // what the tiled CPU kernels do on purpose (Q8 query packing, polynomial exp)
        let outcome = outcome.map(|mut o| {
            // the hybrid NPU backend runs everything but the large prompt projections on the CPU backend
            if matches!(self.device.name(), "CPU" | "npu") {
                o.tol = o.tol.max(o.cpu_tol);
            }
            o
        });
        let (status, mut message) = classify(&outcome, claimed);
        if status == Status::Failed && message.is_none() {
            if let Ok(o) = &outcome {
                message = Some(describe_failure(o));
            }
        }
        // A lost device (the driver killed a job that ran too long, often while the GPU is shared) stays lost: every later check fails with the same error
        // without having run. They stay failures (nothing was verified), but the text says so, and the first one names the check that took the device down.
        if status == Status::Failed && message.as_deref().is_some_and(|m| m.contains("device lost")) {
            static LOST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if LOST.swap(true, std::sync::atomic::Ordering::Relaxed) {
                message = Some("not run: the device was lost earlier in this run (see the first \"device lost\" failure)".to_owned());
            } else {
                message = message.map(|m| format!("{m}; every later check fails with \"not run\" because the device stays lost"));
            }
        }
        WORST_LOG.with(|l| l.borrow_mut().clear());
        let quiet = status != Status::Failed && FAILURES_ONLY.load(std::sync::atomic::Ordering::Relaxed);
        if !quiet && self.printed_group != self.group {
            println!("== {} ==", self.group);
            self.printed_group = self.group;
        }
        let text = message.clone().unwrap_or_default();
        let measured = outcome.as_ref().ok();
        match (measured, status) {
            _ if quiet => {}
            (Some(o), status) => println!(
                "{op:<18} {shape:<40} {:>9.2e} {:>9.2e} {:>9.2e} {:>10.3} {}",
                o.device,
                o.cpu,
                o.tol,
                o.ms,
                if status == Status::Ok { "ok".to_owned() } else { format!("FAIL: {text}") }
            ),
            (None, Status::Skipped) => println!("{op:<18} {shape:<40} SKIPPED (unsupported): {text}"),
            (None, _) => println!("{op:<18} {shape:<40} FAIL: {text}"),
        }
        self.records.push(CheckRecord {
            pass: self.pass,
            group: self.group,
            op: op.to_owned(),
            shape: shape.to_owned(),
            status,
            device_error: measured.map(|o| o.device),
            cpu_error: measured.map(|o| o.cpu),
            tolerance: measured.map(|o| o.tol),
            cpu_tolerance: measured.map(|o| o.cpu_tol),
            device_ms: measured.map(|o| o.ms),
            message,
        });
    }

    /// A check the backend cannot take at all (reported as `SKIPPED (unsupported)`).
    fn record_skip(&mut self, op: &str, shape: &str, reason: &str) {
        self.record(op, shape, Err(BackendError::UnsupportedOperation(reason.to_owned())));
    }

    /// Print the counts per group and in total, hand the records to the report and return the verdict.
    fn finish(self, label: &str) -> Result<(), String> {
        let count = |status: Status| self.records.iter().filter(|r| r.status == status).count();
        let (ok, failed, skipped) = (count(Status::Ok), count(Status::Failed), count(Status::Skipped));
        let groups = tally(&self.records);
        if groups.len() > 1 {
            println!("{:<12} {:>7} {:>7} {:>7} {:>8}", "group", "checks", "ok", "failed", "skipped");
            for (name, c) in &groups {
                println!("{name:<12} {:>7} {:>7} {:>7} {:>8}", c.iter().sum::<usize>(), c[0], c[1], c[2]);
            }
        }
        println!("{} checks, {failed} failed, {skipped} skipped (unsupported), {ok} ok", self.records.len());
        RECORDS.lock().unwrap_or_else(|e| e.into_inner()).extend(self.records);
        if failed > 0 {
            return Err(format!("{failed} {label} checks failed"));
        }
        Ok(())
    }

    /// Run `op` on the device and on the CPU backend; returns both outputs and the device
    /// time.
    fn both(&self, op: &Op<'_>) -> Result<(Vec<f32>, Vec<f32>, f64), BackendError> {
        let mut stream = self.device.create_stream()?;
        let mut ms = 0.0;
        let device = op(self.device, stream.as_mut(), &mut ms)?;
        let mut cpu_stream = self.cpu.create_stream()?;
        let mut cpu_ms = 0.0;
        // the CPU backend is the reference: if it says "unsupported" the check itself is broken, which must not look like a skip
        let cpu = op(&self.cpu, cpu_stream.as_mut(), &mut cpu_ms).map_err(|e| match e {
            BackendError::UnsupportedQuant(_) | BackendError::UnsupportedOperation(_) => {
                BackendError::DriverError(format!("the CPU reference reports: {e}"))
            }
            other => other,
        })?;
        Ok((device, cpu, ms))
    }

    /// Largest error the fast kernels may have relative to the error scale: fp16 activations are good to 5e-4 per
    /// term, int8 activations to about 1e-3 on random data (llama.cpp makes the same trade): 4e-3 catches real errors.
    const FAST_TOL: f64 = 4e-3;

    fn run_fast(&mut self) {
        if self.enter("linear") {
            self.fast_linear_cases();
        }
        if self.enter("mlx") {
            self.mlx_cases(true);
        }
        if self.enter("sampling") {
            self.sampling_cases();
        }
        if self.enter("host") {
            self.host_cases();
        }
        if self.enter("moe") {
            self.moe_cases();
        }
        if self.filter.asked_for("moebench") && self.enter("moebench") {
            self.moe_bench_cases();
        }
        if self.enter("fuse") {
            self.fuse_cases();
        }
        #[cfg(feature = "cuda")]
        if self.enter("prefill") {
            self.prefill_cases();
        }
        #[cfg(feature = "cuda")]
        if self.enter("smallm") {
            self.smallm_cases();
        }
        #[cfg(feature = "cuda")]
        if self.enter("actdtype") {
            self.actdtype_cases();
        }
        #[cfg(feature = "cuda")]
        if self.enter("w8a8") {
            self.w8a8_cases();
        }
        if self.enter("fp8blk") {
            self.fp8blk_cases();
        }
        #[cfg(feature = "cuda")]
        if self.enter("hattn") {
            self.hattn_cases();
        }
        #[cfg(feature = "cuda")]
        if self.enter("hmla") {
            self.hmla_cases();
        }
        #[cfg(feature = "cuda")]
        if self.enter("hgemm") {
            self.hgemm_cases();
        }
        #[cfg(feature = "cuda")]
        if self.enter("fp4sm100") {
            self.fp4sm100_cases();
        }
        // last: the random data of the cases above does not depend on these
        if self.enter("attention") {
            for kind in [1, 2] {
                for case in long_prompt_cases(kind) {
                    let label = format!("attention {}", if case.kind == 1 { "f16" } else { "q8" });
                    let shape = case.describe();
                    let r = self.attention(&case);
                    self.record(&label, &shape, r);
                }
            }
        }
    }

    /// The fast linear kernels per format, the Marlin shapes and the shared-activation cases.
    fn fast_linear_cases(&mut self) {
        // widths that are multiples of 256 take the aligned device layouts; batches select the single-token,
        // integer dot-product, small-tile and (auto-tuned) large-tile kernels
        for kind in self.kinds(&[2u32, 3, 6, 7, 8, 12, 13, 14, 20, 23, 39, 10, 11, 16, 17, 18, 19, 21, 22, 29]) {
            for batch in [1usize, 2, 3, 4, 5, 8, 16, 40, 129, 356] {
                let with_bias = batch == 3 || batch == 129;
                let r = self.linear(kind, 259, 1024, batch, with_bias).map(|mut o| {
                    o.tol = o.tol.max(Self::FAST_TOL);
                    o
                });
                self.record(
                    &format!("fast linear {}", format_name(kind)),
                    &format!("259x1024 batch {batch}{}", if with_bias { " +bias" } else { "" }),
                    r,
                );
            }
        }
        // prompt-sized batches over wide, odd-block-count rows (the tiled matrix-matrix kernel: tile edges in both directions, many K steps)
        for kind in self.kinds(&[2u32, 8, 12, 14, 10, 11]) {
            let block = if matches!(kind, 10..=14) { 256 } else { 32 };
            for (rows, batch) in [(130usize, 65usize), (66, 505)] {
                let cols = 9 * block;
                let r = self.linear(kind, rows, cols, batch, batch == 505).map(|mut o| {
                    o.tol = o.tol.max(Self::FAST_TOL);
                    o
                });
                self.record(
                    &format!("fast linear {}", format_name(kind)),
                    &format!("{rows}x{cols} batch {batch}{}", if batch == 505 { " +bias" } else { "" }),
                    r,
                );
            }
        }
        // bitsandbytes NF4 / FP4 / int8 (the packed blocks stay packed on the device): odd block counts, a row of many blocks, every batch size
        for kind in self.kinds(&BNB) {
            let block = fllm_gguf_probe::bnb::layout(kind).map_or(64, |l| l.block);
            for cols in [5 * block, 16 * block] {
                for batch in [1usize, 2, 3, 4, 5, 8, 16, 40, 129, 356] {
                    let with_bias = batch == 3 || batch == 129;
                    let r = self.linear(kind, 259, cols, batch, with_bias).map(|mut o| {
                        o.tol = o.tol.max(Self::FAST_TOL);
                        o
                    });
                    self.record(
                        &format!("fast linear {}", format_name(kind)),
                        &format!("259x{cols} batch {batch}{}", if with_bias { " +bias" } else { "" }),
                        r,
                    );
                }
            }
            // prompt-sized batches over wide rows: tile edges in both directions, many K steps
            for (rows, batch) in [(130usize, 65usize), (66, 505)] {
                let cols = 9 * block;
                let r = self.linear(kind, rows, cols, batch, batch == 505).map(|mut o| {
                    o.tol = o.tol.max(Self::FAST_TOL);
                    o
                });
                self.record(
                    &format!("fast linear {}", format_name(kind)),
                    &format!("{rows}x{cols} batch {batch}{}", if batch == 505 { " +bias" } else { "" }),
                    r,
                );
            }
        }
        // 4-bit AWQ / GPTQ weights (Marlin GEMM): rows and columns multiples of 128, every batch size the decoder produces
        for kind in self.kinds(&[0x8000_0010u32, 0x8000_0011, 0x8000_0012]) {
            for batch in [1usize, 2, 3, 4, 5, 8, 16, 17, 40, 129, 356] {
                let with_bias = batch == 3 || batch == 129;
                let r = self.linear(kind, 384, 1024, batch, with_bias).map(|mut o| {
                    o.tol = o.tol.max(Self::FAST_TOL);
                    o
                });
                self.record(
                    &format!("fast linear {}", format_name(kind)),
                    &format!("384x1024 batch {batch}{}", if with_bias { " +bias" } else { "" }),
                    r,
                );
            }
        }
        // 4-bit GPTQ weights with act-order (Q4ZP): rows and columns multiples of 128
        for kind in self.kinds(&[0x8000_0014u32, 0x8000_0015, 0x8000_0016]) {
            for batch in [1usize, 2, 3, 4, 5, 8, 16, 17, 40, 129, 356] {
                let with_bias = batch == 3 || batch == 129;
                let r = self.linear(kind, 384, 1024, batch, with_bias).map(|mut o| {
                    o.tol = o.tol.max(Self::FAST_TOL);
                    o
                });
                self.record(
                    &format!("fast linear {}", format_name(kind)),
                    &format!("384x1024 batch {batch}{}", if with_bias { " +bias" } else { "" }),
                    r,
                );
            }
        }
        // packed ExLlamaV2 / ExLlamaV3 matrices (native kernels; backends without them expand at upload and still have to agree): EXL2 widths
        // are multiples of 32 (33 groups here: an odd count, mixed widths, codes straddling words), EXL3 rows and columns multiples of 128;
        // batches above 128 take several passes of the EXL3 kernels
        for kind in self.kinds(&[0x8000_0030u32, 0x8000_0031]) {
            let shapes: &[(usize, usize)] = if kind == 0x8000_0030 { &[(131, 1056), (40, 96)] } else { &[(384, 1024), (128, 384)] };
            for &(rows, cols) in shapes {
                for batch in [1usize, 2, 3, 4, 5, 8, 16, 17, 40, 129, 356] {
                    let with_bias = batch == 3 || batch == 129;
                    let r = self.linear(kind, rows, cols, batch, with_bias).map(|mut o| {
                        o.tol = o.tol.max(Self::FAST_TOL);
                        o
                    });
                    self.record(
                        &format!("fast linear {}", format_name(kind)),
                        &format!("{rows}x{cols} batch {batch}{}", if with_bias { " +bias" } else { "" }),
                        r,
                    );
                }
            }
        }
        // 8-bit FP8 weights (Marlin GEMM): rows and columns multiples of 128
        for kind in self.kinds(&[0x8000_0020u32]) {
            for batch in [1usize, 2, 3, 4, 5, 8, 16, 17, 40, 129, 356] {
                let with_bias = batch == 3 || batch == 129;
                let r = self.linear(kind, 384, 1024, batch, with_bias).map(|mut o| {
                    o.tol = o.tol.max(Self::FAST_TOL);
                    o
                });
                self.record(
                    &format!("fast linear {}", format_name(kind)),
                    &format!("384x1024 batch {batch}{}", if with_bias { " +bias" } else { "" }),
                    r,
                );
            }
        }
        // block-wise FP8 (FP8B): the weights stay 8-bit on the device, the kernels apply the tile scales; the CPU kernel (`cpu_fp8`) is the reference of
        // the CPU backend. Decode batches (1 .. 16 tokens), the prompt kernel (more than 16), widths that are not a multiple of 16 (no tensor-core
        // tile) or of the 128-wide tile, rows that end inside a tile row. The FP8 tensor-core path (sm_89 and newer; it rounds the activations to FP8)
        // is switched off here and checked against its own emulation by the `fp8blk` group.
        if self.filter.wants_kind(0x8000_0021) {
            #[cfg(feature = "cuda")]
            crate::backend::cuda::hopper_fp8::set_fp8_gemm_mode(0);
            for (rows, cols) in [(384usize, 1024usize), (259, 1000), (200, 640)] {
                for batch in [1usize, 2, 3, 4, 5, 8, 9, 16, 17, 40, 129, 356] {
                    let with_bias = batch == 3 || batch == 129;
                    let r = self.linear(0x8000_0021, rows, cols, batch, with_bias).map(|mut o| {
                        o.tol = o.tol.max(Self::FAST_TOL);
                        o
                    });
                    self.record(
                        "fast linear fp8b",
                        &format!("{rows}x{cols} batch {batch}{}", if with_bias { " +bias" } else { "" }),
                        r,
                    );
                }
            }
            #[cfg(feature = "cuda")]
            crate::backend::cuda::hopper_fp8::set_fp8_gemm_mode(-1);
        }
        // NVFP4 through the Marlin W4A16 kernel (the default on GPUs without FP4 tensor cores, FLLM_NVFP4_MARLIN=1 elsewhere); the
        // FP4 tensor-core kernel quantizes its activations to FP4 and is held to a looser bound by the model tests instead
        if self.filter.wants_kind(40) && std::env::var_os("FLLM_NVFP4_MARLIN").is_some() {
            for batch in [1usize, 2, 4, 8, 16, 40, 129, 356] {
                let r = self.linear(40, 384, 1024, batch, batch == 3 || batch == 129).map(|mut o| {
                    o.tol = o.tol.max(Self::FAST_TOL);
                    o
                });
                self.record("fast linear nvfp4", &format!("384x1024 batch {batch} (Marlin)"), r);
            }
        }
        // NVFP4 on the FP4 tensor cores (W4A4): the activations are quantized to FP4 with the per-tensor factor of the checkpoint's input_scale, exactly as
        // vLLM does, and the result is compared with a CPU simulation of that recipe. Skipped (unsupported) where the weights do not stay NVFP4
        // on the device (no sm_12x GPU, FLLM_NVFP4_MARLIN). `calibrated` tokens span 6 orders of magnitude, so a missing activation factor (FLLM_NVFP4_ACT_SCALE=off) fails.
        if self.filter.wants_kind(40) && std::env::var_os("FLLM_NVFP4_MARLIN").is_none() && std::env::var_os("FLLM_CUDA_EXACT").is_none() {
            // (rows, columns, batch, input_scale factor, result kind). Every round draws new data and runs the cases in another order (forward,
            // reversed, rotated, ...), so that state left behind by one case (cached activations, scratch, a weight buffer reused at the same address)
            // meets different successors. FLLM_SELFTEST_NVFP4_ROUNDS sets the number of rounds (default 2).
            let cases: [(usize, usize, usize, f32, u8); 10] = [
                (384, 1024, 1, 1.0, 0),
                (384, 1024, 8, 1.0, 0),
                (384, 1024, 40, 1.0, 0),
                (384, 1024, 129, 1.0, 0),
                (384, 1024, 356, 1.0, 0),
                (384, 1024, 129, 0.25, 0), // input_scale smaller than calibrated: activations saturate at 448 * 6 / 4
                (384, 1024, 356, 1.0, 1),  // accumulating epilogue (output projections)
                (384, 1024, 356, 1.0, 2),  // FP16 results (q/k/v, gate/up)
                (384, 1024, 356, 1.0, 3),  // after an RMSNorm (FLLM_FUSE_NVFP4_NORM=1 runs both in one kernel)
                (768, 2048, 200, 1.0, 0),  // another shape: a weight buffer of a different size follows the others
            ];
            let rounds = std::env::var("FLLM_SELFTEST_NVFP4_ROUNDS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(2).max(1);
            for round in 0..rounds {
                let mut order: Vec<usize> = (0..cases.len()).collect();
                match round % 3 {
                    1 => order.reverse(),
                    2 => order.rotate_left(4),
                    _ => {}
                }
                for i in order {
                    let (rows, cols, batch, factor, mode) = cases[i];
                    let r = self.nvfp4_w4a4(rows, cols, batch, factor, mode);
                    let label = match mode {
                        1 => " +acc",
                        2 => " f16",
                        3 => " +rms_norm",
                        _ => "",
                    };
                    let again = if round == 0 { String::new() } else { format!(" round {round}") };
                    self.record("nvfp4 w4a4", &format!("{rows}x{cols} batch {batch} input_scale x{factor}{label}{again}"), r);
                }
            }
        }
        // 32-value formats at a width that is no multiple of 256 (2880 = 90 blocks, like gpt-oss): padded device layout
        for kind in self.kinds(&[2u32, 3, 6, 7, 8, 20, 39]) {
            for batch in [1usize, 3, 5, 40, 356] {
                let r = self.linear(kind, 131, 2880, batch, batch == 3).map(|mut o| {
                    o.tol = o.tol.max(Self::FAST_TOL);
                    o
                });
                self.record(
                    &format!("fast linear {}", format_name(kind)),
                    &format!("131x2880 batch {batch}{}", if batch == 3 { " +bias" } else { "" }),
                    r,
                );
            }
        }
        // Marlin Q4_K tensor-core shapes (kind 12, 256-aligned rows and cols)
        for &(rows, cols, batches) in &[
            (4096, 4096, &[1usize, 2, 8, 15, 16, 17, 31, 33, 64, 65, 100, 129, 200, 356, 512][..]),
            (4096, 14336, &[16usize][..]),
            (14336, 4096, &[16usize, 17, 100, 356][..]),
            (4096, 14336, &[17usize, 100][..]),
        ] {
            for &batch in batches {
                if !self.filter.wants_kind(12) {
                    continue;
                }
                let r = self.linear(12, rows, cols, batch, false).map(|mut o| {
                    o.tol = o.tol.max(Self::FAST_TOL);
                    o
                });
                self.record(
                    "fast linear q4_k (Marlin)",
                    &format!("{rows}x{cols} batch {batch}"),
                    r,
                );
            }
        }

        // q/k/v style: the same activations through formats with different FP16 layouts, in both orders
        for kinds in [[12u32, 14], [14, 12], [12, 13], [13, 12]] {
            if !(self.filter.wants_kind(kinds[0]) && self.filter.wants_kind(kinds[1])) {
                continue;
            }
            // (batch 16 after another GEMM on the same stream differs by a few percent in the small-batch split-K kernels of
            // qgemm4k/5k/6k; the models do not show it, see docs/cuda-qgemm4k.md)
            for batch in [40usize, 129] {
                let r = self.shared_input(kinds, 4096, 4096, batch);
                self.record(
                    "shared input",
                    &format!("{}+{} 4096x4096 batch {batch}", format_name(kinds[0]), format_name(kinds[1])),
                    r,
                );
            }
        }
    }

    /// Greedy sampling on the device.
    fn sampling_cases(&mut self) {
        for (rows, vocab) in [(1usize, 151_936usize), (7, 128_256), (32, 151_936), (3, 1000), (5, 7)] {
            let r = self.argmax(rows, vocab);
            self.record("argmax", &format!("{rows} rows x {vocab}"), r);
        }
        for (rows, vocab) in [(6usize, 1000usize), (5, 32_003), (3, 151_936)] {
            let r = self.sample_controls(rows, vocab);
            self.record("sample_rows controls", &format!("{rows} rows x {vocab}: greedy with penalties, bias and grammar mask"), r);
            let r = self.sample_filters(rows.max(6), vocab);
            self.record("sample_rows filters", &format!("{} rows x {vocab}: top-k, top-p, min-p, typical-p, seeded draws", rows.max(6)), r);
            let r = self.sample_alternatives(vocab);
            self.record("sample_rows logprobs", &format!("{vocab} vocabulary: raw and processed log-probabilities, top alternatives"), r);
        }
        let r = self.sample_independent_of_the_batch(4000);
        self.record("sample_rows batch independence", "the same seeded row at other batch positions and sizes", r);
        let r = self.sample_distribution(40);
        self.record("sample_rows distribution", "8192 draws from one row against the host sampler's distribution", r);
    }

    /// The mismatches (rows whose token differs) between the device and the host reference of `sample_rows`.
    #[allow(clippy::too_many_arguments)]
    fn sample_compare(&mut self, logits: &[f32], vocab: usize, rows: &[crate::backend::SampleRowParams], pen: &[(i32, i32)], bias: &[(i32, f32)], mask: &[u32], mask_words: usize, top_stride: usize) -> Result<(usize, crate::backend::SampleOutput, Vec<crate::sampling::DeviceSample>, f64), BackendError> {
        let mut stream = self.device.create_stream()?;
        let buf = put(self.device, stream.as_mut(), logits)?;
        let batch = crate::backend::SampleBatch { rows, pen, bias, mask, mask_words, top_stride };
        let start = Instant::now();
        let got = stream.sample_rows(&buf, vocab, &batch)?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        if got.ids.len() != rows.len() {
            return Err(BackendError::DriverError(format!("sample_rows returned {} ids for {} rows", got.ids.len(), rows.len())));
        }
        let mut want = Vec::new();
        let mut wrong = 0;
        for (r, p) in rows.iter().enumerate() {
            let pens = &pen[p.pen_off as usize..(p.pen_off + p.pen_len) as usize];
            let biases = &bias[p.bias_off as usize..(p.bias_off + p.bias_len) as usize];
            let m = (p.mask_row >= 0).then(|| &mask[p.mask_row as usize * mask_words..(p.mask_row as usize + 1) * mask_words]);
            let reference = crate::sampling::device_sample_reference(&logits[r * vocab..(r + 1) * vocab], p, pens, biases, m, top_stride);
            if reference.id != got.ids[r] {
                wrong += 1;
            }
            want.push(reference);
        }
        Ok((wrong, got, want, ms))
    }

    /// Greedy rows with a repetition / presence / frequency penalty window, a logit bias and a grammar bitmask: the device and the host must
    /// pick the same token on every row (these steps are exact, there is no random draw).
    fn sample_controls(&mut self, rows: usize, vocab: usize) -> Result<Outcome, BackendError> {
        let logits = self.rng.vector(rows * vocab, 4.0);
        let words = vocab.div_ceil(32);
        let (mut params, mut pen, mut bias, mut mask) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for r in 0..rows {
            let pen_off = pen.len();
            for k in 0..9 {
                let id = ((self.rng.next() % vocab as u64) as i32).max(0);
                if !pen[pen_off..].iter().any(|e: &(i32, i32)| e.0 == id) {
                    pen.push((id, 1 + (k % 3)));
                }
            }
            let bias_off = bias.len();
            let top = (self.rng.next() % vocab as u64) as i32;
            bias.push((top, 3.0 + r as f32));
            let mut mask_row = -1;
            if r % 2 == 1 {
                mask_row = (mask.len() / words) as i32;
                // allow about 60 % of the tokens
                for _w in 0..words {
                    mask.push((self.rng.next() as u32) | (self.rng.next() as u32));
                }
            }
            params.push(crate::backend::SampleRowParams {
                pen_off: pen_off as i32,
                pen_len: (pen.len() - pen_off) as i32,
                repetition_penalty: 1.0 + 0.3 * (r % 3) as f32,
                presence_penalty: 0.5 * (r % 2) as f32,
                frequency_penalty: 0.25 * (r % 4) as f32,
                bias_off: bias_off as i32,
                bias_len: 1,
                mask_row,
                ..Default::default()
            });
        }
        let (wrong, _, _, ms) = self.sample_compare(&logits, vocab, &params, &pen, &bias, &mask, words, 0)?;
        Ok(Outcome { device: wrong as f64, cpu: 0.0, tol: 0.0, cpu_tol: 1.0, ms })
    }

    /// Stochastic rows (cycling through the filter combinations) with their own seeds and draw numbers: the same tokens as the host
    /// reference, apart from a few rows at most whose threshold decision sits within rounding of a boundary.
    fn sample_filters(&mut self, rows: usize, vocab: usize) -> Result<Outcome, BackendError> {
        let logits = self.rng.vector(rows * vocab, 3.0);
        let mut params = Vec::new();
        for r in 0..rows {
            let mut p = crate::backend::SampleRowParams { temperature: 0.6 + 0.2 * (r % 5) as f32, seed: 1000 + r as u64, draw: (r * 7) as u64, ..Default::default() };
            match r % 6 {
                0 => {}
                1 => p.top_k = 40,
                2 => p.top_p = 0.85,
                3 => p.min_p = 0.05,
                4 => p.typical_p = 0.8,
                _ => {
                    p.top_k = 200;
                    p.top_p = 0.9;
                    p.min_p = 0.01;
                }
            }
            params.push(p);
        }
        let (wrong, _, _, ms) = self.sample_compare(&logits, vocab, &params, &[], &[], &[], 0, 0)?;
        Ok(Outcome { device: wrong as f64, cpu: 0.0, tol: (rows / 100) as f64, cpu_tol: 1.0, ms })
    }

    /// Log-probabilities of the raw and of the sampled distribution and the best alternatives.
    fn sample_alternatives(&mut self, vocab: usize) -> Result<Outcome, BackendError> {
        let rows = 4;
        let logits = self.rng.vector(rows * vocab, 3.0);
        let stride = 8;
        let mut params = Vec::new();
        for r in 0..rows {
            params.push(crate::backend::SampleRowParams {
                temperature: 0.9,
                top_k: if r >= 2 { 6 } else { 0 },
                seed: 5 + r as u64,
                logprobs_mode: 1 + (r % 2) as i32,
                top_n: stride as i32,
                ..Default::default()
            });
        }
        let (wrong, got, want, ms) = self.sample_compare(&logits, vocab, &params, &[], &[], &[], 0, stride)?;
        // element r * (stride + 1): the chosen token (its id, or its log-probability when the id is right); r * (stride + 1) + 1 + j: the
        // j-th alternative (its id, or its log-probability when the id is right); 1e3 for a wrong id
        let len = rows * (stride + 1);
        let mut worst = wrong as f64 * 1e3;
        let mut at = (if wrong > 0 { worst } else { 0.0 }, 0usize, 0.0f32, 0.0f64);
        let mut see = |err: f64, index: usize, have: f32, want: f64| {
            if err > at.0 || (err.is_nan() && !at.0.is_nan()) {
                at = (err, index, have, want);
            }
        };
        for (r, reference) in want.iter().enumerate() {
            if got.ids[r] != reference.id {
                see(1e3, r * (stride + 1), got.ids[r] as f32, f64::from(reference.id));
            } else {
                let e = f64::from((got.logprobs[r] - reference.logprob).abs());
                worst = worst.max(e);
                see(e, r * (stride + 1), got.logprobs[r], f64::from(reference.logprob));
            }
            for j in 0..stride {
                let (gi, gl) = (got.top_ids[r * stride + j], got.top_logprobs[r * stride + j]);
                let index = r * (stride + 1) + 1 + j;
                match reference.top.get(j) {
                    Some(&(wi, wl)) => {
                        if gi != wi {
                            worst = worst.max(1e3);
                            see(1e3, index, gi as f32, f64::from(wi));
                        } else {
                            let e = f64::from((gl - wl).abs());
                            worst = worst.max(e);
                            see(e, index, gl, f64::from(wl));
                        }
                    }
                    None => {
                        if gi != -1 {
                            worst = worst.max(1e3);
                            see(1e3, index, gi as f32, -1.0);
                        }
                    }
                }
            }
        }
        if worst > 2e-4 {
            // the token ids of a wrong draw and of a wrong alternative are reported as "got" / "expected" ids
            note_worst(worst, at.1, len, at.2, at.3, 1.0);
        }
        Ok(Outcome { device: worst, cpu: 0.0, tol: 2e-4, cpu_tol: 1.0, ms })
    }

    /// The token of a seeded row must not depend on the batch it runs in: the same logits, seed and draw at other positions and in
    /// batches of other sizes give the same token.
    fn sample_independent_of_the_batch(&mut self, vocab: usize) -> Result<Outcome, BackendError> {
        let one = self.rng.vector(vocab, 3.0);
        let other = self.rng.vector(vocab, 3.0);
        let row = crate::backend::SampleRowParams { temperature: 1.0, top_p: 0.9, seed: 42, draw: 17, ..Default::default() };
        let mut ids = Vec::new();
        let mut total_ms = 0.0;
        for (position, size) in [(0usize, 1usize), (0, 3), (2, 3), (5, 9), (8, 9)] {
            let mut logits = Vec::new();
            let mut params = Vec::new();
            for r in 0..size {
                if r == position {
                    logits.extend_from_slice(&one);
                    params.push(row);
                } else {
                    logits.extend_from_slice(&other);
                    params.push(crate::backend::SampleRowParams { seed: 9 + r as u64, ..row });
                }
            }
            let (_, got, _, ms) = self.sample_compare(&logits, vocab, &params, &[], &[], &[], 0, 0)?;
            total_ms += ms;
            ids.push(got.ids[position]);
        }
        let differing = ids.iter().filter(|&&i| i != ids[0]).count();
        Ok(Outcome { device: differing as f64, cpu: 0.0, tol: 0.0, cpu_tol: 1.0, ms: total_ms })
    }

    /// Statistical check of the draw: 8192 rows hold the same logits and differ only in their draw number. The frequencies must match the
    /// distribution of the host sampler (`processed_distribution`) for a top-k / top-p / min-p combination.
    fn sample_distribution(&mut self, vocab: usize) -> Result<Outcome, BackendError> {
        let trials = 8192usize;
        let one = self.rng.vector(vocab, 2.0);
        let mut logits = Vec::with_capacity(trials * vocab);
        let mut params = Vec::with_capacity(trials);
        let host = crate::sampling::SamplingParams { temperature: 0.9, top_k: 20, top_p: 0.92, min_p: 0.01, ..Default::default() };
        for t in 0..trials {
            logits.extend_from_slice(&one);
            params.push(crate::backend::SampleRowParams { temperature: 0.9, top_k: 20, top_p: 0.92, min_p: 0.01, seed: 2024, draw: t as u64, ..Default::default() });
        }
        let (_, got, _, ms) = self.sample_compare(&logits, vocab, &params, &[], &[], &[], 0, 0)?;
        let expected = crate::sampling::processed_distribution(&one, &[], &host, 0.0);
        let mut counts = vec![0u32; vocab];
        for &id in &got.ids {
            if id < 0 || id as usize >= vocab {
                return Err(BackendError::DriverError(format!("sample_rows returned the invalid token {id}")));
            }
            counts[id as usize] += 1;
        }
        let (mut chi, mut cells, mut outside) = (0.0f64, 0, 0u32);
        for (id, &count) in counts.iter().enumerate() {
            match expected.iter().find(|e| e.0 == id) {
                None => outside += count,
                Some(&(_, p)) => {
                    let e = f64::from(p) * trials as f64;
                    if e > 5.0 {
                        chi += (f64::from(count) - e).powi(2) / e;
                        cells += 1;
                    }
                }
            }
        }
        // tokens outside the support must never be drawn; inside, chi-square far above its mean (cells - 1) is a failure
        let device = if outside > 0 { f64::INFINITY } else { chi };
        Ok(Outcome { device, cpu: 0.0, tol: 12.0 + 3.0 * f64::from(cells), cpu_tol: 1.0, ms })
    }

    /// Weights streamed from host memory: Q4_K / Q5_K / Q6_K / Q8_0 and the 32-value formats at an unaligned width.
    fn host_cases(&mut self) {
        for kind in self.kinds(&[8u32, 12, 13, 14, 39]) {
            for batch in [3usize, 64] {
                let r = self.host_linear(kind, 130, 768, batch);
                self.record(
                    &format!("host linear {}", format_name(kind)),
                    &format!("130x768 batch {batch}"),
                    r,
                );
            }
        }
        for kind in self.kinds(&[8u32, 39]) {
            let r = self.host_linear(kind, 70, 2880, 20);
            self.record(&format!("host linear {}", format_name(kind)), "70x2880 batch 20", r);
        }
    }

    /// Mixture-of-experts: row gather / scatter, routing, expert products, combination.
    fn moe_cases(&mut self) {
        let r = self.gather_scatter();
        self.record("gather/scatter rows", "37x130, 5 rows", r);
        for normalize in [false, true] {
            let r = self.moe_topk(3, 64, 8, normalize);
            self.record("moe top-k", &format!("3 rows 64 experts k8{}", if normalize { " norm" } else { "" }), r);
        }
        for kind in self.kinds(&[12u32, 13, 14, 2, 8, 20, 39]) {
            let r = self.moe_linear(kind, 130, 512, 6, 3, kind == 39 || kind == 12);
            self.record(
                &format!("moe linear {}", format_name(kind)),
                &format!("6 experts 130x512, 6 slots{}", if kind == 39 || kind == 12 { " +bias" } else { "" }),
                r,
            );
        }
        if self.filter.wants_kind(39) {
            let r = self.moe_linear(39, 70, 2880, 8, 4, true);
            self.record("moe linear mxfp4", "6 experts 70x2880, 8 slots +bias", r);
        }
        let r = self.moe_combine(2, 3, 70);
        self.record("moe combine", "2 tokens k3 x70", r);
        self.moe_device_cases();
        self.run_kv_formats_fast();
    }

    /// Softmax routing on the device against an f64 reference; a wrong expert index counts as a full error.
    fn moe_topk(&mut self, rows: usize, experts: usize, k: usize, normalize: bool) -> Result<Outcome, BackendError> {
        let logits = self.rng.vector(rows * experts, 3.0);
        let scale = 0.5f32;
        let mut stream = self.device.create_stream()?;
        let lb = put(self.device, stream.as_mut(), &logits)?;
        let mut ids = self.device.allocate_buffer(rows * k * 4)?;
        let mut weights = self.device.allocate_buffer(rows * k * 4)?;
        let mut ms = 0.0;
        timed(&mut ms, stream.as_mut(), |s| s.moe_topk(&lb, &mut ids, &mut weights, rows, experts, k, normalize, scale))?;
        let mut raw = vec![0u8; rows * k * 4];
        stream.download(&mut raw, &ids)?;
        let have_ids: Vec<i32> = raw.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c)).collect();
        let have_w = get(stream.as_mut(), &weights, rows * k)?;
        let mut worst_err = 0.0f64;
        for r in 0..rows {
            let row = &logits[r * experts..(r + 1) * experts];
            let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
            let sum: f64 = row.iter().map(|&v| f64::from(v - max).exp()).sum();
            let mut order: Vec<usize> = (0..experts).collect();
            order.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
            let chosen = &order[..k];
            let scores: Vec<f64> = chosen.iter().map(|&e| f64::from(row[e] - max).exp() / sum).collect();
            let total: f64 = scores.iter().sum();
            for (j, &e) in chosen.iter().enumerate() {
                let want = if normalize { scores[j] / total } else { scores[j] } * f64::from(scale);
                if have_ids[r * k + j] != e as i32 {
                    worst_err = worst_err.max(1.0);
                } else {
                    worst_err = worst_err.max((f64::from(have_w[r * k + j]) - want).abs() / want.abs().max(1e-9));
                }
            }
        }
        Ok(Outcome { device: worst_err, cpu: 0.0, tol: 5e-6, cpu_tol: 1.0, ms })
    }

    /// `moe_linear`: expert-indexed products through a pointer table (int8 activations, so the fast-kernel bound).
    fn moe_linear(&mut self, kind: u32, rows: usize, cols: usize, slots: usize, x_div: usize, with_bias: bool) -> Result<Outcome, BackendError> {
        let experts = 6usize;
        let quant = QuantType::from_gguf_type_id(kind);
        let raw: Vec<Vec<u8>> = (0..experts).map(|_| random_kind(&mut self.rng, kind, rows, cols)).collect();
        let dense: Vec<Vec<f32>> = raw
            .iter()
            .map(|w| crate::compute::dequantize(w, kind, rows * cols).map_err(BackendError::DriverError))
            .collect::<Result<_, _>>()?;
        let inputs = slots.div_ceil(x_div);
        let x = self.rng.vector(inputs * cols, 1.0);
        let ids: Vec<i32> = (0..slots).map(|s| ((s * 5 + 3) % experts) as i32).collect();
        let biases: Vec<Vec<f32>> = (0..experts).map(|_| self.rng.vector(rows, 1.0)).collect();
        let mut want = vec![0.0; slots * rows];
        let mut scale = vec![0.0; slots * rows];
        for s in 0..slots {
            let xv = &x[(s / x_div) * cols..(s / x_div + 1) * cols];
            for r in 0..rows {
                let (mut sum, mut mag) = (0.0f64, 0.0f64);
                for (&a, &b) in xv.iter().zip(&dense[ids[s] as usize][r * cols..(r + 1) * cols]) {
                    let term = f64::from(a) * f64::from(b);
                    sum += term;
                    mag += term.abs();
                }
                if with_bias {
                    let bv = f64::from(biases[ids[s] as usize][r]);
                    sum += bv;
                    mag += bv.abs();
                }
                want[s * rows + r] = sum;
                scale[s * rows + r] = mag;
            }
        }
        let weights: Vec<crate::backend::DeviceWeight> = raw
            .iter()
            .zip(&biases)
            .map(|(w, b)| self.device.upload_weight("selftest", w, quant, rows, cols, with_bias.then_some(b.as_slice())))
            .collect::<Result<_, _>>()?;
        let refs: Vec<&crate::backend::DeviceWeight> = weights.iter().collect();
        let table = self.device.upload_pointer_table(&refs)?;
        let bias_table = if with_bias { Some(self.device.upload_bias_table(&refs)?) } else { None };
        let mut stream = self.device.create_stream()?;
        let xb = put(self.device, stream.as_mut(), &x)?;
        let mut idb = self.device.allocate_buffer(slots * 4)?;
        let id_bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
        stream.upload(&mut idb, &id_bytes)?;
        let mut out = self.device.allocate_buffer(slots * rows * 4)?;
        let mut ms = 0.0;
        timed(&mut ms, stream.as_mut(), |s| s.moe_linear(&mut out, &xb, &table, bias_table.as_ref(), &idb, quant, rows, cols, slots, x_div))?;
        let have = get(stream.as_mut(), &out, slots * rows)?;
        Ok(Outcome { device: worst(&have, &want, &scale), cpu: 0.0, tol: Self::FAST_TOL, cpu_tol: 1.0, ms })
    }

    /// `moe_combine`: weighted sum of the k expert outputs of every token.
    fn moe_combine(&mut self, tokens: usize, k: usize, hidden: usize) -> Result<Outcome, BackendError> {
        let y = self.rng.vector(tokens * k * hidden, 1.0);
        let w = self.rng.vector(tokens * k, 1.0);
        let mut want = vec![0.0f64; tokens * hidden];
        let mut scale = vec![0.0f64; tokens * hidden];
        for t in 0..tokens {
            for h in 0..hidden {
                for j in 0..k {
                    let term = f64::from(w[t * k + j]) * f64::from(y[(t * k + j) * hidden + h]);
                    want[t * hidden + h] += term;
                    scale[t * hidden + h] += term.abs();
                }
            }
        }
        let mut stream = self.device.create_stream()?;
        let yb = put(self.device, stream.as_mut(), &y)?;
        let wb = put(self.device, stream.as_mut(), &w)?;
        let mut out = self.device.allocate_buffer(tokens * hidden * 4)?;
        let mut ms = 0.0;
        timed(&mut ms, stream.as_mut(), |s| s.moe_combine(&mut out, &yb, &wb, tokens, k, hidden))?;
        let have = get(stream.as_mut(), &out, tokens * hidden)?;
        Ok(Outcome { device: worst(&have, &want, &scale), cpu: 0.0, tol: gamma(k + 2), cpu_tol: 1.0, ms })
    }

    /// `host_linear`: the weight is copied from host memory for this one product.
    fn host_linear(&mut self, kind: u32, rows: usize, cols: usize, batch: usize) -> Result<Outcome, BackendError> {
        let weights = random_kind(&mut self.rng, kind, rows, cols);
        let x = self.rng.vector(batch * cols, 1.0);
        let w = crate::compute::dequantize(&weights, kind, rows * cols).map_err(BackendError::DriverError)?;
        let mut want = vec![0.0; batch * rows];
        let mut scale = vec![0.0; batch * rows];
        for b in 0..batch {
            for r in 0..rows {
                let (mut sum, mut mag) = (0.0f64, 0.0f64);
                for (&xv, &wv) in x[b * cols..(b + 1) * cols].iter().zip(&w[r * cols..(r + 1) * cols]) {
                    let term = f64::from(xv) * f64::from(wv);
                    sum += term;
                    mag += term.abs();
                }
                want[b * rows + r] = sum;
                scale[b * rows + r] = mag;
            }
        }
        let quant = QuantType::from_gguf_type_id(kind);
        let mut stream = self.device.create_stream()?;
        if !stream.supports_host_linear(quant) {
            return Err(BackendError::UnsupportedQuant(quant));
        }
        let xb = put(self.device, stream.as_mut(), &x)?;
        let mut out = self.device.allocate_buffer(batch * rows * 4)?;
        let mut ms = 0.0;
        timed(&mut ms, stream.as_mut(), |s| s.host_linear(&mut out, &xb, &weights, quant, rows, cols, None))?;
        let have = get(stream.as_mut(), &out, batch * rows)?;
        Ok(Outcome { device: worst(&have, &want, &scale), cpu: 0.0, tol: Self::FAST_TOL, cpu_tol: 1.0, ms })
    }

    /// Row gather with weights followed by a weighted scatter-add into a zeroed buffer (mixture-of-experts layers).
    fn gather_scatter(&mut self) -> Result<Outcome, BackendError> {
        let (total, row_len) = (37usize, 130usize);
        let src = self.rng.vector(total * row_len, 1.0);
        let idx = [5i32, 36, 0, 17, 9];
        let weights = [0.5f32, -1.0, 2.0, 0.25, 1.5];
        let mut stream = self.device.create_stream()?;
        let srcb = put(self.device, stream.as_mut(), &src)?;
        let mut gathered = self.device.allocate_buffer(idx.len() * row_len * 4)?;
        let mut acc = self.device.allocate_buffer(total * row_len * 4)?;
        let mut ms = 0.0;
        timed(&mut ms, stream.as_mut(), |s| {
            s.zero_buffer(&mut acc)?;
            s.gather_rows(&mut gathered, &srcb, &idx, Some(&weights), row_len)?;
            s.scatter_add_rows(&mut acc, &gathered, &idx, None, row_len)?;
            s.scatter_add_rows(&mut acc, &gathered, &idx, Some(&weights), row_len)
        })?;
        let have = get(stream.as_mut(), &acc, total * row_len)?;
        let mut want = vec![0.0f64; total * row_len];
        for (i, &r) in idx.iter().enumerate() {
            for c in 0..row_len {
                let g = f64::from(src[r as usize * row_len + c]) * f64::from(weights[i]);
                want[r as usize * row_len + c] += g + g * f64::from(weights[i]);
            }
        }
        let peak = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let scale = vec![peak; want.len()];
        Ok(Outcome { device: worst(&have, &want, &scale), cpu: 0.0, tol: 4.0 * U, cpu_tol: 1.0, ms })
    }

    /// Gated DeltaNet layer against `cpu_delta` (the host reference): prefill runs of several lengths (shorter than the
    /// convolution history too), decode steps of three sequences in one batch, continued prefill mixed with decode.
    fn delta_cases(&mut self) {
        for (kh, vh, hk, hv, tiled) in [
            (2usize, 2usize, 64usize, 64usize, false),
            (2, 4, 64, 32, true),
            (4, 8, 128, 128, false),
            (4, 8, 128, 128, true),
            (16, 32, 128, 128, true),
            (2, 4, 128, 96, false),
        ] {
            let shape = DeltaShape {
                k_heads: kh,
                v_heads: vh,
                k_dim: hk,
                v_dim: hv,
                conv_kernel: 4,
                tiled_heads: tiled,
                beta_scale: 1.0,
                eps: 1e-6,
            };
            let label = format!("{kh}/{vh} heads {hk}x{hv} {}", if tiled { "tiled" } else { "grouped" });
            match self.device.create_stream() {
                Ok(s) if s.supports_delta(&shape) => {}
                Ok(_) => {
                    self.record_skip("delta_net", &label, "delta_layer for this shape (supports_delta is false)");
                    continue;
                }
                Err(e) => {
                    self.record("delta_net", &label, Err(e));
                    continue;
                }
            }
            let r = self.delta(&shape);
            self.record("delta_net", &label, r);
        }
    }

    fn kda_cases(&mut self) {
        let shapes = [
            KdaShape { heads: 2, head_dim: 16, conv_kernel: 4, eps: 1e-6, lower_bound: None },
            KdaShape { heads: 2, head_dim: 32, conv_kernel: 4, eps: 1e-6, lower_bound: None },
            KdaShape { heads: 2, head_dim: 64, conv_kernel: 4, eps: 1e-6, lower_bound: None },
            KdaShape { heads: 2, head_dim: 128, conv_kernel: 4, eps: 1e-6, lower_bound: None },
            KdaShape { heads: 2, head_dim: 32, conv_kernel: 4, eps: 1e-6, lower_bound: Some(-0.5) },
        ];
        for shape in shapes {
            let label = format!("{} heads {}x{} ksz={}{}", shape.heads, shape.head_dim, shape.head_dim, shape.conv_kernel, if shape.lower_bound.is_some() { " bounded" } else { "" });
            match self.device.create_stream() {
                Ok(s) if s.supports_kda(&shape) => {}
                Ok(_) => {
                    self.record_skip("kda", &label, "kda_layer for this shape (supports_kda is false)");
                    continue;
                }
                Err(e) => {
                    self.record("kda", &label, Err(e));
                    continue;
                }
            }
            let r = self.kda(&shape);
            self.record("kda", &label, r);
        }
    }

    fn kda(&mut self, shape: &KdaShape) -> Result<Outcome, BackendError> {
        let (heads, hd, ksz) = (shape.heads, shape.head_dim, shape.conv_kernel);
        let conv_dim = shape.conv_dim();
        let width = heads * hd;
        let conv = self.rng.vector(conv_dim * ksz, 0.6);
        let dt_bias = self.rng.vector(width, 1.0);
        let rate: Vec<f32> = (0..heads).map(|_| 0.5 + self.rng.unit().abs() * 1.5).collect();
        let norm: Vec<f32> = (0..hd).map(|_| 1.0 + 0.25 * self.rng.unit()).collect();
        let params = crate::cpu_kda::Layer {
            heads,
            k_dim: hd,
            v_dim: hd,
            conv_kernel: ksz,
            eps: shape.eps,
            lower_bound: shape.lower_bound,
            conv: &conv,
            dt_bias: &dt_bias,
            rate: &rate,
            norm: &norm,
        };
        // runs of 64 rows and more take the chunk-parallel scan on the devices that have one (partial and exact chunks, runs that continue a state,
        // a long run next to short ones that take the serial kernel)
        let calls: [&[(usize, usize)]; 10] = [
            &[(0, 37), (1, 5), (2, 2)],
            &[(0, 1), (1, 1), (2, 1)],
            &[(0, 1), (1, 1), (2, 1)],
            &[(0, 20), (1, 1)],
            &[(2, 6), (0, 1), (1, 3)],
            &[(1, 1), (0, 1), (2, 1)],
            &[(0, 150), (1, 64), (2, 3)],
            &[(1, 65), (2, 130)],
            &[(0, 1), (2, 64), (1, 1)],
            &[(0, 70), (1, 2), (2, 1)],
        ];
        let mut stream = self.device.create_stream()?;
        let conv_buf = put(self.device, stream.as_mut(), &conv)?;
        let dt_buf = put(self.device, stream.as_mut(), &dt_bias)?;
        let rate_buf = put(self.device, stream.as_mut(), &rate)?;
        let norm_buf = put(self.device, stream.as_mut(), &norm)?;
        let weights = KdaWeights { conv: &conv_buf, dt_bias: &dt_buf, rate: &rate_buf, norm: &norm_buf };
        let offset = 256;
        let states: Vec<DeviceBuffer> = (0..3)
            .map(|_| self.device.allocate_buffer(offset + shape.layer_state_bytes()))
            .collect::<Result<_, _>>()?;
        let mut cpu_history = vec![vec![0.0f32; conv_dim * (ksz - 1)]; 3];
        let mut cpu_state = vec![vec![0.0f32; heads * hd * hd]; 3];
        let mut seen = [false; 3];
        let mut worst_err = 0.0f64;
        let mut total_ms = 0.0;
        for call in calls {
            let rows: usize = call.iter().map(|&(_, n)| n).sum();
            let qkv = self.rng.vector(rows * conv_dim, 1.0);
            let forget = self.rng.vector(rows * width, 1.0);
            let beta = self.rng.vector(rows * heads, 1.0);
            let gate = self.rng.vector(rows * width, 1.0);
            let mut want = vec![0.0f32; rows * width];
            let mut row = 0;
            let mut runs = Vec::with_capacity(call.len());
            for &(seq, n) in call {
                let reset = !seen[seq];
                seen[seq] = true;
                if reset {
                    cpu_history[seq].fill(0.0);
                    cpu_state[seq].fill(0.0);
                }
                crate::cpu_kda::run(
                    &params,
                    &mut cpu_history[seq],
                    &mut cpu_state[seq],
                    (
                        &qkv[row * conv_dim..(row + n) * conv_dim],
                        &forget[row * width..(row + n) * width],
                        &beta[row * heads..(row + n) * heads],
                        &gate[row * width..(row + n) * width],
                    ),
                    &mut want[row * width..(row + n) * width],
                    n,
                );
                runs.push(DeltaRun {
                    state: &states[seq],
                    first_row: row,
                    rows: n,
                    reset,
                });
                row += n;
            }
            let qkv_buf = put(self.device, stream.as_mut(), &qkv)?;
            let forget_buf = put(self.device, stream.as_mut(), &forget)?;
            let beta_buf = put(self.device, stream.as_mut(), &beta)?;
            let gate_buf = put(self.device, stream.as_mut(), &gate)?;
            let mut mixed = self.device.allocate_buffer(rows * conv_dim * 4)?;
            let mut out = self.device.allocate_buffer(rows * width * 4)?;
            let mut ms = 0.0;
            timed(&mut ms, stream.as_mut(), |s| {
                s.kda_layer(
                    shape, &weights, &runs, offset, &qkv_buf, &forget_buf, &beta_buf, &gate_buf, &mut mixed, &mut out,
                )
            })?;
            total_ms += ms;
            let have = get(stream.as_mut(), &out, rows * width)?;
            let want64: Vec<f64> = want.iter().map(|&v| f64::from(v)).collect();
            worst_err = worst_err.max(worst(&have, &want64, &vec![1.0; want.len()]));
        }
        Ok(Outcome { device: worst_err, cpu: 0.0, tol: 2e-3, cpu_tol: 1.0, ms: total_ms })
    }

    fn delta(&mut self, shape: &DeltaShape) -> Result<Outcome, BackendError> {
        let (kh, vh, hk, hv, ksz) = (shape.k_heads, shape.v_heads, shape.k_dim, shape.v_dim, shape.conv_kernel);
        let (conv_dim, value_dim) = (shape.conv_dim(), shape.value_dim());
        let conv = self.rng.vector(conv_dim * ksz, 0.6);
        let dt_bias = self.rng.vector(vh, 1.0);
        let decay: Vec<f32> = (0..vh).map(|_| -(0.3 + self.rng.unit().abs() * 2.0)).collect();
        let norm: Vec<f32> = (0..hv).map(|_| 1.0 + 0.25 * self.rng.unit()).collect();
        let params = crate::cpu_delta::Layer {
            k_heads: kh,
            v_heads: vh,
            k_dim: hk,
            v_dim: hv,
            conv_kernel: ksz,
            beta_scale: shape.beta_scale,
            eps: shape.eps,
            tiled_heads: shape.tiled_heads,
            conv: &conv,
            dt_bias: &dt_bias,
            decay: &decay,
            norm: &norm,
        };
        // (sequence, tokens) per call
        let calls: [&[(usize, usize)]; 10] = [
            &[(0, 37), (1, 5), (2, 2)],
            &[(0, 1), (1, 1), (2, 1)],
            &[(0, 1), (1, 1), (2, 1)],
            &[(0, 20), (1, 1)],
            &[(2, 6), (0, 1), (1, 3)],
            &[(1, 1), (0, 1), (2, 1)],
            &[(0, 150), (1, 64), (2, 3)],
            &[(1, 65), (2, 130)],
            &[(0, 1), (2, 64), (1, 1)],
            &[(0, 70), (1, 2), (2, 1)],
        ];
        let mut stream = self.device.create_stream()?;
        let conv_buf = put(self.device, stream.as_mut(), &conv)?;
        let dt_buf = put(self.device, stream.as_mut(), &dt_bias)?;
        let decay_buf = put(self.device, stream.as_mut(), &decay)?;
        let norm_buf = put(self.device, stream.as_mut(), &norm)?;
        let weights = DeltaWeights { conv: &conv_buf, dt_bias: &dt_buf, decay: &decay_buf, norm: &norm_buf };
        // the layer's state sits 256 bytes into the buffer (as a second layer would)
        let offset = 256;
        let states: Vec<DeviceBuffer> = (0..3)
            .map(|_| self.device.allocate_buffer(offset + shape.layer_state_bytes()))
            .collect::<Result<_, _>>()?;
        let mut hist = vec![vec![0.0f32; conv_dim * (ksz - 1)]; 3];
        let mut rec = vec![vec![0.0f32; vh * hk * hv]; 3];
        let mut started = [false; 3];
        let (mut worst_err, mut total_ms) = (0.0f64, 0.0f64);
        for call in calls {
            let rows: usize = call.iter().map(|c| c.1).sum();
            let qkv = self.rng.vector(rows * conv_dim, 1.5);
            let z = self.rng.vector(rows * value_dim, 2.0);
            let beta = self.rng.vector(rows * vh, 2.0);
            let alpha = self.rng.vector(rows * vh, 2.0);
            let mut want = vec![0.0f32; rows * value_dim];
            let mut runs = Vec::new();
            let mut first = 0;
            for &(seq, n) in call {
                crate::cpu_delta::run(
                    &params,
                    &mut hist[seq],
                    &mut rec[seq],
                    (
                        &qkv[first * conv_dim..(first + n) * conv_dim],
                        &z[first * value_dim..(first + n) * value_dim],
                        &beta[first * vh..(first + n) * vh],
                        &alpha[first * vh..(first + n) * vh],
                    ),
                    &mut want[first * value_dim..(first + n) * value_dim],
                    n,
                );
                runs.push(DeltaRun { state: &states[seq], first_row: first, rows: n, reset: !started[seq] });
                started[seq] = true;
                first += n;
            }
            let qkv_buf = put(self.device, stream.as_mut(), &qkv)?;
            let z_buf = put(self.device, stream.as_mut(), &z)?;
            let beta_buf = put(self.device, stream.as_mut(), &beta)?;
            let alpha_buf = put(self.device, stream.as_mut(), &alpha)?;
            let mut mixed = self.device.allocate_buffer(rows * conv_dim * 4)?;
            // poisoned output: every element must be written
            let mut out = put(self.device, stream.as_mut(), &vec![f32::NAN; rows * value_dim])?;
            let mut ms = 0.0;
            timed(&mut ms, stream.as_mut(), |s| {
                s.delta_layer(
                    shape, &weights, &runs, offset, &qkv_buf, &z_buf, &beta_buf, &alpha_buf, &mut mixed, &mut out,
                )
            })?;
            total_ms += ms;
            let have = get(stream.as_mut(), &out, rows * value_dim)?;
            let want64: Vec<f64> = want.iter().map(|&v| f64::from(v)).collect();
            worst_err = worst_err.max(worst(&have, &want64, &vec![1.0; want.len()]));
        }
        Ok(Outcome { device: worst_err, cpu: 0.0, tol: 2e-3, cpu_tol: 1.0, ms: total_ms })
    }

    fn run_all(&mut self) {
        // the recurrent layers take the chunk-parallel scan for their long runs in these checks (by default a device times both forms on the first calls and keeps
        // the faster one, which would leave it to the order of the calls which form a check meets)
        #[cfg(feature = "cuda")]
        crate::backend::cuda::set_prefill_switch(crate::backend::cuda::PrefillSwitch::ChunkedScan, 2);
        #[cfg(feature = "vulkan")]
        crate::backend::vulkan::ops::scan::set_policy(crate::backend::vulkan::ops::scan::Policy::Always);
        if self.enter("delta") {
            self.delta_cases();
        }
        if self.enter("kda") {
            self.kda_cases();
        }
        if self.enter("rwkv") {
            self.rwkv_cases();
        }
        if self.enter("arch") {
            self.arch_cases();
            self.dsa_cases();
            self.block_sparse_cases();
            self.inkling_cases();
            self.hc_cases();
            self.v4_cases();
        }
        if self.enter("linear") {
            self.linear_cases();
        }
        if self.enter("mlx") {
            self.mlx_cases(false);
        }
        // the FP8B decode is exact arithmetic, so the group belongs to this pass as well
        if self.enter("fp8blk") {
            self.fp8blk_cases();
        }
        if self.enter("embedding") {
            self.embedding_cases();
        }
        if self.enter("norm") {
            self.norm_cases();
        }
        if self.enter("rope") {
            self.rope_cases();
        }
        if self.enter("activation") {
            self.activation_cases();
        }
        if self.enter("kv") {
            self.kv_cases();
        }
        if self.enter("attention") {
            self.attention_run();
        }
        if self.enter("encoder") {
            self.encoder_cases();
        }
        // These groups use the operations of the fast pass (device sampling, expert products, weights from host memory). A backend without a
        // second pass meets them here and reports SKIPPED where it does not implement them.
        if self.extra_groups {
            if self.enter("sampling") {
                self.sampling_cases();
            }
            if self.enter("host") {
                self.host_cases();
            }
            if self.enter("moe") {
                self.moe_cases();
            }
            if self.filter.asked_for("moebench") && self.enter("moebench") {
                self.moe_bench_cases();
            }
            if self.enter("fuse") {
                self.fuse_cases();
            }
        }
    }

    fn linear_cases(&mut self) {
        // Projections: every native format, two converted at upload (Q2_K, IQ4_NL); 13 rows,
        // batches 1/2/5 (the kernel tiles four batch rows), odd column counts for floats and
        // odd block counts for quants, a bias, and more output rows than one grid dimension.
        for kind in self.kinds(&[0u32, 1, 30, 2, 3, 6, 7, 8, 12, 13, 14, 10, 11, 16, 17, 18, 19, 21, 22, 29, 20, 39, BNB[0], BNB[1], BNB[2], BNB[3], BNB[4]]) {
            let block = match kind {
                0 | 1 | 30 => 1,
                10..=14 | 16..=19 | 21 | 22 | 29 => 256,
                k if fllm_gguf_probe::bnb::is_bnb(k) => fllm_gguf_probe::bnb::layout(k).map_or(64, |l| l.block),
                _ => 32,
            };
            let cols = if block == 1 { 77 } else { 3 * block };
            for batch in [1, 2, 5] {
                let r = self.linear(kind, 13, cols, batch, batch == 5);
                self.record(
                    &format!("linear {}", format_name(kind)),
                    &format!(
                        "13x{cols} batch {batch}{}",
                        if batch == 5 { " +bias" } else { "" }
                    ),
                    r,
                );
            }
        }
        // Rows whose width is a multiple of 256 use the aligned device layouts of the legacy and Q8_0 formats
        // (and the tensor-core / int8 paths when they are not disabled by the exact-math switch).
        for kind in self.kinds(&[2u32, 3, 6, 7, 8, 12, 13, 14, 20, 39]) {
            for batch in [1, 2, 5] {
                let r = self.linear(kind, 13, 768, batch, batch == 5);
                self.record(
                    &format!("linear {}", format_name(kind)),
                    &format!("13x768 batch {batch}{}", if batch == 5 { " +bias" } else { "" }),
                    r,
                );
            }
        }
        if self.filter.wants_kind(8) {
            let r = self.linear(8, 70_001, 32, 2, false);
            self.record("linear q8_0", "70001x32 batch 2", r);
        }
        // Packed ExLlamaV2 matrices through the FP32 kernels: widths that are multiples of 32 with groups of 32 and 64 inputs of mixed code widths (codes that
        // straddle words), rows that fill no tile, batches on both sides of the matrix-vector range
        for kind in self.kinds(&[0x8000_0030u32]) {
            for (rows, cols) in [(13usize, 96usize), (40, 288)] {
                for batch in [1, 2, 5, 9] {
                    let r = self.linear(kind, rows, cols, batch, batch == 5);
                    self.record(&format!("linear {}", format_name(kind)), &format!("{rows}x{cols} batch {batch}{}", if batch == 5 { " +bias" } else { "" }), r);
                }
            }
        }
        // Packed ExLlamaV3 matrices through the FP32 kernels: every width (1 to 8 bits) with a codebook each, tiles in both directions, batches on both sides of
        // the 16-token groups of the trellis kernel
        if self.filter.wants_kind(0x8000_0031) {
            for bits in 1..=8usize {
                let codebook = (bits % 3) as u8;
                let (rows, cols) = (if bits % 2 == 0 { 256usize } else { 128 }, 128 + 128 * (bits % 3));
                for batch in [1usize, 17] {
                    let blob = fllm_gguf_probe::exl::synthetic_exl3(rows, cols, bits, codebook, &mut || self.rng.next());
                    let r = self.linear_with(0x8000_0031, blob, rows, cols, batch, batch == 17);
                    self.record("linear exl3", &format!("{rows}x{cols} {bits} bits codebook {codebook} batch {batch}{}", if batch == 17 { " +bias" } else { "" }), r);
                }
            }
        }
        // Per-row scaled FP8 at shapes the Marlin FP8 kernel cannot take (CUDA repacks them into one-row tiles of the block-wise kernels)
        for kind in self.kinds(&[0x8000_0020u32]) {
            for (rows, cols) in [(12usize, 77usize), (13, 300), (40, 200)] {
                for batch in [1, 2, 5] {
                    let r = self.linear(kind, rows, cols, batch, batch == 5);
                    self.record(&format!("linear {}", format_name(kind)), &format!("{rows}x{cols} batch {batch}{}", if batch == 5 { " +bias" } else { "" }), r);
                }
            }
        }
        // Block-wise FP8: the usual 128 x 128 tiles (odd widths, rows that end inside a tile row, a width of one partial tile) and the other
        // tile shapes the format allows. rows * cols is a multiple of 4 (the device keeps the weight 8-bit then).
        if self.filter.wants_kind(0x8000_0021) {
            let tile = |rows: usize, cols: usize| (128usize, 128usize, rows, cols);
            for (br, bc, rows, cols) in [
                tile(12, 77),
                tile(13, 256),
                tile(13, 300),
                tile(130, 520),
                (32, 32, 70, 200),
                (64, 16, 90, 300),
                (1, 128, 33, 384),
                (128, 32, 130, 256),
                (16, 1, 20, 13 * 4),
                (256, 64, 7 * 4, 200),
            ] {
                let kind = fllm_gguf_probe::fp8b_type_id(br, bc).expect("tile shape with a type id");
                for batch in [1, 2, 5] {
                    let r = self.linear(kind, rows, cols, batch, batch == 5);
                    self.record(
                        "linear fp8b",
                        &format!("{rows}x{cols} tiles {br}x{bc} batch {batch}{}", if batch == 5 { " +bias" } else { "" }),
                        r,
                    );
                }
            }
        }
    }

    fn embedding_cases(&mut self) {
        for (kind, cols) in [
            (0u32, 77),
            (1, 96),
            (30, 64),
            (2, 96),
            (8, 96),
            (12, 256),
            (14, 256),
            (20, 96),
        ] {
            if !self.filter.wants_kind(kind) {
                continue;
            }
            let r = self.embedding(kind, 300, cols);
            self.record(
                &format!("embedding {}", format_name(kind)),
                &format!("300x{cols} 13 ids"),
                r,
            );
        }
    }

    fn norm_cases(&mut self) {
        for (rows, cols) in [(1, 64), (13, 4097), (5, 77)] {
            let r = self.rms_norm(rows, cols);
            self.record("rms_norm", &format!("{rows}x{cols}"), r);
        }
        for rows in [1, 3, 17] {
            for cols in [256, 1024, 3584] {
                for groups in [1, 2, 4, 8] {
                    let r = self.rms_norm_grouped(rows, cols, groups);
                    self.record("rms_norm_grouped", &format!("{rows}x{cols} g={groups}"), r);
                }
            }
        }
    }

    fn rope_cases(&mut self) {
        for (rows, dim, rope_dim, neox) in [
            (1, 64, 64, true),
            (5, 80, 32, false),
            (13, 16, 16, true),
            (2, 128, 64, false),
        ] {
            let r = self.rope(rows, dim, rope_dim, neox);
            self.record(
                "rope",
                &format!(
                    "{rows} rows 4/2 heads d{dim} r{rope_dim} {}",
                    if neox { "neox" } else { "pairs" }
                ),
                r,
            );
        }
    }

    fn activation_cases(&mut self) {
        for count in [1usize, 77, 13 * 4099, 70_000 * 64] {
            for gelu in [false, true] {
                let r = self.glu(count, gelu);
                self.record(
                    if gelu { "geglu" } else { "swiglu" },
                    &format!("{count}"),
                    r,
                );
            }
            let r = self.residual(count);
            self.record("residual_add", &format!("{count} scale 0.5"), r);
        }
    }


    fn encoder_cases(&mut self) {
        // Full bidirectional attention over variable-length sequences
        for (batch_size, seq_len, heads, kv_heads, head_dim) in [
            (1, 16, 4, 4, 32),
            (2, 24, 6, 2, 64),
            (3, 12, 8, 8, 64),
        ] {
            let r = self.encoder_attention(batch_size, seq_len, heads, kv_heads, head_dim, None, None);
            self.record(
                "encoder_attn",
                &format!("b{batch_size} s{seq_len} h{heads}/{kv_heads} d{head_dim} full"),
                r,
            );
        }

        // Local sliding-window bidirectional attention (ModernBERT style)
        for (batch_size, seq_len, heads, kv_heads, head_dim, window) in [
            (1, 32, 4, 4, 64, 8),
            (2, 48, 6, 6, 64, 16),
        ] {
            let r = self.encoder_attention(batch_size, seq_len, heads, kv_heads, head_dim, Some(window), Some(&[0]));
            self.record(
                "encoder_attn",
                &format!("b{batch_size} s{seq_len} h{heads}/{kv_heads} d{head_dim} win{window}"),
                r,
            );
        }

        // Encoder GeGLU feed-forward
        for count in [64, 256, 1024] {
            let r = self.glu(count, true);
            self.record("encoder_geglu", &format!("{count}"), r);
        }
    }


    #[allow(clippy::too_many_arguments)]
    fn encoder_attention(
        &mut self,
        batch_size: usize,
        seq_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        window: Option<usize>,
        global_tokens: Option<&[i32]>,
    ) -> Result<Outcome, BackendError> {
        let total_tokens = batch_size * seq_len;
        let mut cu_seqlens = Vec::with_capacity(batch_size + 1);
        for b in 0..=batch_size {
            cu_seqlens.push((b * seq_len) as i32);
        }
        let q = self.rng.vector(total_tokens * heads * head_dim, 1.0);
        let k = self.rng.vector(total_tokens * kv_heads * head_dim, 1.0);
        let v = self.rng.vector(total_tokens * kv_heads * head_dim, 1.0);
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Reference CPU f64 computation
        let gqa = heads / kv_heads;
        let mut want = vec![0.0f64; total_tokens * heads * head_dim];
        for b in 0..batch_size {
            let start = cu_seqlens[b] as usize;
            let end = cu_seqlens[b + 1] as usize;
            let slen = end - start;
            for i in 0..slen {
                let tok_i = start + i;
                let i_global = global_tokens.is_some_and(|g| g.contains(&(tok_i as i32)));
                for h in 0..heads {
                    let kh = h / gqa;
                    let q_offset = tok_i * heads * head_dim + h * head_dim;
                    let q_slice = &q[q_offset..q_offset + head_dim];
                    let mut scores = Vec::with_capacity(slen);
                    let mut max_s = -1e38f64;
                    for j in 0..slen {
                        let tok_j = start + j;
                        let mut vis = true;
                        if let Some(win) = window {
                            if win > 0 && win < slen {
                                let j_global = global_tokens.is_some_and(|g| g.contains(&(tok_j as i32)));
                                if !i_global && !j_global {
                                    let dist = (i as isize - j as isize).unsigned_abs();
                                    if dist > (win / 2) {
                                        vis = false;
                                    }
                                }
                            }
                        }
                        if vis {
                            let k_offset = tok_j * kv_heads * head_dim + kh * head_dim;
                            let k_slice = &k[k_offset..k_offset + head_dim];
                            let dot: f64 = q_slice.iter().zip(k_slice).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum();
                            let s = dot * f64::from(scale);
                            scores.push((tok_j, s));
                            if s > max_s { max_s = s; }
                        }
                    }
                    let mut sum_e = 0.0f64;
                    let mut exps = Vec::with_capacity(scores.len());
                    for &(tok_j, s) in &scores {
                        let e = (s - max_s).exp();
                        exps.push((tok_j, e));
                        sum_e += e;
                    }
                    let inv_sum = if sum_e > 0.0 { 1.0 / sum_e } else { 0.0 };
                    let out_offset = tok_i * heads * head_dim + h * head_dim;
                    for &(tok_j, e) in &exps {
                        let w = e * inv_sum;
                        let v_offset = tok_j * kv_heads * head_dim + kh * head_dim;
                        let v_slice = &v[v_offset..v_offset + head_dim];
                        for d in 0..head_dim {
                            want[out_offset + d] += w * f64::from(v_slice[d]);
                        }
                    }
                }
            }
        }
        let err_scale: Vec<f64> = want.iter().map(|&x| x.abs().max(1.0)).collect();

        let op = |backend: &dyn Backend, stream: &mut dyn Stream, ms: &mut f64| -> Result<Vec<f32>, BackendError> {
            let q_buf = put(backend, stream, &q)?;
            let k_buf = put(backend, stream, &k)?;
            let v_buf = put(backend, stream, &v)?;
            let mut out_buf = backend.allocate_buffer(want.len() * 4)?;
            timed(ms, stream, |s| {
                s.attn_encoder(
                    &mut out_buf,
                    &q_buf,
                    &k_buf,
                    &v_buf,
                    &cu_seqlens,
                    batch_size,
                    total_tokens,
                    heads,
                    kv_heads,
                    head_dim,
                    scale,
                    window,
                    global_tokens,
                )
            })?;
            get(stream, &out_buf, want.len())
        };

        let (device, cpu, ms) = self.both(&op)?;
        let tol = gamma(head_dim + 1) + 1e-4;
        Ok(Outcome {
            device: worst(&device, &want, &err_scale),
            cpu: worst(&cpu, &want, &err_scale),
            tol,
            cpu_tol: tol,
            ms,
        })
    }

    fn kv_cases(&mut self) {
        // KV writes: f16 and q8, head widths with and without 32-value groups, 1/5/13 rows,
        // block size 5 so q8 layers start on odd 16-bit words.
        for kind in [1, 2] {
            for (dim, block_size) in [(16, 16), (32, 16), (64, 16), (80, 5), (128, 16)] {
                for rows in [1, 5, 13] {
                    let r = self.kv_write(kind, dim, block_size, rows);
                    self.record(
                        &format!("kv_cache_write {}", if kind == 1 { "f16" } else { "q8" }),
                        &format!("{rows} rows 2x{dim} block {block_size}"),
                        r,
                    );
                }
            }
        }
    }

    fn attention_run(&mut self) {
        for case in attention_cases() {
            let label = format!("attention {}", if case.kind == 1 { "f16" } else { "q8" });
            let shape = case.describe();
            let r = self.attention(&case);
            self.record(&label, &shape, r);
        }
        self.run_kv_formats();
    }

    fn linear(
        &mut self,
        kind: u32,
        rows: usize,
        cols: usize,
        batch: usize,
        with_bias: bool,
    ) -> Result<Outcome, BackendError> {
        let weights = random_kind(&mut self.rng, kind, rows, cols);
        self.linear_with(kind, weights, rows, cols, batch, with_bias)
    }

    /// [`Self::linear`] for weights the caller made (a format whose random generator has parameters the case wants to choose).
    fn linear_with(
        &mut self,
        kind: u32,
        weights: Vec<u8>,
        rows: usize,
        cols: usize,
        batch: usize,
        with_bias: bool,
    ) -> Result<Outcome, BackendError> {
        self.claim = Some(kind);
        let x = self.rng.vector(batch * cols, 1.0);
        let bias = with_bias.then(|| self.rng.vector(rows, 1.0));
        let w = crate::compute::dequantize(&weights, kind, rows * cols)
            .map_err(BackendError::DriverError)?;
        let mut want = vec![0.0; batch * rows];
        let mut scale = vec![0.0; batch * rows];
        for b in 0..batch {
            for r in 0..rows {
                let (mut sum, mut mag) = (0.0f64, 0.0f64);
                for (&xv, &wv) in x[b * cols..(b + 1) * cols]
                    .iter()
                    .zip(&w[r * cols..(r + 1) * cols])
                {
                    let term = f64::from(xv) * f64::from(wv);
                    sum += term;
                    mag += term.abs();
                }
                if let Some(bias) = &bias {
                    sum += f64::from(bias[r]);
                    mag += f64::from(bias[r]).abs();
                }
                want[b * rows + r] = sum;
                scale[b * rows + r] = mag;
            }
        }
        let op = |backend: &dyn Backend,
                  stream: &mut dyn Stream,
                  ms: &mut f64|
         -> Result<Vec<f32>, BackendError> {
            // NVFP4 weights carry a per-tensor factor; the scaled upload is the one that may pick the Marlin kernel
            let weight = backend.upload_weight_scaled(
                "selftest",
                &weights,
                QuantType::from_gguf_type_id(kind),
                rows,
                cols,
                bias.as_deref(),
                1.0,
            )?;
            let xb = put(backend, stream, &x)?;
            let mut out = backend.allocate_buffer(batch * rows * 4)?;
            timed(ms, stream, |s| {
                s.quantized_linear(&mut out, &xb, &weight, weight.bias.as_ref())
            })?;
            get(stream, &out, batch * rows)
        };
        let (device, cpu, ms) = self.both(&op)?;
        // A dot product of n products plus a bias: |error| <= gamma(n + 1) * sum |terms| in any
        // order (FMA only tightens it); two more units cover a decoder that rounds an affine
        // weight (d * q - m) differently from the CPU dequantizer.
        let tol = gamma(cols + 3);
        // Formats a backend only reaches by re-quantizing them at upload (CUDA: Q5_K for wide rows of
        // formats without a decoder) carry that conversion error: judge the device against 2 % of
        // the error scale; the CPU backend is still held to the exact tolerance.
        let supported = self.device.capabilities().supports_quant(QuantType::from_gguf_type_id(kind));
        let converted = !supported
            && ((cols.is_multiple_of(256) && !matches!(kind, 20 | 23 | 39)) // 20, 23, 39 run natively through the shared 16-entry grid kernels
                || fllm_gguf_probe::bnb::is_bnb(kind)) // expanded to 16-bit values on upload by backends without a decoder
            && self.device.name() != "CPU";
        // block-wise FP8 on a device without a kernel for it is expanded to binary16 at upload: the weights are rounded (2^-11 relative)
        // (or a shape the device kernels cannot take: it checks that by uploading the weight, see `kept_8bit`)
        let f16_expanded = (fllm_gguf_probe::fp8b_type(kind) || fllm_gguf_probe::fp8m_type(kind))
            && (!supported || !self.kept_8bit(kind, &weights, rows, cols))
            && self.device.name() != "CPU";
        let device_tol = if converted {
            0.02
        } else if f16_expanded {
            tol.max(Self::FAST_TOL)
        } else {
            tol
        }
        .max(self.npu_requant_tol(kind));
        Ok(Outcome {
            device: worst(&device, &want, &scale),
            cpu: worst(&cpu, &want, &scale),
            tol: device_tol,
            cpu_tol: tol,
            ms,
        })
    }

    /// Whether the device keeps the FP8 weight 8-bit (any layout of its own) instead of expanding it to a 16 or 32-bit float format.
    fn kept_8bit(&self, kind: u32, weights: &[u8], rows: usize, cols: usize) -> bool {
        self.device
            .upload_weight("selftest", weights, QuantType::from_gguf_type_id(kind), rows, cols, None)
            .is_ok_and(|w| !matches!(w.quant_type, QuantType::F16 | QuantType::BF16 | QuantType::F32))
    }

    /// NVFP4 W4A4 against the CPU simulation of vLLM's recipe (see `nvfp4_activation_reference`). `factor` scales the checkpoint-style
    /// input_scale (amax / 2688 of the activations) to test saturation; `mode`: 0 = plain result, 1 = added to a residual
    /// (`quantized_linear_add`), 2 = FP16 result (`linear_f16_out`), 3 = after an RMSNorm.
    ///
    /// The device side runs twice (fresh stream, fresh upload of the weight, fresh buffers) and must give bit-identical results: a difference is a
    /// race or stale state, not rounding. Before each run the trailer of the uploaded weight is read back and compared with what the loader wrote
    /// (a weight whose upload is not complete when the first GEMM reads it shows up here). A result outside the tolerance comes back as an error that
    /// names the worst element, the number of elements and tokens above the tolerance and the mean error, so that "one token flipped a rounding"
    /// (a few tokens, the worst error just above the tolerance) can be told from a systematic error (every token).
    fn nvfp4_w4a4(&mut self, rows: usize, cols: usize, batch: usize, factor: f32, mode: u8) -> Result<Outcome, BackendError> {
        self.claim = None;
        if self.device.name() == "CPU" {
            return Err(BackendError::UnsupportedOperation("the CPU backend does not quantize activations".into()));
        }
        let weights = random_kind(&mut self.rng, 40, rows, cols);
        let weight_scale = 0.37f32;
        // tokens of very different magnitude: a block scale that is not helped by the per-tensor factor loses most of the small ones
        let mut x = self.rng.vector(batch * cols, 1.0);
        for b in 0..batch {
            let gain = [1.0f32, 1e-2, 3.0, 1e-3, 0.3][b % 5];
            for v in &mut x[b * cols..(b + 1) * cols] {
                *v *= gain;
            }
        }
        // mode 3: the GEMM reads the output of an RMSNorm (FLLM_FUSE_NVFP4_NORM fuses the two); the CPU reference normalizes in f32 like the kernels
        let x_raw = x.clone();
        let norm_weight: Vec<f32> = self.rng.vector(cols, 1.0).iter().map(|v| 0.5 + v.abs()).collect();
        let eps = 1e-5f32;
        if mode == 3 {
            for row in x.chunks_mut(cols) {
                let mean = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
                let inv = 1.0 / (mean + eps).sqrt();
                for (v, w) in row.iter_mut().zip(&norm_weight) {
                    *v *= inv * w;
                }
            }
        }
        let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let input_scale = amax / 2688.0 * factor;
        let gs = 1.0 / input_scale;
        let grid = nvfp4_weight_values(&weights, rows, cols);
        let residual: Vec<f32> = (0..batch * rows).map(|i| ((i % 17) as f32 - 8.0) * 0.125).collect();
        let residual_scale = if mode == 1 { 0.75f32 } else { 1.0 };
        let mut want = vec![0.0f64; batch * rows];
        let mut scale = vec![0.0f64; batch * rows];
        let alpha = f64::from(weight_scale) / f64::from(gs);
        for b in 0..batch {
            let xq = nvfp4_activation_reference(&x[b * cols..(b + 1) * cols], gs);
            for r in 0..rows {
                let (mut sum, mut mag) = (0.0f64, 0.0f64);
                for (&a, &w) in xq.iter().zip(&grid[r * cols..(r + 1) * cols]) {
                    let term = a * f64::from(w);
                    sum += term;
                    mag += term.abs();
                }
                let base = if mode == 1 { f64::from(residual[b * rows + r]) } else { 0.0 };
                want[b * rows + r] = base + f64::from(residual_scale) * alpha * sum;
                scale[b * rows + r] = base.abs() + f64::from(residual_scale) * alpha * mag;
            }
        }

        let device = self.device;
        let pass = || -> Result<(Vec<f32>, f64), BackendError> {
            let mut stream = device.create_stream()?;
            crate::backend::set_next_nvfp4_input_scale(input_scale);
            let uploaded = device.upload_weight_scaled("selftest", &weights, QuantType::NvFp4, rows, cols, None, weight_scale);
            crate::backend::set_next_nvfp4_input_scale(0.0);
            let weight = uploaded?;
            if weight.quant_type != QuantType::NvFp4 {
                return Err(BackendError::UnsupportedOperation("the weights are not kept as NVFP4 on this device (no FP4 tensor cores)".into()));
            }
            stream.synchronize()?;
            // the trailer behind the scales: {1 / input_scale, weight_scale_2 * input_scale, weight_scale_2, 1}
            let trailer_at = nvfp4_trailer_offset(rows, cols);
            if weight.buffer.size < trailer_at + 16 {
                // a backend that keeps NVFP4 in its own planar layout (Vulkan) has no activation-quantizing W4A4 path: nothing to compare
                return Err(BackendError::UnsupportedOperation(format!(
                    "NVFP4 W4A4 (this device stores NVFP4 weights in {} bytes without the {trailer_at}-byte tensor-core layout and trailer)",
                    weight.buffer.size
                )));
            }
            let mut image = vec![0u8; weight.buffer.size];
            stream.download(&mut image, &weight.buffer)?;
            let got: Vec<f32> = image[trailer_at..trailer_at + 16].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
            let expected = [1.0 / input_scale, weight_scale * input_scale, weight_scale, 1.0];
            if got != expected {
                return Err(BackendError::DriverError(format!(
                    "the weight trailer on the device is {got:?}, the loader wrote {expected:?} (upload not complete or overwritten before the GEMM)"
                )));
            }
            let xb = put(device, stream.as_mut(), &x)?;
            let mut ms = 0.0;
            let have: Vec<f32> = match mode {
                1 => {
                    let mut res = put(device, stream.as_mut(), &residual)?;
                    let mut taken = false;
                    timed(&mut ms, stream.as_mut(), |s| {
                        taken = s.quantized_linear_add(&mut res, &xb, &weight, None, residual_scale)?;
                        Ok(())
                    })?;
                    if !taken {
                        return Err(BackendError::UnsupportedOperation("the accumulating NVFP4 GEMM is not available for this shape".into()));
                    }
                    get(stream.as_mut(), &res, batch * rows)?
                }
                2 => {
                    // linear_f16_out writes through quantized_linear, which wants a buffer sized for FP32 results (the decoder's buffers are)
                    let mut out16 = device.allocate_buffer(batch * rows * 4)?;
                    let mut taken = false;
                    timed(&mut ms, stream.as_mut(), |s| {
                        taken = s.linear_f16_out(&mut out16, &xb, &weight)?;
                        Ok(())
                    })?;
                    if !taken {
                        return Err(BackendError::UnsupportedOperation("the FP16-result NVFP4 GEMM is not available for this shape".into()));
                    }
                    let mut bytes = vec![0u8; batch * rows * 2];
                    stream.download(&mut bytes, &out16)?;
                    bytes.as_chunks::<2>().0.iter().map(|c| f16::from_le_bytes(*c).to_f32()).collect()
                }
                3 => {
                    let xr = put(device, stream.as_mut(), &x_raw)?;
                    let wb = put(device, stream.as_mut(), &norm_weight)?;
                    let mut normed = device.allocate_buffer(batch * cols * 4)?;
                    let mut out = device.allocate_buffer(batch * rows * 4)?;
                    timed(&mut ms, stream.as_mut(), |s| {
                        s.rms_norm(&mut normed, &xr, &wb, eps)?;
                        s.quantized_linear(&mut out, &normed, &weight, None)
                    })?;
                    get(stream.as_mut(), &out, batch * rows)?
                }
                _ => {
                    let mut out = device.allocate_buffer(batch * rows * 4)?;
                    timed(&mut ms, stream.as_mut(), |s| s.quantized_linear(&mut out, &xb, &weight, None))?;
                    get(stream.as_mut(), &out, batch * rows)?
                }
            };
            Ok((have, ms))
        };
        let (have, ms) = pass()?;
        let (again, _) = pass()?;
        if let Some(i) = have.iter().zip(&again).position(|(a, b)| a.to_bits() != b.to_bits()) {
            let differing = have.iter().zip(&again).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            return Err(BackendError::DriverError(format!(
                "not deterministic: {differing} of {} results differ between two identical runs, first at element {i} (token {}, row {}): {} then {}",
                have.len(),
                i / rows,
                i % rows,
                have[i],
                again[i]
            )));
        }

        // FP32 accumulation of the same quantized operands (a different summation order) and, for mode 2, one FP16 rounding of the result
        let tol = if mode == 2 { 4e-3 } else { 2e-3 };
        let mean_tol = 1e-4;
        if have.len() != want.len() {
            return Err(BackendError::DriverError(format!("{} results for {} expected", have.len(), want.len())));
        }
        let (mut max_err, mut max_at, mut sum_err, mut above, mut above_1e3) = (0.0f64, 0usize, 0.0f64, 0usize, 0usize);
        let mut tokens_above = vec![false; batch];
        for (i, ((&h, &w), &s)) in have.iter().zip(&want).zip(&scale).enumerate() {
            let e = if h.is_nan() { f64::INFINITY } else { (f64::from(h) - w).abs() / s.max(1e-30) };
            sum_err += e;
            if e > max_err {
                (max_err, max_at) = (e, i);
            }
            if e > tol {
                above += 1;
                tokens_above[i / rows] = true;
            }
            if e > 1e-3 {
                above_1e3 += 1;
            }
        }
        let mean_err = sum_err / have.len() as f64;
        if max_err > tol || mean_err > mean_tol {
            let tokens = tokens_above.iter().filter(|&&t| t).count();
            return Err(BackendError::DriverError(format!(
                "max error {max_err:.3e} (tolerance {tol:.0e}) at element {max_at} (token {}, row {}): got {}, expected {:.6e}, error scale {:.3e}; mean error {mean_err:.2e} (tolerance {mean_tol:.0e}); \
                 {above} of {} elements above the tolerance in {tokens} of {batch} tokens, {above_1e3} above 1e-3; the two device runs were bit-identical",
                max_at / rows,
                max_at % rows,
                have[max_at],
                want[max_at],
                scale[max_at],
                have.len()
            )));
        }
        Ok(Outcome { device: max_err, cpu: 0.0, tol, cpu_tol: tol, ms })
    }

    /// Two weights of different formats read the same activations back to back (q, k and v of a layer): the activations that
    /// the first GEMM converted to FP16 must not be reused by a kernel that expects another layout.
    fn shared_input(&mut self, kinds: [u32; 2], rows: usize, cols: usize, batch: usize) -> Result<Outcome, BackendError> {
        let weights = [random_kind(&mut self.rng, kinds[0], rows, cols), random_kind(&mut self.rng, kinds[1], rows, cols)];
        let x = self.rng.vector(batch * cols, 1.0);
        let mut want = Vec::new();
        let mut scale = Vec::new();
        for (w, &kind) in weights.iter().zip(&kinds) {
            let w = crate::compute::dequantize(w, kind, rows * cols).map_err(BackendError::DriverError)?;
            for b in 0..batch {
                for r in 0..rows {
                    let (mut sum, mut mag) = (0.0f64, 0.0f64);
                    for (&xv, &wv) in x[b * cols..(b + 1) * cols].iter().zip(&w[r * cols..(r + 1) * cols]) {
                        let term = f64::from(xv) * f64::from(wv);
                        sum += term;
                        mag += term.abs();
                    }
                    want.push(sum);
                    scale.push(mag);
                }
            }
        }
        let op = |backend: &dyn Backend, stream: &mut dyn Stream, ms: &mut f64| -> Result<Vec<f32>, BackendError> {
            let xb = put(backend, stream, &x)?;
            let mut result = Vec::new();
            // all weights and results stay alive until the end: a freed allocation could be handed out again at the same address
            let mut live = Vec::new();
            for (w, &kind) in weights.iter().zip(&kinds) {
                let weight = backend.upload_weight("selftest", w, QuantType::from_gguf_type_id(kind), rows, cols, None)?;
                // the host-to-device copy of a weight is only staged when upload_weight returns: wait for it before computing
                stream.synchronize()?;
                let mut out = backend.allocate_buffer(batch * rows * 4)?;
                // no download in between: that would reset the cached activations, exactly what this case is about
                timed(ms, stream, |s| s.quantized_linear(&mut out, &xb, &weight, None))?;
                live.push((weight, out));
            }
            for (_, out) in &live {
                result.extend(get(stream, out, batch * rows)?);
            }
            drop(live);
            Ok(result)
        };
        let (device, cpu, ms) = self.both(&op)?;
        let half = want.len() / 2;
        let halves = [
            worst(&device[..half], &want[..half], &scale[..half]),
            worst(&device[half..], &want[half..], &scale[half..]),
        ];
        let shared_tol = Self::FAST_TOL.max(self.npu_requant_tol(kinds[0])).max(self.npu_requant_tol(kinds[1]));
        if halves.iter().any(|h| h.is_nan() || h.abs() > shared_tol) {
            let nan = device.iter().filter(|v| v.is_nan()).count();
            return Err(BackendError::DriverError(format!(
                "first weight {:.2e}, second weight {:.2e}, {nan} NaN of {}",
                halves[0],
                halves[1],
                device.len()
            )));
        }
        Ok(Outcome {
            device: halves[0].max(halves[1]),
            cpu: worst(&cpu, &want, &scale),
            tol: shared_tol,
            cpu_tol: gamma(cols + 3),
            ms,
        })
    }

    /// Bound for the formats the NPU provider re-quantizes before its fp16 matrix multiplication (OPENVINO.md: Q4_0, Q8_0 and f16 / f32 weights are
    /// carried over exactly, every other format becomes symmetric int4, or int8 above 4.5 bits, in groups of 128). That conversion error is the
    /// design of the backend, not a kernel defect, and shows in the fast pass only (projections of 64 or more rows). 0: no extra allowance. The native
    /// AMD XDNA engine dequantizes every format exactly into bf16, so it gets none (its error is the bf16 rounding of activations and weights).
    fn npu_requant_tol(&self, kind: u32) -> f64 {
        if self.device.name() != "npu" || self.pass != "fast" || self.device.capabilities().name.contains("native XDNA") {
            return 0.0;
        }
        match kind {
            0 | 1 | 2 | 8 | 30 => 0.0,
            6 | 7 | 13 | 14 => 1e-2,
            _ => 3e-2,
        }
    }

    /// Greedy sampling on the device: the index of the largest logit of every row, the first one on a tie, NaN ignored.
    fn argmax(&mut self, rows: usize, vocab: usize) -> Result<Outcome, BackendError> {
        let mut logits = self.rng.vector(rows * vocab, 4.0);
        for r in 0..rows {
            let row = &mut logits[r * vocab..(r + 1) * vocab];
            let peak = 9.0f32 + r as f32;
            // two equal maxima: the earlier index must win; one NaN that must not
            row[vocab / 3 + r] = peak;
            row[vocab - 1 - r] = peak;
            row[r] = f32::NAN;
        }
        let want: Vec<i32> = (0..rows).map(|r| crate::sampling::argmax(&logits[r * vocab..(r + 1) * vocab]) as i32).collect();
        let mut stream = self.device.create_stream()?;
        let buf = put(self.device, stream.as_mut(), &logits)?;
        let start = Instant::now();
        let got = stream.argmax_rows(&buf, rows, vocab)?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        let wrong = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        Ok(Outcome { device: if wrong == 0 && got.len() == want.len() { 0.0 } else { f64::INFINITY }, cpu: 0.0, tol: 0.0, cpu_tol: 1.0, ms })
    }

    fn embedding(&mut self, kind: u32, rows: usize, cols: usize) -> Result<Outcome, BackendError> {
        self.claim = Some(kind);
        let weights = random_kind(&mut self.rng, kind, rows, cols);
        let mut ids: Vec<i64> = (0..13)
            .map(|_| (self.rng.next() % rows as u64) as i64)
            .collect();
        ids[0] = 0;
        ids[1] = rows as i64 - 1;
        ids[2] = ids[3];
        let table = crate::compute::dequantize(&weights, kind, rows * cols)
            .map_err(BackendError::DriverError)?;
        let want: Vec<f64> = ids
            .iter()
            .flat_map(|&id| {
                table[id as usize * cols..(id as usize + 1) * cols]
                    .iter()
                    .map(|&v| f64::from(v))
            })
            .collect();
        let peak = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let scale = vec![peak; want.len()];
        let op = |backend: &dyn Backend,
                  stream: &mut dyn Stream,
                  ms: &mut f64|
         -> Result<Vec<f32>, BackendError> {
            let weight = backend.upload_weight(
                "selftest",
                &weights,
                QuantType::from_gguf_type_id(kind),
                rows,
                cols,
                None,
            )?;
            let mut out = backend.allocate_buffer(ids.len() * cols * 4)?;
            timed(ms, stream, |s| s.embedding_lookup(&mut out, &ids, &weight))?;
            get(stream, &out, ids.len() * cols)
        };
        let (device, cpu, ms) = self.both(&op)?;
        // Decoding is exact except one rounding in affine formats, which both sides perform
        // in the same order: allow two units of the largest value.
        let tol = 2.0 * U;
        Ok(Outcome {
            device: worst(&device, &want, &scale),
            cpu: worst(&cpu, &want, &scale),
            tol,
            cpu_tol: tol,
            ms,
        })
    }

    fn rms_norm(&mut self, rows: usize, cols: usize) -> Result<Outcome, BackendError> {
        let x = self.rng.vector(rows * cols, 2.0);
        let w = self.rng.vector(cols, 1.0);
        let eps = 1e-6f32;
        let mut want = vec![0.0; rows * cols];
        let mut scale = vec![0.0; rows * cols];
        for r in 0..rows {
            let row = &x[r * cols..(r + 1) * cols];
            let ms: f64 = row
                .iter()
                .map(|&v| f64::from(v) * f64::from(v))
                .sum::<f64>()
                / cols as f64;
            let inv = 1.0 / (ms + f64::from(eps)).sqrt();
            for c in 0..cols {
                let v = f64::from(row[c]) * inv * f64::from(w[c]);
                want[r * cols + c] = v;
                scale[r * cols + c] = v.abs();
            }
        }
        let op = |backend: &dyn Backend,
                  stream: &mut dyn Stream,
                  ms: &mut f64|
         -> Result<Vec<f32>, BackendError> {
            let xb = put(backend, stream, &x)?;
            let wb = put(backend, stream, &w)?;
            let mut out = backend.allocate_buffer(rows * cols * 4)?;
            timed(ms, stream, |s| s.rms_norm(&mut out, &xb, &wb, eps))?;
            get(stream, &out, rows * cols)
        };
        let (device, cpu, ms) = self.both(&op)?;
        // Sum of squares (positive terms): relative gamma(n), halved by the square root; the
        // mean, eps, 1/sqrt (WGSL: 2.5 ULP division of a 2 ULP inverseSqrt) and two products
        // stay below 32 units.
        let tol = gamma(cols) / 2.0 + 32.0 * U;
        Ok(Outcome {
            device: worst(&device, &want, &scale),
            cpu: worst(&cpu, &want, &scale),
            tol,
            cpu_tol: tol,
            ms,
        })
    }

    fn rms_norm_grouped(
        &mut self,
        rows: usize,
        cols: usize,
        groups: usize,
    ) -> Result<Outcome, BackendError> {
        let x = self.rng.vector(rows * cols, 2.0);
        let w = self.rng.vector(cols, 1.0);
        let eps = 1e-6f32;
        let mut want = vec![0.0; rows * cols];
        let mut scale = vec![0.0; rows * cols];
        let group_size = cols / groups;
        for r in 0..rows {
            let row = &x[r * cols..(r + 1) * cols];
            for g in 0..groups {
                let grp = &row[g * group_size..(g + 1) * group_size];
                let ms: f64 = grp
                    .iter()
                    .map(|&v| f64::from(v) * f64::from(v))
                    .sum::<f64>()
                    / group_size as f64;
                let inv = 1.0 / (ms + f64::from(eps)).sqrt();
                for i in 0..group_size {
                    let idx = g * group_size + i;
                    let v = f64::from(row[idx]) * inv * f64::from(w[idx]);
                    want[r * cols + idx] = v;
                    scale[r * cols + idx] = v.abs();
                }
            }
        }
        let op = |backend: &dyn Backend,
                  stream: &mut dyn Stream,
                  ms: &mut f64|
         -> Result<Vec<f32>, BackendError> {
            let xb = put(backend, stream, &x)?;
            let wb = put(backend, stream, &w)?;
            let mut out = backend.allocate_buffer(rows * cols * 4)?;
            timed(ms, stream, |s| s.rms_norm_grouped(&mut out, &xb, &wb, eps, groups))?;
            get(stream, &out, rows * cols)
        };
        let (device, cpu, ms) = self.both(&op)?;
        let tol = gamma(group_size) / 2.0 + 32.0 * U;
        Ok(Outcome {
            device: worst(&device, &want, &scale),
            cpu: worst(&cpu, &want, &scale),
            tol,
            cpu_tol: tol,
            ms,
        })
    }

    fn rope(
        &mut self,
        rows: usize,
        dim: usize,
        rope_dim: usize,
        neox: bool,
    ) -> Result<Outcome, BackendError> {
        let (heads, kv_heads, max_pos) = (4usize, 2usize, 64usize);
        let half = rope_dim / 2;
        let mut cos = Vec::with_capacity(max_pos * half);
        let mut sin = Vec::with_capacity(max_pos * half);
        for p in 0..max_pos {
            for d in 0..half {
                let angle = p as f64 / 10_000f64.powf(2.0 * d as f64 / rope_dim as f64);
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
        let positions: Vec<usize> = (0..rows)
            .map(|_| (self.rng.next() % max_pos as u64) as usize)
            .collect();
        let q = self.rng.vector(rows * heads * dim, 1.0);
        let k = self.rng.vector(rows * kv_heads * dim, 1.0);
        let reference = |x: &[f32], nh: usize| {
            let mut want: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
            let mut scale: Vec<f64> = want.iter().map(|v| v.abs()).collect();
            for (r, &p) in positions.iter().enumerate() {
                for h in 0..nh {
                    let base = (r * nh + h) * dim;
                    for d in 0..half {
                        let (ia, ib) = if neox {
                            (d, d + half)
                        } else {
                            (2 * d, 2 * d + 1)
                        };
                        let (a, b) = (f64::from(x[base + ia]), f64::from(x[base + ib]));
                        let (c, s) = (f64::from(cos[p * half + d]), f64::from(sin[p * half + d]));
                        want[base + ia] = a * c - b * s;
                        want[base + ib] = b * c + a * s;
                        scale[base + ia] = (a * c).abs() + (b * s).abs();
                        scale[base + ib] = (b * c).abs() + (a * s).abs();
                    }
                }
            }
            (want, scale)
        };
        let (mut want, mut scale) = reference(&q, heads);
        let (kw, ks) = reference(&k, kv_heads);
        want.extend(kw);
        scale.extend(ks);
        let params = RopeParams {
            heads,
            kv_heads,
            dim,
            rope_dim,
            neox,
            cos: Arc::new(cos.clone()),
            sin: Arc::new(sin.clone()),
        };
        let op = |backend: &dyn Backend,
                  stream: &mut dyn Stream,
                  ms: &mut f64|
         -> Result<Vec<f32>, BackendError> {
            let mut qb = put(backend, stream, &q)?;
            let mut kb = put(backend, stream, &k)?;
            timed(ms, stream, |s| {
                s.rope(&mut qb, &mut kb, &positions, &params)
            })?;
            let mut out = get(stream, &qb, q.len())?;
            out.extend(get(stream, &kb, k.len())?);
            Ok(out)
        };
        let (device, cpu, ms) = self.both(&op)?;
        // Two products and one sum: gamma(2) of the absolute terms; untouched dims are exact.
        let tol = gamma(2) + U;
        Ok(Outcome {
            device: worst(&device, &want, &scale),
            cpu: worst(&cpu, &want, &scale),
            tol,
            cpu_tol: tol,
            ms,
        })
    }

    fn glu(&mut self, count: usize, gelu: bool) -> Result<Outcome, BackendError> {
        let g = self.rng.vector(count, 4.0);
        let u = self.rng.vector(count, 2.0);
        let mut want = Vec::with_capacity(count);
        let mut scale = Vec::with_capacity(count);
        for (&g, &u) in g.iter().zip(&u) {
            let (g, u) = (f64::from(g), f64::from(u));
            if gelu {
                let t = f64::from(0.797_884_6f32) * (g + f64::from(0.044_715f32) * g * g * g);
                want.push(0.5 * g * (1.0 + t.tanh()) * u);
                scale.push((g * u).abs());
            } else {
                let v = g / (1.0 + (-g).exp()) * u;
                want.push(v);
                scale.push(v.abs());
            }
        }
        let op = |backend: &dyn Backend,
                  stream: &mut dyn Stream,
                  ms: &mut f64|
         -> Result<Vec<f32>, BackendError> {
            let gb = put(backend, stream, &g)?;
            let ub = put(backend, stream, &u)?;
            let mut out = backend.allocate_buffer(count * 4)?;
            timed(ms, stream, |s| {
                if gelu {
                    s.geglu(&mut out, &gb, &ub)
                } else {
                    s.swiglu(&mut out, &gb, &ub)
                }
            })?;
            get(stream, &out, count)
        };
        let (device, cpu, ms) = self.both(&op)?;
        let tol = if gelu {
            // Relative to |g u|: tanh is accurate to a few ULP of 1 and its argument (|t| <= 6)
            // carries about 4 roundings, each passed through tanh' <= 1.
            64.0 * U + 4.0 * 6.0 * U + 16.0 * U
        } else {
            // Relative to the result: exp is accurate to 3 + 2|x| ULP (|x| <= 4), division to
            // 2.5 ULP, plus one sum and one product.
            2.0 * (3.0 + 2.0 * 4.0) * U * 2.0 + 5.0 * U + 4.0 * U
        };
        Ok(Outcome {
            device: worst(&device, &want, &scale),
            cpu: worst(&cpu, &want, &scale),
            tol,
            cpu_tol: tol,
            ms,
        })
    }

    fn residual(&mut self, count: usize) -> Result<Outcome, BackendError> {
        let x = self.rng.vector(count, 1.0);
        let r = self.rng.vector(count, 1.0);
        let s = 0.5f32;
        let want: Vec<f64> = x
            .iter()
            .zip(&r)
            .map(|(&x, &r)| f64::from(x) + f64::from(r) * f64::from(s))
            .collect();
        let scale: Vec<f64> = x
            .iter()
            .zip(&r)
            .map(|(&x, &r)| f64::from(x).abs() + (f64::from(r) * f64::from(s)).abs())
            .collect();
        let op = |backend: &dyn Backend,
                  stream: &mut dyn Stream,
                  ms: &mut f64|
         -> Result<Vec<f32>, BackendError> {
            let mut xb = put(backend, stream, &x)?;
            let rb = put(backend, stream, &r)?;
            timed(ms, stream, |st| st.residual_add(&mut xb, &rb, s))?;
            get(stream, &xb, count)
        };
        let (device, cpu, ms) = self.both(&op)?;
        // One product and one sum.
        let tol = gamma(2);
        Ok(Outcome {
            device: worst(&device, &want, &scale),
            cpu: worst(&cpu, &want, &scale),
            tol,
            cpu_tol: tol,
            ms,
        })
    }

    fn kv_write(
        &mut self,
        kind: i32,
        dim: usize,
        block_size: usize,
        rows: usize,
    ) -> Result<Outcome, BackendError> {
        let (layers, layer, kv_heads) = (3usize, 1usize, 2usize);
        let pool_blocks = 33usize;
        let slots_total = pool_blocks * block_size;
        let mut slots: Vec<usize> = Vec::new();
        while slots.len() < rows {
            let s = (self.rng.next() % slots_total as u64) as usize;
            if !slots.contains(&s) {
                slots.push(s);
            }
        }
        let kv = kv_heads * dim;
        let mut k = self.rng.vector(rows * kv, 2.0);
        let v = self.rng.vector(rows * kv, 2.0);
        // An all-zero group exercises the d == 0 branch of the q8 encoder.
        k[..dim.min(32)].fill(0.0);
        let words = kv_layer_words(kind, slots_total, kv_heads, dim);
        let op = |backend: &dyn Backend,
                  stream: &mut dyn Stream,
                  ms: &mut f64|
         -> Result<Vec<f32>, BackendError> {
            let cache = backend.allocate_kv_cache(
                layers,
                pool_blocks,
                block_size,
                kv_heads,
                dim,
                dim,
                kind,
            )?;
            let kb = put(backend, stream, &k)?;
            let vb = put(backend, stream, &v)?;
            timed(ms, stream, |s| {
                s.kv_cache_write(&cache, layer, &slots, &kb, &vb)
            })?;
            let mut out = Vec::new();
            for pool in [&cache.key_pool, &cache.value_pool] {
                let mut bytes = vec![0u8; layers * words * 2];
                stream.download(&mut bytes, pool)?;
                out.extend(
                    bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|c| f32::from(u16::from_le_bytes(*c))),
                );
            }
            Ok(out)
        };
        let (device, cpu, ms) = self.both(&op)?;
        let group = kv_quant_group(dim);
        let evaluate = |pools: &[f32]| -> f64 {
            let all = words_of(pools);
            let (kw, vw) = all.split_at(layers * words);
            let mut err = 0.0f64;
            // where the largest error sits (pool, row, column, value found, value expected): named in the failure message
            let mut at = (0usize, 0usize, 0usize, 0.0f64, 0.0f64);
            for (pool_index, (pool, src)) in [(kw, &k), (vw, &v)].into_iter().enumerate() {
                let view = PoolView {
                    words: pool,
                    layer_words: words,
                    slots: slots_total,
                    kv,
                    group,
                    q8: kind == 2,
                    layout: None,
                };
                for l in 0..layers {
                    let written: &[usize] = if l == layer { &slots } else { &[] };
                    if !view.untouched_is_zero(l, written) {
                        return f64::INFINITY;
                    }
                }
                for (r, &slot) in slots.iter().enumerate() {
                    for c in 0..kv {
                        let x = f64::from(src[r * kv + c]);
                        let have = view.value(layer, slot, c);
                        let e = if kind == 2 {
                            // In units of the group's exact step amax / 127.
                            let g = c / group;
                            let amax = src[r * kv + g * group..r * kv + (g + 1) * group]
                                .iter()
                                .fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
                            if amax == 0.0 {
                                have.abs() * 1e30
                            } else {
                                (have - x).abs() / (amax / 127.0)
                            }
                        } else {
                            (have - x).abs() / x.abs().max(1.0 / 16384.0)
                        };
                        if e.is_nan() {
                            return log_worst(WorstAt { err: f64::INFINITY, index: (pool_index * rows + r) * kv + c, have: have as f32, want: x, scale: 1.0, len: 2 * rows * kv, want_len: 2 * rows * kv });
                        }
                        if e > err {
                            err = e;
                            at = (pool_index, r, c, have, x);
                        }
                    }
                }
            }
            log_worst(WorstAt { err, index: (at.0 * rows + at.1) * kv + at.2, have: at.3 as f32, want: at.4, scale: 1.0, len: 2 * rows * kv, want_len: 2 * rows * kv })
        };
        // f16: either neighbour of x is a correct conversion, 2^-10 relative. q8: rounding to
        // the nearest step (0.5) plus the f16 scale (127 steps x 2^-10 = 0.124), plus one
        // step because WGSL division may move a value lying on a half step to the other side.
        let tol = if kind == 2 { 1.7 } else { 1.0 / 1024.0 };
        Ok(Outcome {
            device: evaluate(&device),
            cpu: evaluate(&cpu),
            tol,
            cpu_tol: tol,
            ms,
        })
    }

    fn batch(&mut self, case: &AttnCase) -> Batch {
        let blocks_needed: usize = case
            .sequences
            .iter()
            .map(|&n| n.div_ceil(case.block_size))
            .sum();
        assert!(
            blocks_needed <= case.pool_blocks,
            "attention case needs a larger pool"
        );
        // A random permutation of the pool hands out distinct blocks.
        let mut free: Vec<usize> = (0..case.pool_blocks).collect();
        for i in (1..free.len()).rev() {
            let j = (self.rng.next() % (i as u64 + 1)) as usize;
            free.swap(i, j);
        }
        let shared: Vec<i64> = (0..case.shared_blocks).map(|_| free.pop().unwrap_or(0) as i64).collect();
        let mut lengths = Vec::new();
        let mut tables = Vec::new();
        let mut cached = Vec::new();
        for (s, &n) in case.sequences.iter().enumerate() {
            let mut table = vec![-1i64; case.stride];
            for (i, entry) in table.iter_mut().take(n.div_ceil(case.block_size)).enumerate() {
                *entry = shared.get(i).copied().unwrap_or_else(|| free.pop().unwrap_or(0) as i64);
            }
            for p in 0..n {
                cached.push((
                    table[p / case.block_size] as usize * case.block_size + p % case.block_size,
                    s,
                    p,
                ));
            }
            // Prefill rows: the last `case.prefill` positions of the sequence, causally.
            for len in n + 1 - case.prefill.min(n)..=n {
                lengths.push(len as i64);
                tables.extend_from_slice(&table);
            }
        }
        Batch {
            lengths,
            tables,
            cached,
        }
    }

    fn attention(&mut self, case: &AttnCase) -> Result<Outcome, BackendError> {
        let (dk, dv, heads, kv_heads) = (case.dk, case.dv, case.heads, case.kv_heads);
        let batch = self.batch(case);
        let rows = batch.lengths.len();
        let slots_total = case.pool_blocks * case.block_size;
        let cached = batch.cached.len();
        let k = self.rng.vector(cached * kv_heads * dk, 1.0);
        let v = self.rng.vector(cached * kv_heads * dv, 1.0);
        let q = self.rng.vector(rows * heads * dk, 1.0);
        let sinks = case.sinks.then(|| Arc::new(self.rng.vector(heads, 2.0)));
        let alibi = case.alibi.then(|| {
            Arc::new(
                (0..heads)
                    .map(|h| 0.5f32.powi(h as i32 + 1))
                    .collect::<Vec<_>>(),
            )
        });
        let params = AttentionParams {
            num_heads: heads,
            num_kv_heads: kv_heads,
            head_dim: dk,
            value_dim: dv,
            stride: case.stride,
            block_size: case.block_size,
            scale: case.scale.unwrap_or(1.0 / (dk as f32).sqrt()),
            softcap: case.softcap,
            window: case.window,
            chunk: case.chunk,
            sinks: sinks.clone(),
            alibi_slopes: alibi.clone(),
        };
        let layers = 2usize;
        let layer = 1usize;
        let slot_list: Vec<usize> = batch.cached.iter().map(|c| c.0).collect();

        // Encode the cache once on the CPU backend; both backends attend over these bytes.
        let cpu_cache = self.cpu.allocate_kv_cache(
            layers,
            case.pool_blocks,
            case.block_size,
            kv_heads,
            dk,
            dv,
            case.kind,
        )?;
        {
            let mut s = self.cpu.create_stream()?;
            let kb = put(&self.cpu, s.as_mut(), &k)?;
            let vb = put(&self.cpu, s.as_mut(), &v)?;
            s.kv_cache_write(&cpu_cache, layer, &slot_list, &kb, &vb)?;
        }
        let mut pool_bytes = Vec::new();
        {
            let mut s = self.cpu.create_stream()?;
            for pool in [&cpu_cache.key_pool, &cpu_cache.value_pool] {
                let mut bytes = vec![0u8; pool.size];
                s.download(&mut bytes, pool)?;
                pool_bytes.push(bytes);
            }
        }
        let words: Vec<Vec<u16>> = pool_bytes
            .iter()
            .map(|b| {
                b.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| u16::from_le_bytes(*c))
                    .collect()
            })
            .collect();
        let key_view = PoolView {
            words: &words[0],
            layer_words: cpu_cache.words_per_layer_k,
            slots: slots_total,
            kv: kv_heads * dk,
            group: kv_quant_group(dk),
            q8: case.kind == 2,
            layout: layout_of(case.kind, true, slots_total, kv_heads, dk),
        };
        let value_view = PoolView {
            words: &words[1],
            layer_words: cpu_cache.words_per_layer_v,
            slots: slots_total,
            kv: kv_heads * dv,
            group: kv_quant_group(dv),
            q8: case.kind == 2,
            layout: layout_of(case.kind, false, slots_total, kv_heads, dv),
        };

        // f64 reference with the error budget of every output element (see `tol` below).
        let group = heads / kv_heads;
        let mut want = vec![0.0f64; rows * heads * dv];
        let mut scale = vec![0.0f64; rows * heads * dv];
        let (mut tol, mut cpu_tol) = (0.0f64, 0.0f64);
        let tiled = params.window.is_none()
            && params.chunk.is_none()
            && params.softcap.is_none()
            && sinks.is_none()
            && alibi.is_none()
            && case.scale.is_none()
            && dk == dv
            && dk.is_multiple_of(32)
            && dk <= 256
            && group <= 16;
        let sp = f64::from(params.scale);
        for r in 0..rows {
            let n = batch.lengths[r] as usize;
            let start = match (params.chunk, params.window) {
                (Some(c), _) => (n - 1) / c * c,
                (None, Some(w)) => n.saturating_sub(w),
                (None, None) => 0,
            };
            let table = &batch.tables[r * case.stride..(r + 1) * case.stride];
            let slot_of = |p: usize| {
                table[p / case.block_size] as usize * case.block_size + p % case.block_size
            };
            for h in 0..heads {
                let kh = h / group;
                let query = &q[(r * heads + h) * dk..(r * heads + h + 1) * dk];
                let q_amax = query.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
                let mut scores = Vec::with_capacity(n - start);
                let (mut ds, mut ds_q8) = (0.0f64, 0.0f64);
                for p in start..n {
                    let slot = slot_of(p);
                    let (mut dot, mut mag, mut kabs) = (0.0f64, 0.0f64, 0.0f64);
                    for (d, &qv) in query.iter().enumerate() {
                        let kv = key_view.value(layer, slot, kh * dk + d);
                        dot += f64::from(qv) * kv;
                        mag += (f64::from(qv) * kv).abs();
                        kabs += kv.abs();
                    }
                    let mut s = dot * sp;
                    let mut bound = gamma(dk + 2) * mag * sp.abs();
                    if let Some(slopes) = &alibi {
                        let term = f64::from(slopes[h]) * (p as f64 - (n - 1) as f64);
                        s += term;
                        bound += 4.0 * U * (term.abs() + s.abs());
                    }
                    if let Some(cap) = params.softcap {
                        let cap = f64::from(cap);
                        s = cap * (s / cap).tanh();
                        bound += 16.0 * U * cap;
                    }
                    ds = ds.max(bound);
                    // The tiled CPU kernel scores Q8 caches with Q8-packed queries: each query
                    // value moves by at most 0.6 of its group's step (rounding plus f16 scale).
                    ds_q8 = ds_q8.max(sp * kabs * 0.6 * q_amax / 127.0);
                    scores.push(s);
                }
                let sink = sinks.as_ref().map(|s| f64::from(s[h]));
                let max = scores
                    .iter()
                    .copied()
                    .fold(sink.unwrap_or(f64::NEG_INFINITY), f64::max);
                let mut total = sink.map_or(0.0, |s| (s - max).exp());
                let weights: Vec<f64> = scores.iter().map(|&s| (s - max).exp()).collect();
                total += weights.iter().sum::<f64>();
                let mut v_peak = vec![0.0f64; dv];
                for (i, &w) in weights.iter().enumerate() {
                    let slot = slot_of(start + i);
                    for (d, peak) in v_peak.iter_mut().enumerate() {
                        let x = value_view.value(layer, slot, kh * dv + d);
                        want[(r * heads + h) * dv + d] += w / total * x;
                        *peak = peak.max(x.abs());
                    }
                }
                // |error| <= max|v| * (2 * score error + roundings of the softmax sum, the
                // weighted sum and the online rescaling (gamma(2n)), exp (3 + 2|x| ULP, at most
                // ~8 units after weighting) and the final division).
                let count = n - start;
                let budget = 2.0 * ds + gamma(2 * count + dk) + 32.0 * U;
                tol = tol.max(budget);
                cpu_tol = cpu_tol.max(
                    budget
                        + if tiled && case.kind == 2 {
                            2.0 * ds_q8
                        } else {
                            0.0
                        },
                );
                for (d, &peak) in v_peak.iter().enumerate() {
                    scale[(r * heads + h) * dv + d] = peak;
                }
            }
        }
        if tiled {
            // The tiled CPU kernel (polynomial exp, Q8 query packing) is held to the contract
            // `selftest-cpu` checks: 1e-4 of the sum of absolute terms, here at most max|v|.
            cpu_tol = cpu_tol.max(1e-4);
        }
        if self.fast {
            tol = tol.max(Self::FAST_TOL);
        }
        let op = |backend: &dyn Backend,
                  stream: &mut dyn Stream,
                  ms: &mut f64|
         -> Result<Vec<f32>, BackendError> {
            let cache = backend.allocate_kv_cache(
                layers,
                case.pool_blocks,
                case.block_size,
                kv_heads,
                dk,
                dv,
                case.kind,
            )?;
            for (mut pool, bytes) in [cache.key_pool.clone(), cache.value_pool.clone()]
                .into_iter()
                .zip(&pool_bytes)
            {
                // Device pools may be padded to whole 32-bit words.
                let mut padded = bytes.clone();
                padded.resize(pool.size.max(bytes.len()), 0);
                stream.upload(&mut pool, &padded)?;
            }
            let qb = put(backend, stream, &q)?;
            let mut out = backend.allocate_buffer(rows * heads * dv * 4)?;
            if case.step {
                stream.begin_step();
            }
            let run = timed(ms, stream, |s| {
                s.attention_paged(
                    &mut out,
                    &qb,
                    &cache,
                    &batch.tables,
                    &batch.lengths,
                    layer,
                    &params,
                )
            });
            if case.step {
                stream.end_step();
            }
            run?;
            get(stream, &out, rows * heads * dv)
        };
        let (device, cpu, ms) = self.both(&op)?;
        Ok(Outcome {
            device: worst(&device, &want, &scale),
            cpu: worst(&cpu, &want, &scale),
            tol,
            cpu_tol,
            ms,
        })
    }
}

/// One paged-attention configuration.
#[derive(Clone)]
struct AttnCase {
    kind: i32,
    dk: usize,
    dv: usize,
    heads: usize,
    kv_heads: usize,
    block_size: usize,
    stride: usize,
    pool_blocks: usize,
    /// Cached length of each sequence.
    sequences: Vec<usize>,
    /// Rows per sequence: 1 decodes the last position, more prefill the last positions.
    prefill: usize,
    window: Option<usize>,
    chunk: Option<usize>,
    softcap: Option<f32>,
    sinks: bool,
    alibi: bool,
    scale: Option<f32>,
    /// Leading blocks that all sequences share (a common prompt prefix in the prefix cache).
    shared_blocks: usize,
    /// Run the call inside a forward pass (`begin_step`), as the decoder does: the library then caches the block tables and groups the rows once.
    step: bool,
}

impl AttnCase {
    fn new(kind: i32, dk: usize, sequences: &[usize]) -> Self {
        Self {
            kind,
            dk,
            dv: dk,
            heads: 8,
            kv_heads: 2,
            block_size: 16,
            stride: 24,
            pool_blocks: 64,
            sequences: sequences.to_vec(),
            prefill: 1,
            window: None,
            chunk: None,
            softcap: None,
            sinks: false,
            alibi: false,
            scale: None,
            shared_blocks: 0,
            step: false,
        }
    }

    fn describe(&self) -> String {
        let mut s = format!("{}/{} h d{}", self.heads, self.kv_heads, self.dk);
        if self.dv != self.dk {
            s += &format!("/{}", self.dv);
        }
        s += &format!(
            " {} {:?}",
            if self.prefill > 1 {
                format!("prefill{}", self.prefill)
            } else {
                "decode".into()
            },
            self.sequences
        );
        if self.block_size != 16 {
            s += &format!(" bs{}", self.block_size);
        }
        if let Some(w) = self.window {
            s += &format!(" win{w}");
        }
        if let Some(c) = self.chunk {
            s += &format!(" chunk{c}");
        }
        if self.softcap.is_some() {
            s += " cap";
        }
        if self.sinks {
            s += " sinks";
        }
        if self.alibi {
            s += " alibi";
        }
        if self.scale.is_some() {
            s += " scale";
        }
        if self.shared_blocks > 0 {
            s += &format!(" shared{}", self.shared_blocks);
        }
        if self.step {
            s += " step";
        }
        s
    }
}

/// Runs of 16 or more rows (a prompt of 16 tokens or more) over head widths that the tensor-core prompt kernel and the decode kernels
/// decline (16 and 80, the groups of 2 query heads per KV head), with and without the step cache of a forward pass: the general kernel has to
/// take them. On CUDA and HIP this failed with an empty driver error inside a forward pass; only the fast pass runs these cases (the exact
/// math skips the tensor-core branch).
fn long_prompt_cases(kind: i32) -> Vec<AttnCase> {
    let mut cases = Vec::new();
    for step in [false, true] {
        cases.push(AttnCase {
            prefill: 16,
            step,
            heads: 4,
            ..AttnCase::new(kind, 16, &[40])
        });
        cases.push(AttnCase {
            prefill: 24,
            step,
            ..AttnCase::new(kind, 16, &[40, 30])
        });
        cases.push(AttnCase {
            block_size: 5,
            stride: 16,
            pool_blocks: 33,
            prefill: 18,
            step,
            ..AttnCase::new(kind, 80, &[61, 23])
        });
        cases.push(AttnCase {
            prefill: 20,
            step,
            ..AttnCase::new(kind, 32, &[50])
        });
    }
    cases
}

fn attention_cases() -> Vec<AttnCase> {
    let mut cases = Vec::new();
    for kind in [1, 2] {
        // Decode with batches of 1, 2 and 5 over head widths 32/64/128 (tiled on the CPU)
        // and 16/80 (scalar on the CPU).
        cases.push(AttnCase::new(kind, 64, &[37]));
        cases.push(AttnCase::new(kind, 64, &[1, 64]));
        cases.push(AttnCase::new(kind, 128, &[1, 17, 33, 64, 100]));
        cases.push(AttnCase::new(kind, 32, &[5, 16, 31, 48, 90]));
        cases.push(AttnCase::new(kind, 16, &[3, 40]));
        // Head width 80 with block size 5: q8 layers start on odd 16-bit words.
        cases.push(AttnCase {
            block_size: 5,
            stride: 16,
            pool_blocks: 33,
            ..AttnCase::new(kind, 80, &[23, 61])
        });
        // Several 64-key tiles.
        cases.push(AttnCase::new(kind, 64, &[300]));
        // Prefill: causal rows of one and of two sequences.
        cases.push(AttnCase {
            prefill: 13,
            ..AttnCase::new(kind, 64, &[40])
        });
        cases.push(AttnCase {
            prefill: 5,
            ..AttnCase::new(kind, 128, &[20, 50])
        });
        // Options the decoder passes per model.
        cases.push(AttnCase {
            window: Some(24),
            ..AttnCase::new(kind, 64, &[100, 10])
        });
        cases.push(AttnCase {
            chunk: Some(32),
            prefill: 2,
            ..AttnCase::new(kind, 64, &[70, 31])
        });
        cases.push(AttnCase {
            softcap: Some(5.0),
            scale: Some(0.5),
            ..AttnCase::new(kind, 64, &[50])
        });
        cases.push(AttnCase {
            sinks: true,
            ..AttnCase::new(kind, 64, &[45, 2])
        });
        cases.push(AttnCase {
            alibi: true,
            heads: 4,
            kv_heads: 4,
            ..AttnCase::new(kind, 32, &[60])
        });
        cases.push(AttnCase {
            dv: 32,
            ..AttnCase::new(kind, 64, &[33, 9])
        });
    }
    // Wider than the WGSL kernel: host fallback.
    cases.push(AttnCase {
        heads: 2,
        kv_heads: 1,
        ..AttnCase::new(1, 576, &[20])
    });
    cases
}


/// Checks of the kernels of the less common architectures: LayerNorm, the activations beyond SiLU, clamping, the MLA data movement
/// and the Mamba-2 layer (arch_ops.cu, mla.cu, mamba2.cu), against straightforward host references.
impl Tester<'_> {
    fn arch_cases(&mut self) {
        let ops = self.device.create_stream().map(|s| s.supports_arch_ops()).unwrap_or(false);
        if !ops {
            self.record_skip("arch ops", "LayerNorm, activations, clamp, MLA, Mamba, ShortConv", "supports_arch_ops is false");
            return;
        }
        for (rows, cols, bias) in [(3usize, 64usize, true), (5, 300, true), (2, 1536, false)] {
            let r = self.layer_norm_check(rows, cols, bias);
            self.record("layer_norm", &format!("{rows}x{cols}{}", if bias { " +bias" } else { "" }), r);
        }
        for mode in 0..5 {
            let name = ["silu", "gelu tanh", "gelu erf", "relu", "relu squared"][mode as usize];
            let r = self.activation_check(mode, 2049);
            self.record("activation", name, r);
        }
        let r = self.softplus_check();
        self.record("softplus_gate", "2001 values beta 0.693", r);
        let r = self.clamp_check();
        self.record("clamp", "1001 values limit 0.7", r);
        let heads_ok = self.device.create_stream().map(|s| s.supports_head_gate()).unwrap_or(false);
        if heads_ok {
            for (rows, heads, dv, beta) in [(1usize, 4usize, 16usize, None), (3, 8, 24, None), (5, 6, 20, Some(std::f32::consts::LN_2))] {
                let r = self.head_gate_check(rows, heads, dv, beta);
                let kind = if beta.is_some() { "softplus" } else { "sigmoid" };
                self.record("head_gate", &format!("{kind} {rows} rows x {heads} heads x {dv}"), r);
            }
            let r = self.scale_check();
            self.record("scale", "1001 values factor 0.37", r);
        } else {
            self.record_skip("head_gate", "per-head attention gate, value scale", "supports_head_gate is false");
        }
        for (heads, nope, rope, value, kv_lora, rows) in [(3usize, 5usize, 4usize, 6usize, 7usize, 5usize), (16, 128, 64, 128, 512, 3)] {
            let r = self.mla_check(heads, nope, rope, value, kv_lora, rows);
            self.record("mla_assemble", &format!("{heads} heads {nope}+{rope}/{value} latent {kv_lora}"), r);
        }
        for (inner, state, rank, eps) in [(96usize, 16usize, 6usize, None), (200, 16, 8, Some(1e-5f32)), (128, 32, 4, None)] {
            let shape = Mamba1Shape { inner, state, rank, conv_kernel: 4, bc_rms_eps: eps };
            match self.device.create_stream() {
                Ok(st) if st.supports_mamba1(&shape) => {}
                _ => continue,
            }
            let r = self.mamba1_check(&shape);
            self.record("mamba1", &format!("inner {inner} state {state} rank {rank}{}", if eps.is_some() { " +bc norm" } else { "" }), r);
        }
        for (hidden, kernel) in [(64usize, 3usize), (200, 4), (1024, 3)] {
            if !self.device.create_stream().map(|st| st.supports_shortconv(hidden, kernel)).unwrap_or(false) {
                continue;
            }
            let r = self.shortconv_check(hidden, kernel);
            self.record("shortconv", &format!("{hidden} channels kernel {kernel}"), r);
        }
        for (inner, state, heads, groups, norm_groups, norm_mode) in [
            (128usize, 16usize, 4usize, 2usize, 2usize, 0u8),
            (256, 64, 8, 1, 1, 0),
            (512, 128, 8, 2, 2, 0),
            (192, 32, 3, 3, 1, 0),
            (128, 16, 4, 2, 2, 1),
            (256, 64, 8, 1, 1, 2),
            (192, 32, 3, 3, 3, 2),
        ] {
            let shape = MambaShape { inner, state, heads, groups, conv_kernel: 4, norm_groups, step_min: 0.0, eps: 1e-5, norm_mode };
            match self.device.create_stream() {
                Ok(st) if st.supports_mamba2(&shape) => {}
                _ => continue,
            }
            let r = self.mamba2_check(&shape);
            self.record("mamba2", &format!("inner {inner} state {state} {heads} heads {groups} groups, {norm_groups} norm groups{}", ["", ", no norm", ", norm before the gate"][usize::from(norm_mode)]), r);
        }
    }

    fn layer_norm_check(&mut self, rows: usize, cols: usize, bias: bool) -> Result<Outcome, BackendError> {
        let x = self.rng.vector(rows * cols, 3.0);
        let w: Vec<f32> = (0..cols).map(|_| 1.0 + 0.3 * self.rng.unit()).collect();
        let b: Vec<f32> = self.rng.vector(cols, 0.5);
        let eps = 1e-5f32;
        let mut want = vec![0.0f64; rows * cols];
        for r in 0..rows {
            let row = &x[r * cols..(r + 1) * cols];
            let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / cols as f64;
            let var = row.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / cols as f64;
            let inv = 1.0 / (var + f64::from(eps)).sqrt();
            for c in 0..cols {
                want[r * cols + c] = (f64::from(row[c]) - mean) * inv * f64::from(w[c]) + if bias { f64::from(b[c]) } else { 0.0 };
            }
        }
        let mut stream = self.device.create_stream()?;
        let xb = put(self.device, stream.as_mut(), &x)?;
        let wb = put(self.device, stream.as_mut(), &w)?;
        let bb = put(self.device, stream.as_mut(), &b)?;
        let mut out = put(self.device, stream.as_mut(), &vec![f32::NAN; rows * cols])?;
        let mut ms = 0.0;
        timed(&mut ms, stream.as_mut(), |s| s.layer_norm(&mut out, &xb, &wb, bias.then_some(&bb), eps))?;
        let have = get(stream.as_mut(), &out, rows * cols)?;
        let scale: Vec<f64> = want.iter().map(|v| v.abs().max(1.0)).collect();
        Ok(Outcome { device: worst(&have, &want, &scale), cpu: 0.0, tol: 1e-4, cpu_tol: 1.0, ms })
    }

    fn activation_check(&mut self, mode: i32, n: usize) -> Result<Outcome, BackendError> {
        let x = self.rng.vector(n, 4.0);
        let u = self.rng.vector(n, 2.0);
        let erf = |v: f64| {
            // Abramowitz-Stegun 7.1.26 is too coarse for 1e-5: series / continued fraction via the complementary function
            let t = 1.0 / (1.0 + 0.5 * v.abs());
            let tau = t
                * (-v * v - 1.265_512_23
                    + t * (1.000_023_68
                        + t * (0.374_091_96
                            + t * (0.096_784_18
                                + t * (-0.186_288_06
                                    + t * (0.278_868_07
                                        + t * (-1.135_203_98 + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
                    .exp();
            if v >= 0.0 {
                1.0 - tau
            } else {
                tau - 1.0
            }
        };
        let act = |v: f64| match mode {
            0 => v / (1.0 + (-v).exp()),
            1 => 0.5 * v * (1.0 + (0.797_884_560_802_865_4 * (v + 0.044_715 * v * v * v)).tanh()),
            2 => 0.5 * v * (1.0 + erf(v * std::f64::consts::FRAC_1_SQRT_2)),
            3 => v.max(0.0),
            _ => v.max(0.0) * v.max(0.0),
        };
        let want: Vec<f64> = x.iter().map(|&v| act(f64::from(v))).collect();
        let want_glu: Vec<f64> = x.iter().zip(&u).map(|(&v, &g)| act(f64::from(v)) * f64::from(g)).collect();
        let mut stream = self.device.create_stream()?;
        let xb = put(self.device, stream.as_mut(), &x)?;
        let ub = put(self.device, stream.as_mut(), &u)?;
        let mut out = put(self.device, stream.as_mut(), &vec![f32::NAN; n])?;
        let mut out2 = put(self.device, stream.as_mut(), &vec![f32::NAN; n])?;
        let mut ms = 0.0;
        timed(&mut ms, stream.as_mut(), |s| {
            s.activation(&mut out, &xb, mode)?;
            s.activation_glu(&mut out2, &xb, &ub, mode)
        })?;
        let have = get(stream.as_mut(), &out, n)?;
        let have2 = get(stream.as_mut(), &out2, n)?;
        let scale: Vec<f64> = want.iter().map(|v| v.abs().max(1.0)).collect();
        let scale2: Vec<f64> = want_glu.iter().map(|v| v.abs().max(1.0)).collect();
        // erf of the reference is a 1.2e-7 fit; tanh/exp of the device are a few ulp
        let err = worst(&have, &want, &scale).max(worst(&have2, &want_glu, &scale2));
        // the tanh form of GELU cancels in 1 + tanh(x) for negative x: a few ulp of the large intermediate
        Ok(Outcome { device: err, cpu: 0.0, tol: if mode == 1 { 3e-5 } else { 5e-6 }, cpu_tol: 1.0, ms })
    }

    fn softplus_check(&mut self) -> Result<Outcome, BackendError> {
        let beta = std::f32::consts::LN_2;
        let x = self.rng.vector(2001, 2.0);
        let g = self.rng.vector(2001, 40.0);
        let want: Vec<f64> = x
            .iter()
            .zip(&g)
            .map(|(&v, &gate)| {
                let scaled = f64::from(beta) * f64::from(gate);
                f64::from(v) * if scaled > 20.0 { f64::from(gate) } else { scaled.exp().ln_1p() / f64::from(beta) }
            })
            .collect();
        let mut stream = self.device.create_stream()?;
        let mut buf = put(self.device, stream.as_mut(), &x)?;
        let gate = put(self.device, stream.as_mut(), &g)?;
        let mut ms = 0.0;
        timed(&mut ms, stream.as_mut(), |s| s.softplus_gate_mul(&mut buf, &gate, beta))?;
        let have = get(stream.as_mut(), &buf, x.len())?;
        let scale: Vec<f64> = want.iter().map(|v| v.abs().max(1.0)).collect();
        Ok(Outcome { device: worst(&have, &want, &scale), cpu: 0.0, tol: 2e-6, cpu_tol: 1.0, ms })
    }

    /// `x[r][h][d] *= act(gate[r][h])`: the per-head attention output gate (Laguna, Step-3.5) against an f64 reference.
    fn head_gate_check(&mut self, rows: usize, heads: usize, dv: usize, beta: Option<f32>) -> Result<Outcome, BackendError> {
        let x = self.rng.vector(rows * heads * dv, 2.0);
        let g = self.rng.vector(rows * heads, 6.0);
        let want: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let gate = f64::from(g[i / dv]);
                let act = match beta {
                    None => 1.0 / (1.0 + (-gate).exp()),
                    Some(b) => {
                        let scaled = f64::from(b) * gate;
                        if scaled > 20.0 { gate } else { scaled.exp().ln_1p() / f64::from(b) }
                    }
                };
                f64::from(v) * act
            })
            .collect();
        let mut stream = self.device.create_stream()?;
        let mut buf = put(self.device, stream.as_mut(), &x)?;
        let gate = put(self.device, stream.as_mut(), &g)?;
        let mut ms = 0.0;
        timed(&mut ms, stream.as_mut(), |s| s.head_gate_mul(&mut buf, &gate, dv, beta))?;
        let have = get(stream.as_mut(), &buf, x.len())?;
        let scale: Vec<f64> = want.iter().map(|v| v.abs().max(1.0)).collect();
        Ok(Outcome { device: worst(&have, &want, &scale), cpu: 0.0, tol: 2e-6, cpu_tol: 1.0, ms })
    }

    fn scale_check(&mut self) -> Result<Outcome, BackendError> {
        let x = self.rng.vector(1001, 2.0);
        let want: Vec<f64> = x.iter().map(|&v| f64::from(v) * f64::from(0.37f32)).collect();
        let mut stream = self.device.create_stream()?;
        let mut buf = put(self.device, stream.as_mut(), &x)?;
        let mut ms = 0.0;
        timed(&mut ms, stream.as_mut(), |s| s.scale_inplace(&mut buf, 0.37))?;
        let have = get(stream.as_mut(), &buf, x.len())?;
        Ok(Outcome { device: worst(&have, &want, &vec![1.0; want.len()]), cpu: 0.0, tol: 1e-6, cpu_tol: 1.0, ms })
    }

    fn clamp_check(&mut self) -> Result<Outcome, BackendError> {
        let x = self.rng.vector(1001, 2.0);
        let want: Vec<f64> = x.iter().map(|&v| f64::from(v).clamp(-0.7, 0.7)).collect();
        let mut stream = self.device.create_stream()?;
        let mut buf = put(self.device, stream.as_mut(), &x)?;
        let mut ms = 0.0;
        timed(&mut ms, stream.as_mut(), |s| s.clamp(&mut buf, 0.7))?;
        let have = get(stream.as_mut(), &buf, x.len())?;
        Ok(Outcome { device: worst(&have, &want, &vec![1.0; want.len()]), cpu: 0.0, tol: 1e-7, cpu_tol: 1.0, ms })
    }

    fn mla_check(&mut self, heads: usize, nope: usize, rope: usize, value: usize, kv_lora: usize, rows: usize) -> Result<Outcome, BackendError> {
        let dk = nope + rope;
        let width = kv_lora + rope;
        let q_full = self.rng.vector(rows * heads * dk, 1.0);
        let kv_a = self.rng.vector(rows * width, 1.0);
        let kv = self.rng.vector(rows * heads * (nope + value), 1.0);
        // reference: `mla_qkv` of the decoder
        let mut want_q = vec![0.0f64; rows * heads * dk];
        let mut want_k = vec![0.0f64; rows * heads * dk];
        let mut want_v = vec![0.0f64; rows * heads * value];
        let mut want_latent = vec![0.0f64; rows * kv_lora];
        for r in 0..rows {
            for c in 0..kv_lora {
                want_latent[r * kv_lora + c] = f64::from(kv_a[r * width + c]);
            }
            let pe = &kv_a[r * width + kv_lora..(r + 1) * width];
            for h in 0..heads {
                let at = r * heads + h;
                let source = &q_full[at * dk..(at + 1) * dk];
                for d in 0..rope {
                    want_q[at * dk + d] = f64::from(source[nope + d]);
                    want_k[at * dk + d] = f64::from(pe[d]);
                }
                for d in 0..nope {
                    want_q[at * dk + rope + d] = f64::from(source[d]);
                    want_k[at * dk + rope + d] = f64::from(kv[at * (nope + value) + d]);
                }
                for e in 0..value {
                    want_v[at * value + e] = f64::from(kv[at * (nope + value) + nope + e]);
                }
            }
        }
        let mut stream = self.device.create_stream()?;
        let qf = put(self.device, stream.as_mut(), &q_full)?;
        let ka = put(self.device, stream.as_mut(), &kv_a)?;
        let kvb = put(self.device, stream.as_mut(), &kv)?;
        let mut q = put(self.device, stream.as_mut(), &vec![f32::NAN; rows * heads * dk])?;
        let mut k = put(self.device, stream.as_mut(), &vec![f32::NAN; rows * heads * dk])?;
        let mut v = put(self.device, stream.as_mut(), &vec![f32::NAN; rows * heads * value])?;
        let mut latent = put(self.device, stream.as_mut(), &vec![f32::NAN; rows * kv_lora])?;
        let shape = MlaShape { heads, nope, rope, value, kv_lora };
        let mut ms = 0.0;
        timed(&mut ms, stream.as_mut(), |s| {
            s.mla_latent(&mut latent, &ka, rows, kv_lora, width)?;
            s.mla_assemble(&mut q, &mut k, &mut v, &qf, &ka, &kvb, rows, &shape)
        })?;
        let one = |want: &[f64], have: Vec<f32>| worst(&have, want, &vec![1.0; want.len()]);
        let err = one(&want_q, get(stream.as_mut(), &q, want_q.len())?)
            .max(one(&want_k, get(stream.as_mut(), &k, want_k.len())?))
            .max(one(&want_v, get(stream.as_mut(), &v, want_v.len())?))
            .max(one(&want_latent, get(stream.as_mut(), &latent, want_latent.len())?));
        Ok(Outcome { device: err, cpu: 0.0, tol: 0.0, cpu_tol: 1.0, ms })
    }

    /// Mamba-2 layer: prefill runs of several lengths (shorter than the convolution history too), decode steps of three sequences
    /// in one batch and mixed batches, against a transcription of `mamba_forward` (rust_decoder.rs).
    fn mamba2_check(&mut self, shape: &MambaShape) -> Result<Outcome, BackendError> {
        let (inner, state, heads, groups, ksz) = (shape.inner, shape.state, shape.heads, shape.groups, shape.conv_kernel);
        let conv_dim = shape.conv_dim();
        let head_dim = inner / heads;
        let per_group = heads / groups;
        let width = inner + conv_dim + heads;
        let conv = self.rng.vector(conv_dim * ksz, 0.6);
        let conv_bias = self.rng.vector(conv_dim, 0.3);
        let dt_bias = self.rng.vector(heads, 1.0);
        let decay: Vec<f32> = (0..heads).map(|_| -(0.2 + self.rng.unit().abs() * 2.0)).collect();
        let skip = self.rng.vector(heads, 1.0);
        let norm: Vec<f32> = (0..inner).map(|_| 1.0 + 0.25 * self.rng.unit()).collect();
        let calls: [&[(usize, usize)]; 10] = [
            &[(0, 37), (1, 5), (2, 2)],
            &[(0, 1), (1, 1), (2, 1)],
            &[(0, 1), (1, 1), (2, 1)],
            &[(0, 20), (1, 1)],
            &[(2, 6), (0, 1), (1, 3)],
            &[(1, 1), (0, 1), (2, 1)],
            &[(0, 150), (1, 64), (2, 3)],
            &[(1, 65), (2, 130)],
            &[(0, 1), (2, 64), (1, 1)],
            &[(0, 70), (1, 2), (2, 1)],
        ];
        let mut stream = self.device.create_stream()?;
        let conv_buf = put(self.device, stream.as_mut(), &conv)?;
        let bias_buf = put(self.device, stream.as_mut(), &conv_bias)?;
        let dt_buf = put(self.device, stream.as_mut(), &dt_bias)?;
        let decay_buf = put(self.device, stream.as_mut(), &decay)?;
        let skip_buf = put(self.device, stream.as_mut(), &skip)?;
        let norm_buf = put(self.device, stream.as_mut(), &norm)?;
        let weights = MambaWeights {
            conv: &conv_buf,
            conv_bias: Some(&bias_buf),
            dt_bias: &dt_buf,
            decay: &decay_buf,
            skip: &skip_buf,
            norm: &norm_buf,
        };
        let offset = 256;
        let states: Vec<DeviceBuffer> = (0..3)
            .map(|_| self.device.allocate_buffer(offset + shape.layer_state_bytes()))
            .collect::<Result<_, _>>()?;
        let mut hist = vec![vec![0.0f32; conv_dim * (ksz - 1)]; 3];
        let mut rec = vec![vec![0.0f32; heads * head_dim * state]; 3];
        let mut started = [false; 3];
        let (mut worst_err, mut total_ms) = (0.0f64, 0.0f64);
        for call in calls {
            let rows: usize = call.iter().map(|c| c.1).sum();
            let proj = self.rng.vector(rows * width, 1.5);
            let mut want = vec![0.0f32; rows * inner];
            let mut runs = Vec::new();
            let mut first = 0;
            for &(seq, n) in call {
                for t in 0..n {
                    let row = &proj[(first + t) * width..(first + t + 1) * width];
                    let (gate, rest) = row.split_at(inner);
                    let (xbc, dt_raw) = rest.split_at(conv_dim);
                    let mut mixed = vec![0.0f32; conv_dim];
                    for (c, m) in mixed.iter_mut().enumerate() {
                        let w = &conv[c * ksz..(c + 1) * ksz];
                        let h = &mut hist[seq][c * (ksz - 1)..(c + 1) * (ksz - 1)];
                        let mut acc = w[ksz - 1] * xbc[c] + conv_bias[c];
                        for (wi, hi) in w.iter().zip(h.iter()) {
                            acc += wi * hi;
                        }
                        h.copy_within(1.., 0);
                        h[ksz - 2] = xbc[c];
                        *m = acc / (1.0 + (-acc).exp());
                    }
                    let (xs, bc) = mixed.split_at(inner);
                    let (b_all, c_all) = bc.split_at(groups * state);
                    let y_row = &mut want[(first + t) * inner..(first + t + 1) * inner];
                    for h in 0..heads {
                        let raw = dt_raw[h] + dt_bias[h];
                        let dt = (if raw > 20.0 { raw } else { raw.exp().ln_1p() }).max(shape.step_min);
                        let da = (dt * decay[h]).exp();
                        let g = h / per_group;
                        let b = &b_all[g * state..(g + 1) * state];
                        let c = &c_all[g * state..(g + 1) * state];
                        for p in 0..head_dim {
                            let xp = xs[h * head_dim + p];
                            let cell = &mut rec[seq][(h * head_dim + p) * state..(h * head_dim + p + 1) * state];
                            let mut acc = 0.0f32;
                            for nn in 0..state {
                                cell[nn] = cell[nn] * da + dt * xp * b[nn];
                                acc += cell[nn] * c[nn];
                            }
                            y_row[h * head_dim + p] = acc + skip[h] * xp;
                        }
                    }
                    let silu = |g: f32| g / (1.0 + (-g).exp());
                    if shape.norm_mode != 2 {
                        for (v, g) in y_row.iter_mut().zip(gate) {
                            *v *= silu(*g);
                        }
                    }
                    let size = inner / shape.norm_groups;
                    if shape.norm_mode != 1 {
                        for (chunk, weight) in y_row.chunks_mut(size).zip(norm.chunks(size)) {
                            let inv = (chunk.iter().map(|v| v * v).sum::<f32>() / size as f32 + shape.eps).sqrt().recip();
                            for (v, w) in chunk.iter_mut().zip(weight) {
                                *v *= inv * w;
                            }
                        }
                    }
                    if shape.norm_mode == 2 {
                        for (v, g) in y_row.iter_mut().zip(gate) {
                            *v *= silu(*g);
                        }
                    }
                }
                runs.push(DeltaRun { state: &states[seq], first_row: first, rows: n, reset: !started[seq] });
                started[seq] = true;
                first += n;
            }
            let proj_buf = put(self.device, stream.as_mut(), &proj)?;
            let mut mixed = self.device.allocate_buffer(rows * conv_dim * 4)?;
            let mut out = put(self.device, stream.as_mut(), &vec![f32::NAN; rows * inner])?;
            let mut ms = 0.0;
            timed(&mut ms, stream.as_mut(), |s| s.mamba2_layer(shape, &weights, &runs, offset, &proj_buf, &mut mixed, &mut out))?;
            total_ms += ms;
            let have = get(stream.as_mut(), &out, rows * inner)?;
            let want64: Vec<f64> = want.iter().map(|&v| f64::from(v)).collect();
            let scale: Vec<f64> = want64.iter().map(|v| v.abs().max(1.0)).collect();
            worst_err = worst_err.max(worst(&have, &want64, &scale));
        }
        Ok(Outcome { device: worst_err, cpu: 0.0, tol: 2e-3, cpu_tol: 1.0, ms: total_ms })
    }

    /// LFM2 short convolution against a transcription of `shortconv_forward` (rust_decoder.rs).
    fn shortconv_check(&mut self, hidden: usize, kernel: usize) -> Result<Outcome, BackendError> {
        let conv = self.rng.vector(hidden * kernel, 0.7);
        let bias = self.rng.vector(hidden, 0.3);
        let calls: [&[(usize, usize)]; 5] = [&[(0, 29), (1, 2), (2, 1)], &[(0, 1), (1, 1), (2, 1)], &[(0, 1), (1, 1), (2, 1)], &[(2, 9), (0, 1)], &[(1, 1), (0, 1), (2, 1)]];
        let mut stream = self.device.create_stream()?;
        let conv_buf = put(self.device, stream.as_mut(), &conv)?;
        let bias_buf = put(self.device, stream.as_mut(), &bias)?;
        let offset = 256;
        let states: Vec<DeviceBuffer> =
            (0..3).map(|_| self.device.allocate_buffer(offset + hidden * (kernel - 1) * 4)).collect::<Result<_, _>>()?;
        let mut hist = vec![vec![0.0f32; hidden * (kernel - 1)]; 3];
        let mut started = [false; 3];
        let (mut worst_err, mut total_ms) = (0.0f64, 0.0f64);
        for call in calls {
            let rows: usize = call.iter().map(|c| c.1).sum();
            let proj = self.rng.vector(rows * 3 * hidden, 1.5);
            let mut want = vec![0.0f64; rows * hidden];
            let mut runs = Vec::new();
            let mut first = 0;
            for &(seq, n) in call {
                for t in 0..n {
                    let row = &proj[(first + t) * 3 * hidden..(first + t + 1) * 3 * hidden];
                    let (b, rest) = row.split_at(hidden);
                    let (c, x) = rest.split_at(hidden);
                    for ch in 0..hidden {
                        let h = &mut hist[seq][ch * (kernel - 1)..(ch + 1) * (kernel - 1)];
                        let w = &conv[ch * kernel..(ch + 1) * kernel];
                        let cur = b[ch] * x[ch];
                        let mut value = bias[ch];
                        for (wi, hi) in w.iter().zip(h.iter()) {
                            value += wi * hi;
                        }
                        value += w[kernel - 1] * cur;
                        want[(first + t) * hidden + ch] = f64::from(c[ch] * value);
                        h.copy_within(1.., 0);
                        h[kernel - 2] = cur;
                    }
                }
                runs.push(DeltaRun { state: &states[seq], first_row: first, rows: n, reset: !started[seq] });
                started[seq] = true;
                first += n;
            }
            let proj_buf = put(self.device, stream.as_mut(), &proj)?;
            let mut out = put(self.device, stream.as_mut(), &vec![f32::NAN; rows * hidden])?;
            let mut ms = 0.0;
            timed(&mut ms, stream.as_mut(), |s| {
                s.shortconv_layer(hidden, kernel, &conv_buf, Some(&bias_buf), &runs, offset, &proj_buf, &mut out)
            })?;
            total_ms += ms;
            let have = get(stream.as_mut(), &out, rows * hidden)?;
            let scale: Vec<f64> = want.iter().map(|v| v.abs().max(1.0)).collect();
            worst_err = worst_err.max(worst(&have, &want, &scale));
        }
        Ok(Outcome { device: worst_err, cpu: 0.0, tol: 1e-5, cpu_tol: 1.0, ms: total_ms })
    }

    /// Mamba-1 kernels (convolution, dt/B/C preparation, selective scan) against a transcription of `mamba1_forward`; the GEMMs of
    /// x_proj and dt_proj are replaced by random matrices of the same shape (`split_raw`, `dt`).
    fn mamba1_check(&mut self, shape: &Mamba1Shape) -> Result<Outcome, BackendError> {
        let (inner, n, rank, ksz) = (shape.inner, shape.state, shape.rank, shape.conv_kernel);
        let conv = self.rng.vector(inner * ksz, 0.6);
        let conv_bias = self.rng.vector(inner, 0.3);
        let decay: Vec<f32> = (0..inner * n).map(|_| -(0.2 + self.rng.unit().abs() * 2.0)).collect();
        let skip = self.rng.vector(inner, 1.0);
        let calls: [&[(usize, usize)]; 5] = [&[(0, 33), (1, 5), (2, 2)], &[(0, 1), (1, 1), (2, 1)], &[(0, 1), (1, 1), (2, 1)], &[(2, 7), (0, 1), (1, 3)], &[(1, 1), (0, 1), (2, 1)]];
        let mut stream = self.device.create_stream()?;
        let conv_buf = put(self.device, stream.as_mut(), &conv)?;
        let bias_buf = put(self.device, stream.as_mut(), &conv_bias)?;
        let decay_buf = put(self.device, stream.as_mut(), &decay)?;
        let skip_buf = put(self.device, stream.as_mut(), &skip)?;
        let offset = 256;
        let states: Vec<DeviceBuffer> = (0..3).map(|_| self.device.allocate_buffer(offset + shape.layer_state_bytes())).collect::<Result<_, _>>()?;
        let mut hist = vec![vec![0.0f32; inner * (ksz - 1)]; 3];
        let mut rec = vec![vec![0.0f32; inner * n]; 3];
        let mut started = [false; 3];
        let (mut worst_err, mut total_ms) = (0.0f64, 0.0f64);
        let width = rank + 2 * n;
        for call in calls {
            let rows: usize = call.iter().map(|c| c.1).sum();
            let proj = self.rng.vector(rows * 2 * inner, 1.5);
            let split_raw = self.rng.vector(rows * width, 1.5);
            let dt = self.rng.vector(rows * inner, 2.0);
            let mut want_mixed = vec![0.0f32; rows * inner];
            let mut want_split = split_raw.clone();
            let mut want_y = vec![0.0f32; rows * inner];
            let mut runs = Vec::new();
            let mut first = 0;
            for &(seq, cnt) in call {
                for t in 0..cnt {
                    let row = first + t;
                    let xs = &proj[row * 2 * inner..row * 2 * inner + inner];
                    let gate = &proj[row * 2 * inner + inner..(row + 1) * 2 * inner];
                    let mixed = &mut want_mixed[row * inner..(row + 1) * inner];
                    for (c, m) in mixed.iter_mut().enumerate() {
                        let w = &conv[c * ksz..(c + 1) * ksz];
                        let h = &mut hist[seq][c * (ksz - 1)..(c + 1) * (ksz - 1)];
                        let mut acc = w[ksz - 1] * xs[c] + conv_bias[c];
                        for (wi, hi) in w.iter().zip(h.iter()) {
                            acc += wi * hi;
                        }
                        h.copy_within(1.., 0);
                        h[ksz - 2] = xs[c];
                        *m = acc / (1.0 + (-acc).exp());
                    }
                    let sp = &mut want_split[row * width..(row + 1) * width];
                    if let Some(eps) = shape.bc_rms_eps {
                        let (dt_part, rest) = sp.split_at_mut(rank);
                        let (b_part, c_part) = rest.split_at_mut(n);
                        for part in [dt_part, b_part, c_part] {
                            let inv = (part.iter().map(|v| v * v).sum::<f32>() / part.len() as f32 + eps).sqrt().recip();
                            part.iter_mut().for_each(|v| *v *= inv);
                        }
                    }
                    let (b, c) = (&sp[rank..rank + n], &sp[rank + n..]);
                    for ch in 0..inner {
                        let raw = dt[row * inner + ch];
                        let step = if raw > 20.0 { raw } else { raw.exp().ln_1p() };
                        let xc = mixed[ch];
                        let s = &mut rec[seq][ch * n..(ch + 1) * n];
                        let mut acc = 0.0f32;
                        for k in 0..n {
                            s[k] = s[k] * (step * decay[ch * n + k]).exp() + step * b[k] * xc;
                            acc += s[k] * c[k];
                        }
                        let z = gate[ch];
                        want_y[row * inner + ch] = (acc + skip[ch] * xc) * (z / (1.0 + (-z).exp()));
                    }
                }
                runs.push(DeltaRun { state: &states[seq], first_row: first, rows: cnt, reset: !started[seq] });
                started[seq] = true;
                first += cnt;
            }
            let proj_buf = put(self.device, stream.as_mut(), &proj)?;
            let mut split_buf = put(self.device, stream.as_mut(), &split_raw)?;
            let dt_buf = put(self.device, stream.as_mut(), &dt)?;
            let mut mixed_buf = put(self.device, stream.as_mut(), &vec![f32::NAN; rows * inner])?;
            let mut dt_in_buf = put(self.device, stream.as_mut(), &vec![f32::NAN; rows * rank])?;
            let mut out = put(self.device, stream.as_mut(), &vec![f32::NAN; rows * inner])?;
            let mut ms = 0.0;
            timed(&mut ms, stream.as_mut(), |s| {
                s.mamba1_conv(shape, &conv_buf, Some(&bias_buf), &runs, offset, &proj_buf, &mut mixed_buf)?;
                s.mamba1_prep(shape, &mut split_buf, &mut dt_in_buf, rows, None)?;
                s.mamba1_scan(shape, &decay_buf, &skip_buf, &runs, offset, &split_buf, &dt_buf, &mixed_buf, &proj_buf, &mut out)
            })?;
            total_ms += ms;
            let cmp = |have: Vec<f32>, want: &[f32]| {
                let want64: Vec<f64> = want.iter().map(|&v| f64::from(v)).collect();
                let scale: Vec<f64> = want64.iter().map(|v| v.abs().max(1.0)).collect();
                worst(&have, &want64, &scale)
            };
            let dt_in_want: Vec<f32> = (0..rows).flat_map(|r| want_split[r * width..r * width + rank].to_vec()).collect();
            worst_err = worst_err
                .max(cmp(get(stream.as_mut(), &mixed_buf, rows * inner)?, &want_mixed))
                .max(cmp(get(stream.as_mut(), &split_buf, rows * width)?, &want_split))
                .max(cmp(get(stream.as_mut(), &dt_in_buf, rows * rank)?, &dt_in_want))
                .max(cmp(get(stream.as_mut(), &out, rows * inner)?, &want_y));
        }
        Ok(Outcome { device: worst_err, cpu: 0.0, tol: 2e-3, cpu_tol: 1.0, ms: total_ms })
    }
}
/// Number of differing bits between two results: 0 when identical, otherwise a value far above any tolerance.
fn bit_diff(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.iter().zip(b).any(|(x, y)| x.to_bits() != y.to_bits()) {
        1.0
    } else {
        0.0
    }
}

/// Largest |a - b| relative to the largest |b| (a GEMM that sums in a fixed order gives 0; a quantization that differs does not).
fn max_relative(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    let top = b.iter().fold(0.0f64, |m, &v| m.max(f64::from(v).abs())).max(1e-30);
    let mut worst = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        let e = (f64::from(x) - f64::from(y)).abs() / top;
        if e.is_nan() {
            return f64::INFINITY;
        }
        worst = worst.max(e);
    }
    worst
}

/// The fused decode kernels (docs/gpu-backends.md, "Fewer kernels per decode token"). Each check runs the fused operation and the unfused
/// sequence it replaces on the same device and expects the same bits (the GEMMs that follow: the same values up to the order of a
/// reduction), and holds the fused result against an f64 reference where the operation has one. A device without the fused operation (or
/// with it switched off by `FLLM_FUSE*`) is skipped.
impl<'a> Tester<'a> {
    /// Tolerance of the GEMM results that follow a fused operation (the order of the reduction is the only difference allowed).
    const FUSE_GEMM_TOL: f64 = 1e-5;

    fn fuse_cases(&mut self) {
        for (rows, cols) in [(1usize, 256usize), (1, 2048), (4, 4096), (3, 5120), (2, 260)] {
            let r = self.fuse_add_norm(rows, cols, None, 0.0);
            self.record("add_rms_norm", &format!("{rows}x{cols}"), r);
        }
        // the activations of the GEMM that follows are prepared by the same kernel: NVFP4 blocks, int8 blocks, FP16 for Marlin
        for kind in self.kinds(&[40u32, 12, 13, 14, 0x8000_0010]) {
            for rows in [1usize, 4] {
                let r = self.fuse_add_norm(rows, 1024, Some(kind), 0.0);
                self.record(&format!("add_rms_norm +{}", format_name(kind)), &format!("{rows}x1024"), r);
            }
            if kind == 40 {
                // the fused kernel must quantize with the activation factor of the weight that reads the result (checkpoint input_scale)
                for rows in [1usize, 4] {
                    let r = self.fuse_add_norm(rows, 1024, Some(kind), 0.0015);
                    self.record("add_rms_norm +nvfp4", &format!("{rows}x1024 input_scale 0.0015"), r);
                }
            }
        }
        if self.filter.wants_kind(0x8000_0010) {
            for (batch, bias) in [(1usize, false), (4, true)] {
                let r = self.fuse_q4z_add_norm(batch, bias);
                self.record("marlin add_rms_norm", &format!("{batch} rows 1024x1024{}", if bias { " +bias" } else { "" }), r);
            }
        }
        for (rows, dim, rope_dim, neox) in [(1usize, 64usize, 64usize, true), (5, 80, 32, false), (3, 128, 64, false), (2, 128, 128, true)] {
            let r = self.fuse_rope_kv(rows, dim, rope_dim, neox);
            self.record(
                "rope_kv_write",
                &format!("{rows} rows 4/2 heads d{dim} r{rope_dim} {}", if neox { "neox" } else { "pairs" }),
                r,
            );
        }
        for kind in self.kinds(&[12u32, 13, 14]) {
            for (batch, geglu) in [(1usize, false), (3, false), (1, true)] {
                let r = self.fuse_glu_q8(kind, batch, 1024, geglu);
                self.record(
                    &format!("{} + int8 quantization {}", if geglu { "geglu" } else { "swiglu" }, format_name(kind)),
                    &format!("{batch}x1024"),
                    r,
                );
            }
        }
        // A gated activation followed by a prompt GEMM that folds it into the FP16 conversion of its input. The second half of the cases gives the kernels the
        // shared memory of a Turing GPU (64 KB): the larger tile shapes do not fit, and a GEMM that finds out after the conversion has to hand the next kernel
        // the activation, not the raw gate.
        for kind in self.kinds(&[2u32, 3, 6, 7, 8, 12, 13, 14, 20]) {
            for (batch, smem) in [(16usize, None), (200, None), (16, Some(65536usize)), (200, Some(65536)), (300, Some(65536))] {
                let r = self.fuse_glu_gemm(kind, batch, 1024, smem);
                let limit = smem.map_or(String::new(), |b| format!(", {} KB shared memory", b / 1024));
                self.record(&format!("swiglu + prompt GEMM {}", format_name(kind)), &format!("{batch}x1024{limit}"), r);
            }
        }
    }

    /// `x += scale * branch; out = rms_norm(x)` (and a GEMM on `out` when `next` names a weight format).
    fn fuse_add_norm(&mut self, rows: usize, cols: usize, next: Option<u32>, input_scale: f32) -> Result<Outcome, BackendError> {
        let x = self.rng.vector(rows * cols, 2.0);
        let branch = self.rng.vector(rows * cols, 1.0);
        let w = self.rng.vector(cols, 1.0);
        let (eps, scale) = (1e-6f32, 0.75f32);
        let out_rows = 384usize;
        let raw = match next {
            Some(kind) => Some(random_kind(&mut self.rng, kind, out_rows, cols)),
            None => None,
        };
        let device = self.device;
        let mut stream = device.create_stream()?;
        let s = stream.as_mut();
        crate::backend::set_next_nvfp4_input_scale(input_scale);
        let weight = match (&raw, next) {
            (Some(raw), Some(kind)) => device.upload_weight_scaled("selftest", raw, QuantType::from_gguf_type_id(kind), out_rows, cols, None, 1.0).map(Some),
            _ => Ok(None),
        };
        crate::backend::set_next_nvfp4_input_scale(0.0);
        let weight = weight?;
        s.synchronize()?;
        let branch_buf = put(device, s, &branch)?;
        let w_buf = put(device, s, &w)?;
        // the unfused sequence
        let mut x_a = put(device, s, &x)?;
        let mut norm_a = device.allocate_buffer(rows * cols * 4)?;
        s.residual_add(&mut x_a, &branch_buf, scale)?;
        s.rms_norm(&mut norm_a, &x_a, &w_buf, eps)?;
        let mut y_a = device.allocate_buffer(rows * out_rows * 4)?;
        if let Some(weight) = &weight {
            s.quantized_linear(&mut y_a, &norm_a, weight, None)?;
        }
        // the fused kernel
        let mut x_b = put(device, s, &x)?;
        let mut norm_b = device.allocate_buffer(rows * cols * 4)?;
        let mut done = false;
        let mut ms = 0.0;
        timed(&mut ms, s, |st| {
            done = st.add_rms_norm(&mut x_b, &branch_buf, scale, &w_buf, eps, &mut norm_b, weight.as_ref())?;
            Ok(())
        })?;
        if !done {
            return Err(BackendError::UnsupportedOperation("add_rms_norm".into()));
        }
        let mut y_b = device.allocate_buffer(rows * out_rows * 4)?;
        if let Some(weight) = &weight {
            s.quantized_linear(&mut y_b, &norm_b, weight, None)?;
        }
        let (xa, xb) = (get(s, &x_a, rows * cols)?, get(s, &x_b, rows * cols)?);
        let (na, nb) = (get(s, &norm_a, rows * cols)?, get(s, &norm_b, rows * cols)?);
        let mut err = bit_diff(&xa, &xb).max(bit_diff(&na, &nb));
        if weight.is_some() {
            err = err.max(max_relative(&get(s, &y_b, rows * out_rows)?, &get(s, &y_a, rows * out_rows)?));
        }
        // f64 reference of the norm output
        let mut want = vec![0.0; rows * cols];
        let mut scale_of = vec![0.0; rows * cols];
        for r in 0..rows {
            let row: Vec<f64> = (0..cols).map(|c| f64::from(x[r * cols + c]) + f64::from(branch[r * cols + c]) * f64::from(scale)).collect();
            let ms_row: f64 = row.iter().map(|v| v * v).sum::<f64>() / cols as f64;
            let inv = 1.0 / (ms_row + f64::from(eps)).sqrt();
            for c in 0..cols {
                let v = row[c] * inv * f64::from(w[c]);
                want[r * cols + c] = v;
                scale_of[r * cols + c] = v.abs();
            }
        }
        Ok(Outcome {
            device: err,
            cpu: worst(&nb, &want, &scale_of),
            tol: Self::FUSE_GEMM_TOL,
            cpu_tol: gamma(cols) / 2.0 + 128.0 * U,
            ms,
        })
    }

    /// The Marlin output projection with the residual update, the norm and the conversion of the next Marlin GEMM.
    fn fuse_q4z_add_norm(&mut self, batch: usize, with_bias: bool) -> Result<Outcome, BackendError> {
        let kind = 0x8000_0010u32;
        let (rows, cols, next_rows) = (1024usize, 1024usize, 384usize);
        let raw = random_kind(&mut self.rng, kind, rows, cols);
        let raw_next = random_kind(&mut self.rng, kind, next_rows, rows);
        let bias = with_bias.then(|| self.rng.vector(rows, 1.0));
        let act = self.rng.vector(batch * cols, 1.0);
        let x = self.rng.vector(batch * rows, 2.0);
        let w = self.rng.vector(rows, 1.0);
        let (eps, scale) = (1e-6f32, 1.0f32);
        let device = self.device;
        let mut stream = device.create_stream()?;
        let s = stream.as_mut();
        let q = QuantType::from_gguf_type_id(kind);
        let weight = device.upload_weight_scaled("selftest", &raw, q, rows, cols, bias.as_deref(), 1.0)?;
        let next = device.upload_weight_scaled("selftest", &raw_next, q, next_rows, rows, None, 1.0)?;
        s.synchronize()?;
        let act_buf = put(device, s, &act)?;
        let w_buf = put(device, s, &w)?;
        // unfused: the product is added to the residual by the Marlin conversion, then norm, then the next GEMM converts its input
        let mut x_a = put(device, s, &x)?;
        let mut norm_a = device.allocate_buffer(batch * rows * 4)?;
        if !s.quantized_linear_add(&mut x_a, &act_buf, &weight, weight.bias.as_ref(), scale)? {
            return Err(BackendError::UnsupportedOperation("quantized_linear_add".into()));
        }
        s.rms_norm(&mut norm_a, &x_a, &w_buf, eps)?;
        let mut y_a = device.allocate_buffer(batch * next_rows * 4)?;
        s.quantized_linear(&mut y_a, &norm_a, &next, None)?;
        // fused
        let mut x_b = put(device, s, &x)?;
        let mut norm_b = device.allocate_buffer(batch * rows * 4)?;
        let mut done = false;
        let mut ms = 0.0;
        timed(&mut ms, s, |st| {
            done = st.quantized_linear_add_norm(&mut x_b, &act_buf, &weight, weight.bias.as_ref(), scale, &w_buf, eps, &mut norm_b, Some(&next))?;
            Ok(())
        })?;
        if !done {
            return Err(BackendError::UnsupportedOperation("quantized_linear_add_norm".into()));
        }
        let mut y_b = device.allocate_buffer(batch * next_rows * 4)?;
        s.quantized_linear(&mut y_b, &norm_b, &next, None)?;
        let (xa, xb) = (get(s, &x_a, batch * rows)?, get(s, &x_b, batch * rows)?);
        let (na, nb) = (get(s, &norm_a, batch * rows)?, get(s, &norm_b, batch * rows)?);
        let err = bit_diff(&xa, &xb)
            .max(bit_diff(&na, &nb))
            .max(max_relative(&get(s, &y_b, batch * next_rows)?, &get(s, &y_a, batch * next_rows)?));
        Ok(Outcome { device: err, cpu: 0.0, tol: Self::FUSE_GEMM_TOL, cpu_tol: 1.0, ms })
    }

    /// Rotary embedding of q and the cache write of k and v (f16 cache) in one kernel.
    fn fuse_rope_kv(&mut self, rows: usize, dim: usize, rope_dim: usize, neox: bool) -> Result<Outcome, BackendError> {
        let (heads, kv_heads, max_pos) = (4usize, 2usize, 64usize);
        let (layers, layer, pool_blocks, block_size) = (3usize, 1usize, 33usize, 16usize);
        let half = rope_dim / 2;
        let mut cos = Vec::with_capacity(max_pos * half);
        let mut sin = Vec::with_capacity(max_pos * half);
        for p in 0..max_pos {
            for d in 0..half {
                let angle = p as f64 / 10_000f64.powf(2.0 * d as f64 / rope_dim as f64);
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
        let positions: Vec<usize> = (0..rows).map(|_| (self.rng.next() % max_pos as u64) as usize).collect();
        let slots_total = pool_blocks * block_size;
        let mut slots: Vec<usize> = Vec::new();
        while slots.len() < rows {
            let slot = (self.rng.next() % slots_total as u64) as usize;
            if !slots.contains(&slot) {
                slots.push(slot);
            }
        }
        let q = self.rng.vector(rows * heads * dim, 1.0);
        let k = self.rng.vector(rows * kv_heads * dim, 1.0);
        let v = self.rng.vector(rows * kv_heads * dim, 2.0);
        let params = RopeParams { heads, kv_heads, dim, rope_dim, neox, cos: Arc::new(cos), sin: Arc::new(sin) };
        let device = self.device;
        let mut stream = device.create_stream()?;
        let s = stream.as_mut();
        let words = kv_layer_words(1, slots_total, kv_heads, dim);
        // rotate, then write
        let mut q_a = put(device, s, &q)?;
        let mut k_a = put(device, s, &k)?;
        let v_a = put(device, s, &v)?;
        let cache_a = device.allocate_kv_cache(layers, pool_blocks, block_size, kv_heads, dim, dim, 1)?;
        s.rope(&mut q_a, &mut k_a, &positions, &params)?;
        s.kv_cache_write(&cache_a, layer, &slots, &k_a, &v_a)?;
        // both in one kernel
        let mut q_b = put(device, s, &q)?;
        let k_b = put(device, s, &k)?;
        let v_b = put(device, s, &v)?;
        let cache_b = device.allocate_kv_cache(layers, pool_blocks, block_size, kv_heads, dim, dim, 1)?;
        let mut done = false;
        let mut ms = 0.0;
        timed(&mut ms, s, |st| {
            done = st.rope_kv_write(&mut q_b, &k_b, &v_b, &cache_b, layer, &positions, &slots, &params)?;
            Ok(())
        })?;
        if !done {
            return Err(BackendError::UnsupportedOperation("rope_kv_write".into()));
        }
        let mut err = bit_diff(&get(s, &q_a, q.len())?, &get(s, &q_b, q.len())?);
        for (pool_a, pool_b) in [(&cache_a.key_pool, &cache_b.key_pool), (&cache_a.value_pool, &cache_b.value_pool)] {
            let (mut bytes_a, mut bytes_b) = (vec![0u8; layers * words * 2], vec![0u8; layers * words * 2]);
            s.download(&mut bytes_a, pool_a)?;
            s.download(&mut bytes_b, pool_b)?;
            if bytes_a != bytes_b {
                err = err.max(1.0);
            }
        }
        Ok(Outcome { device: err, cpu: 0.0, tol: 0.0, cpu_tol: 1.0, ms })
    }

    /// A gated activation followed by a K-quant GEMV: the int8 quantization of the GEMV computes the activation itself.
    #[cfg(feature = "cuda")]
    fn fuse_glu_q8(&mut self, kind: u32, batch: usize, cols: usize, geglu: bool) -> Result<Outcome, BackendError> {
        use super::cuda::{set_fusion, Fusion};
        let out_rows = 384usize;
        let raw = random_kind(&mut self.rng, kind, out_rows, cols);
        let gate = self.rng.vector(batch * cols, 4.0);
        let up = self.rng.vector(batch * cols, 2.0);
        let device = self.device;
        let mut stream = device.create_stream()?;
        let s = stream.as_mut();
        let weight = device.upload_weight_scaled("selftest", &raw, QuantType::from_gguf_type_id(kind), out_rows, cols, None, 1.0)?;
        s.synchronize()?;
        let gate_buf = put(device, s, &gate)?;
        let up_buf = put(device, s, &up)?;
        let mut results: Vec<Vec<f32>> = Vec::new();
        let mut ms = 0.0;
        for fused in [false, true] {
            set_fusion(Fusion::GluQ8, Some(fused));
            let mut act = device.allocate_buffer(batch * cols * 4)?;
            let mut y = device.allocate_buffer(batch * out_rows * 4)?;
            let run = timed(&mut ms, s, |st| {
                if geglu {
                    st.geglu(&mut act, &gate_buf, &up_buf)?;
                } else {
                    st.swiglu(&mut act, &gate_buf, &up_buf)?;
                }
                st.quantized_linear(&mut y, &act, &weight, None)
            });
            set_fusion(Fusion::GluQ8, None);
            run?;
            results.push(get(s, &y, batch * out_rows)?);
        }
        Ok(Outcome { device: max_relative(&results[1], &results[0]), cpu: 0.0, tol: Self::FUSE_GEMM_TOL, cpu_tol: 1.0, ms })
    }

    /// A gated activation followed by a prompt GEMM: the fused sequence (the activation is computed by the conversion of the GEMM's input) against the same
    /// GEMM on the activation read back from the device, with `smem` bytes of shared memory per block pretended (`None`: the device's own).
    #[cfg(feature = "cuda")]
    fn fuse_glu_gemm(&mut self, kind: u32, batch: usize, cols: usize, smem: Option<usize>) -> Result<Outcome, BackendError> {
        use super::cuda::force_shared_memory;
        let out_rows = 384usize;
        let raw = random_kind(&mut self.rng, kind, out_rows, cols);
        let gate = self.rng.vector(batch * cols, 4.0);
        let up = self.rng.vector(batch * cols, 2.0);
        let device = self.device;
        let mut stream = device.create_stream()?;
        let s = stream.as_mut();
        let weight = device.upload_weight_scaled("selftest", &raw, QuantType::from_gguf_type_id(kind), out_rows, cols, None, 1.0)?;
        s.synchronize()?;
        let gate_buf = put(device, s, &gate)?;
        let up_buf = put(device, s, &up)?;
        force_shared_memory(smem);
        let mut ms = 0.0;
        let result = (|| -> Result<(Vec<f32>, Vec<f32>), BackendError> {
            // reference: reading the activation back flushes it (nothing is fused), and the GEMM reads the uploaded copy
            let mut act = device.allocate_buffer(batch * cols * 4)?;
            s.swiglu(&mut act, &gate_buf, &up_buf)?;
            let act_host = get(s, &act, batch * cols)?;
            let act_copy = put(device, s, &act_host)?;
            let mut y_ref = device.allocate_buffer(batch * out_rows * 4)?;
            s.quantized_linear(&mut y_ref, &act_copy, &weight, None)?;
            let mut act2 = device.allocate_buffer(batch * cols * 4)?;
            let mut y = device.allocate_buffer(batch * out_rows * 4)?;
            timed(&mut ms, s, |st| {
                st.swiglu(&mut act2, &gate_buf, &up_buf)?;
                st.quantized_linear(&mut y, &act2, &weight, None)
            })?;
            Ok((get(s, &y, batch * out_rows)?, get(s, &y_ref, batch * out_rows)?))
        })();
        force_shared_memory(None);
        let (have, want) = result?;
        // the activation reaches the GEMM as f16 either way; the only difference allowed is the rounding of the gated activation itself
        Ok(Outcome { device: max_relative(&have, &want), cpu: 0.0, tol: 1e-3, cpu_tol: 1.0, ms })
    }

    #[cfg(not(feature = "cuda"))]
    fn fuse_glu_gemm(&mut self, _kind: u32, _batch: usize, _cols: usize, _smem: Option<usize>) -> Result<Outcome, BackendError> {
        Err(BackendError::UnsupportedOperation("swiglu + prompt GEMM".into()))
    }

    #[cfg(not(feature = "cuda"))]
    fn fuse_glu_q8(&mut self, _kind: u32, _batch: usize, _cols: usize, _geglu: bool) -> Result<Outcome, BackendError> {
        Err(BackendError::UnsupportedOperation("swiglu + int8 quantization".into()))
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn nvfp4_reference_rounds_like_the_hardware() {
        // ties to even on the E2M1 grid: the midpoints 0.25, 1.25, 2.5 and 5 go down, 0.75, 1.75 and 3.5 go up
        for (a, want) in [(0.25, 0.0), (0.26, 0.5), (0.75, 1.0), (1.25, 1.0), (1.75, 2.0), (2.5, 2.0), (3.5, 4.0), (5.0, 4.0), (5.1, 6.0), (100.0, 6.0)] {
            assert_eq!(E2M1_GRID[e2m1_index(a)], want, "{a}");
        }
        // every UE4M3 value is its own nearest code; 448 is the largest, anything above saturates
        for code in 0..=0x7Eu8 {
            assert_eq!(ue4m3_code(ue4m3_value(code)), code);
        }
        assert_eq!(ue4m3_value(0x7E), 448.0);
        assert_eq!(ue4m3_code(1e9), 0x7E);
        assert_eq!(ue4m3_code(0.0), 0);
        // a block whose largest value maps to 6: the scale is amax * gs / 6 rounded to E4M3 and the values follow
        let mut x = vec![0.0f32; 16];
        x[0] = 3.0;
        x[1] = -1.5;
        x[2] = 0.7;
        let q = nvfp4_activation_reference(&x, 1.0);
        assert_eq!((q[0], q[1], q[2], q[3]), (3.0, -1.5, 0.75, 0.0), "scale 0.5: 3.0 -> 6, -1.5 -> -3, 0.7 -> 1.4 -> 1.5, times 0.5");
        // a per-tensor factor moves small blocks out of the subnormal range of E4M3: the block of 1e-3 values survives with gs = 1000 only (the result is x * gs)
        let small = vec![1e-3f32; 16];
        assert_eq!(nvfp4_activation_reference(&small, 1.0)[0], 0.0);
        let kept = nvfp4_activation_reference(&small, 1000.0)[0];
        assert!((kept - 1.0).abs() < 0.07, "{kept}");
    }
    use super::*;

    fn outcome(device: f64, tol: f64) -> Result<Outcome, BackendError> {
        Ok(Outcome { device, cpu: 0.0, tol, cpu_tol: 1.0, ms: 0.5 })
    }

    #[test]
    fn quant_names_are_matched_without_separators() {
        assert_eq!(parse_quants("q4_k, Q8-0").unwrap(), vec![12, 8]);
        assert_eq!(parse_quants("IQ4_NL").unwrap(), vec![20]);
        assert_eq!(parse_quants("12,39").unwrap(), vec![12, 39]);
        assert_eq!(parse_quants("q4_k,q4k").unwrap(), vec![12], "duplicates collapse");
        // families
        assert_eq!(parse_quants("q4z").unwrap(), vec![0x8000_0010, 0x8000_0011, 0x8000_0012]);
        assert_eq!(parse_quants("q4zp").unwrap(), vec![0x8000_0014, 0x8000_0015, 0x8000_0016]);
        assert_eq!(parse_quants("q4z64").unwrap(), vec![0x8000_0011]);
        assert!(parse_quants("iq").unwrap().contains(&16));
        // errors name the known formats
        let error = parse_quants("q9_z").unwrap_err();
        assert!(error.contains("q9_z") && error.contains("q4_k"), "{error}");
        assert!(parse_quants(" , ").is_err());
        assert!(parse_quants("999").is_err());
    }

    #[test]
    fn every_known_format_has_a_name_and_round_trips() {
        for &kind in KINDS {
            assert_ne!(format_name(kind), "?", "{kind:#x}");
            assert_eq!(parse_quants(format_name(kind)).unwrap(), vec![kind]);
        }
    }

    #[test]
    fn filters_select_groups_and_formats() {
        let all = Filter::all();
        assert!(!all.is_active() && GROUPS.iter().all(|g| all.wants_group(g.0)) && all.wants_kind(12));
        assert_eq!(all.describe(), "all checks");
        // --quants alone: the groups with formats
        let quants = Filter::parse(None, Some("q4_k")).unwrap();
        assert!(quants.is_active() && quants.wants_group("linear") && quants.wants_group("moe") && !quants.wants_group("attention"));
        assert!(quants.wants_kind(12) && !quants.wants_kind(8));
        // --only wins over the implication of --quants
        let both = Filter::parse(Some("attention, norm"), Some("q8_0")).unwrap();
        assert!(both.wants_group("attention") && both.wants_group("norm") && !both.wants_group("linear"));
        assert!(Filter::parse(Some("all"), None).unwrap().wants_group("kv"));
        let error = Filter::parse(Some("nonsense"), None).unwrap_err();
        assert!(error.contains("nonsense") && error.contains("attention"), "{error}");
        assert!(Filter::parse(Some(","), None).is_err());
        assert!(both.describe().contains("groups attention,norm") && both.describe().contains("formats q8_0"));
    }

    #[test]
    fn every_group_name_is_unique_and_the_quant_groups_exist() {
        for (i, (name, _)) in GROUPS.iter().enumerate() {
            assert!(GROUPS[i + 1..].iter().all(|g| g.0 != *name), "{name} twice");
        }
        assert!(QUANT_GROUPS.iter().all(|q| GROUPS.iter().any(|g| g.0 == *q)));
    }

    #[test]
    fn arguments() {
        let args = |list: &[&str]| list.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
        match parse_args(&args(&["--device", "vulkan:1", "--only", "norm", "--quants", "q4_k", "--json", "x.json", "--strict"])).unwrap() {
            Command::Run(o) => {
                assert_eq!((o.device.as_str(), o.json.as_deref(), o.strict), ("vulkan:1", Some("x.json"), true));
                assert!(o.filter.wants_group("norm") && !o.filter.wants_group("linear"));
            }
            _ => panic!("expected a run"),
        }
        match parse_args(&args(&[])).unwrap() {
            Command::Run(o) => assert_eq!(o.device, "auto"),
            _ => panic!("expected a run"),
        }
        match parse_args(&args(&["--failures-only", "--only", "kv"])).unwrap() {
            Command::Run(o) => assert!(o.failures_only && !o.strict),
            _ => panic!("expected a run"),
        }
        match parse_args(&args(&[])).unwrap() {
            Command::Run(o) => assert!(!o.failures_only),
            _ => panic!("expected a run"),
        }
        assert!(matches!(parse_args(&args(&["--list"])).unwrap(), Command::List));
        assert!(matches!(parse_args(&args(&["-h"])).unwrap(), Command::Help));
        assert!(parse_args(&args(&["--device"])).unwrap_err().contains("missing value"));
        assert!(parse_args(&args(&["--bogus"])).unwrap_err().contains("unknown argument"));
        assert!(parse_args(&args(&["--only", "zzz"])).is_err());
    }

    #[test]
    fn unsupported_is_a_skip_and_wrong_is_a_failure() {
        // results
        assert_eq!(classify(&outcome(1e-7, 1e-6), false).0, Status::Ok);
        assert_eq!(classify(&outcome(1e-5, 1e-6), false).0, Status::Failed);
        assert_eq!(classify(&outcome(f64::INFINITY, 1e-6), false).0, Status::Failed);
        assert_eq!(classify(&outcome(f64::NAN, 1e-6), false).0, Status::Failed);
        let cpu_wrong = Ok(Outcome { device: 0.0, cpu: 2.0, tol: 1.0, cpu_tol: 1.0, ms: 0.0 });
        assert_eq!(classify(&cpu_wrong, false).0, Status::Failed, "the CPU backend is held to its tolerance too");
        // unsupported answers
        let op = Err(BackendError::UnsupportedOperation("rope".into()));
        let (status, message) = classify(&op, false);
        assert_eq!((status, message.as_deref()), (Status::Skipped, Some("rope")));
        let quant = Err(BackendError::UnsupportedQuant(QuantType::Q4_K));
        assert_eq!(classify(&quant, false).0, Status::Skipped);
        // ... unless the backend lists the format as supported
        let (status, message) = classify(&quant, true);
        assert_eq!(status, Status::Failed);
        assert!(message.unwrap().contains("lists this weight format as supported"));
        assert_eq!(classify(&op, true).0, Status::Failed);
        // everything else is a failure, never a skip
        for error in [
            BackendError::DriverError("boom".into()),
            BackendError::DeviceNotFound("x".into()),
            BackendError::InvalidShape("x".into()),
            BackendError::BufferError("x".into()),
            BackendError::OutOfMemory { requested: 2, available: 1 },
        ] {
            let text = error.to_string();
            let (status, message) = classify(&Err(error), false);
            assert_eq!((status, message), (Status::Failed, Some(text)));
        }
    }

    #[test]
    fn expected_skips_are_parsed_and_compared() {
        let text = "intro
```expected-skips
# comment
attention | attention bf16 | 2  # why
fuse | add_rms_norm | 1
```
ignored | line | 9
";
        let expected = parse_expected_skips(text).unwrap();
        assert_eq!(expected.len(), 2);
        assert_eq!(expected[&("attention".to_owned(), "attention bf16".to_owned())], 2);
        assert!(parse_expected_skips("```expected-skips
broken line
```").is_err());
        let record = |group: &'static str, op: &str, status| CheckRecord {
            pass: "exact",
            group,
            op: op.to_owned(),
            shape: String::new(),
            status,
            device_error: None,
            cpu_error: None,
            tolerance: None,
            cpu_tolerance: None,
            device_ms: None,
            message: Some("reason".into()),
        };
        let records = [
            record("attention", "attention bf16", Status::Skipped),
            record("attention", "attention bf16", Status::Skipped),
            record("fuse", "add_rms_norm", Status::Skipped),
            record("norm", "rms_norm", Status::Ok),
        ];
        let dir = std::env::temp_dir().join(format!("fllm-expected-skips-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("EXPECTED_SKIPS.md");
        std::fs::write(&path, text).unwrap();
        assert!(check_expected_skips(path.to_str().unwrap(), &records).is_ok());
        let error = check_expected_skips(path.to_str().unwrap(), &records[..2]).unwrap_err();
        assert!(error.contains("fuse / add_rms_norm: expected 1 skipped, none skipped"), "{error}");
        let error = check_expected_skips(path.to_str().unwrap(), &records[1..]).unwrap_err();
        assert!(error.contains("attention bf16: expected 2 skipped, found 1"), "{error}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_check_with_its_own_error_names_the_worst_element() {
        let cpu = CpuBackend::new();
        let filter = Filter::all();
        let mut t = Tester::new(&cpu, &filter, "exact");
        t.enter("kv");
        let err = note_worst(6.5, 3, 10, 1.5, 2.5, 0.5);
        t.record("kv_write", "x", Ok(Outcome { device: err, cpu: 0.0, tol: 1.0, cpu_tol: 1.0, ms: 0.0 }));
        let message = t.records[0].message.clone().unwrap_or_default();
        assert!(message.contains("at element 3 of 10: got 1.5, expected 2.500000e0"), "{message}");
        let err = note_worst(f64::INFINITY, 7, 10, f32::NAN, 1.0, 1.0);
        t.record("kv_write", "y", Ok(Outcome { device: err, cpu: 0.0, tol: 1.0, cpu_tol: 1.0, ms: 0.0 }));
        let message = t.records[1].message.clone().unwrap_or_default();
        assert!(message.contains("NaN at element 7 of 10"), "{message}");
    }

    #[test]
    fn records_count_skips_apart_from_failures() {
        let cpu = CpuBackend::new();
        let filter = Filter::all();
        let mut t = Tester::new(&cpu, &filter, "exact");
        assert!(t.enter("rope"));
        t.record("rope", "a", outcome(0.0, 1.0));
        t.record_skip("rope", "b", "not here");
        t.record("rope", "c", Err(BackendError::UnsupportedOperation("rope".into())));
        let skipped = t.records.iter().filter(|r| r.status == Status::Skipped).count();
        assert_eq!(skipped, 2);
        assert_eq!(t.records[1].message.as_deref(), Some("not here"));
        assert!(t.records.iter().all(|r| r.group == "rope" && r.pass == "exact"));
        assert!(t.finish("test").is_ok(), "skips alone do not fail a run");

        let mut t = Tester::new(&cpu, &filter, "fast");
        t.enter("kv");
        t.record("kv", "a", outcome(5.0, 1.0));
        t.record_skip("kv", "b", "x");
        assert_eq!(t.finish("test").unwrap_err(), "1 test checks failed");
    }

    #[test]
    fn a_claimed_format_that_reports_unsupported_fails() {
        let cpu = CpuBackend::new();
        let filter = Filter::all();
        let mut t = Tester::new(&cpu, &filter, "exact");
        t.enter("linear");
        // the CPU backend lists Q4_K as supported
        assert!(cpu.capabilities().supports_quant(QuantType::Q4_K));
        t.claim = Some(12);
        t.record("linear q4_k", "x", Err(BackendError::UnsupportedQuant(QuantType::Q4_K)));
        assert_eq!(t.records[0].status, Status::Failed);
        // the claim is used up by the check it belongs to
        assert!(t.claim.is_none());
        t.record("linear q4_k", "y", Err(BackendError::UnsupportedQuant(QuantType::Q4_K)));
        assert_eq!(t.records[1].status, Status::Skipped);
    }

    #[test]
    fn tally_and_json_numbers() {
        let record = |group: &'static str, status| CheckRecord {
            pass: "exact",
            group,
            op: String::new(),
            shape: String::new(),
            status,
            device_error: None,
            cpu_error: None,
            tolerance: None,
            cpu_tolerance: None,
            device_ms: None,
            message: None,
        };
        let all = [record("norm", Status::Ok), record("norm", Status::Skipped), record("kv", Status::Failed), record("norm", Status::Ok)];
        let counts = tally(&all);
        assert_eq!(counts, vec![("norm", [2, 0, 1]), ("kv", [0, 1, 0])]);
        assert_eq!(json_number(Some(0.5)), serde_json::json!(0.5));
        assert_eq!(json_number(Some(f64::INFINITY)), serde_json::json!("inf"));
        assert_eq!(json_number(Some(f64::NAN)), serde_json::json!("nan"));
        assert_eq!(json_number(None), serde_json::Value::Null);
    }

    #[test]
    fn filtered_groups_draw_their_own_data() {
        let cpu = CpuBackend::new();
        let filter = Filter::parse(Some("norm,rope"), None).unwrap();
        let mut t = Tester::new(&cpu, &filter, "exact");
        assert!(!t.enter("attention"));
        assert!(t.enter("norm"));
        let first = t.rng.next();
        assert!(t.enter("rope"));
        let second = t.rng.next();
        // a group's data does not depend on which groups ran before it
        let mut other = Tester::new(&cpu, &filter, "exact");
        assert!(other.enter("rope"));
        assert_eq!(other.rng.next(), second);
        assert_ne!(first, second);
        // without a filter the single stream of the earlier versions is kept
        let all = Filter::all();
        let mut plain = Tester::new(&cpu, &all, "exact");
        let before = plain.rng.0;
        assert!(plain.enter("norm") && plain.enter("rope"));
        assert_eq!(plain.rng.0, before);
    }

    /// The test checks the CPU backend against itself: a tolerance or reference that is wrong for the CPU backend would be wrong for
    /// every device too. The groups here are the cheap ones.
    #[test]
    fn the_checks_hold_for_the_cpu_backend_moe_layers() {
        let cpu = CpuBackend::new();
        let filter = Filter::parse(Some("moe"), Some("q4_k")).unwrap();
        let mut t = Tester::new(&cpu, &filter, "exact");
        t.run_all();
        let failed: Vec<String> = t
            .records
            .iter()
            .filter(|r| r.status == Status::Failed)
            .map(|r| format!("{} {} ({:?}, {:?}): {:?}", r.op, r.shape, r.device_error, r.tolerance, r.message))
            .collect();
        assert!(failed.is_empty(), "{failed:#?}");
    }

    #[test]
    fn the_checks_hold_for_the_cpu_backend() {
        let cpu = CpuBackend::new();
        let filter = Filter::parse(Some("norm,rope,kv,sampling,attention"), None).unwrap();
        let mut t = Tester::new(&cpu, &filter, "exact");
        t.run_all();
        t.run_fast();
        let failed: Vec<String> = t
            .records
            .iter()
            .filter(|r| r.status == Status::Failed)
            .map(|r| format!("{} {} ({:?}, {:?}): {:?}", r.op, r.shape, r.device_error, r.tolerance, r.message))
            .collect();
        assert!(failed.is_empty(), "{failed:#?}");
        assert!(t.records.len() > 40, "{} checks", t.records.len());
        for group in ["norm", "rope", "kv", "attention"] {
            assert!(t.records.iter().any(|r| r.group == group && r.status == Status::Ok), "{group} ran");
        }
        // the CPU backend has no device-side argmax: the check is skipped, not failed
        assert!(t.records.iter().any(|r| r.group == "sampling"));
        assert!(t.records.iter().all(|r| ["norm", "rope", "kv", "sampling", "attention"].contains(&r.group)));
    }

    #[test]
    fn the_quant_filter_limits_the_linear_checks() {
        let cpu = CpuBackend::new();
        let filter = Filter::parse(Some("linear,embedding"), Some("q8_0,f32")).unwrap();
        let mut t = Tester::new(&cpu, &filter, "exact");
        t.run_all();
        assert!(!t.records.is_empty());
        assert!(t.records.iter().all(|r| ["linear", "embedding"].contains(&r.group)));
        assert!(t.records.iter().all(|r| r.op.ends_with("q8_0") || r.op.ends_with("f32") || r.op == "linear q8_0"), "{:?}", t.records.iter().map(|r| &r.op).collect::<Vec<_>>());
        let failed = t.records.iter().filter(|r| r.status == Status::Failed).count();
        assert_eq!(failed, 0);
    }
}

#[cfg(test)]
mod extra_group_tests {
    use super::*;

    /// A backend without a fast pass meets the sampling, host and moe groups in its only pass; one with a fast pass does not meet them twice.
    #[test]
    fn extra_groups_run_where_no_fast_pass_follows() {
        let cpu = CpuBackend::new();
        let filter = Filter::parse(Some("sampling,host,moe"), None).unwrap();
        let mut single = Tester::new(&cpu, &filter, "exact");
        single.extra_groups = true;
        single.run_all();
        for group in ["sampling", "host", "moe"] {
            assert!(single.records.iter().any(|r| r.group == group), "{group} did not run");
        }
        // whatever the CPU backend does not implement is a skip; nothing may fail
        let failed: Vec<_> = single.records.iter().filter(|r| r.status == Status::Failed).map(|r| (&r.op, &r.shape, &r.message)).collect();
        assert!(failed.is_empty(), "{failed:#?}");

        let mut with_fast_pass = Tester::new(&cpu, &filter, "exact");
        with_fast_pass.run_all();
        assert!(with_fast_pass.records.is_empty(), "the exact pass of cuda / hip leaves these groups to the fast pass");
        with_fast_pass.run_fast();
        assert!(!with_fast_pass.records.is_empty());
    }

    
    

    /// The block-wise FP8 checks (every tile shape, odd shapes, grouped experts) run against the CPU backend, whose kernel is `cpu_fp8`, and pass:
    /// the f64 reference, the random weights and the shapes of the checks are right before they are run on a GPU.
    #[test]
    fn block_wise_fp8_checks_pass_on_the_cpu_backend() {
        let cpu = CpuBackend::new();
        let filter = Filter::parse(Some("linear,moe"), Some("fp8b")).unwrap();
        let mut t = Tester::new(&cpu, &filter, "exact");
        t.extra_groups = true;
        t.run_all();
        t.run_fast();
        let fp8b: Vec<_> = t.records.iter().filter(|r| r.op.contains("fp8b")).collect();
        assert!(fp8b.len() > 40, "only {} fp8b checks ran", fp8b.len());
        let failed: Vec<_> = fp8b.iter().filter(|r| r.status == Status::Failed).map(|r| (&r.op, &r.shape, &r.message)).collect();
        assert!(failed.is_empty(), "{failed:#?}");
        let skipped = fp8b.iter().filter(|r| r.status == Status::Skipped).count();
        assert_eq!(skipped, 0, "the CPU backend runs every fp8b check");
    }
}
