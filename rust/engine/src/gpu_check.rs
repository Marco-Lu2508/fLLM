//! `fllm-rs doctor --gpu` and the Studio button "Test this GPU": a few seconds of real work on every GPU, compared with the CPU.
//!
//! The check uploads one quantized weight matrix (Q4_K, 4096 x 4096, the format most models are shipped in) to the device, multiplies it with
//! one token (decode) and with 64 tokens (prompt), and does the same on the CPU with the engine's own CPU kernels. It then says in one line
//! whether the GPU does the work: `OK (12.3 times faster than the CPU)`, or `FALLS BACK: <reason>` when the device cannot be created, a
//! kernel is missing and the backend hands the operation to the CPU, the kernel launch fails, or the result is wrong; or `SLOWER THAN THE CPU`.

use crate::backend::{Backend, DeviceBuffer, Stream};
use std::sync::Arc;
use std::time::{Duration, Instant};

const ROWS: usize = 4096;
const COLS: usize = 4096;
const KIND: u32 = 12; // Q4_K
/// Each timing loop stops after this long (or after `MAX_ITERATIONS`).
const BUDGET: Duration = Duration::from_millis(350);
const MAX_ITERATIONS: usize = 400;

/// The verdict for one device.
#[derive(Clone, Debug)]
pub struct Verdict {
    /// The `--device` value of the device.
    pub spec: String,
    pub name: String,
    pub backend: String,
    pub ok: bool,
    /// `ok` (works and is faster), `slow` (works, slower than the CPU), `falls_back`, `wrong`, `error`.
    pub status: &'static str,
    /// The one line: `OK (3.1 times faster than the CPU)`, `FALLS BACK: ...`.
    pub text: String,
    pub decode_speedup: Option<f64>,
    pub prompt_speedup: Option<f64>,
    pub device_ms: Option<(f64, f64)>,
    pub cpu_ms: Option<(f64, f64)>,
}

impl Verdict {
    fn bad(spec: &str, name: &str, backend: &str, status: &'static str, text: String) -> Self {
        Self { spec: spec.to_owned(), name: name.to_owned(), backend: backend.to_owned(), ok: false, status, text, decode_speedup: None, prompt_speedup: None, device_ms: None, cpu_ms: None }
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "device": self.spec,
            "name": self.name,
            "backend": self.backend,
            "ok": self.ok,
            "status": self.status,
            "verdict": self.text,
            "decode_speedup": self.decode_speedup,
            "prompt_speedup": self.prompt_speedup,
            "device_ms_decode_prompt": self.device_ms.map(|m| [m.0, m.1]),
            "cpu_ms_decode_prompt": self.cpu_ms.map(|m| [m.0, m.1]),
        })
    }

    /// `cuda:0  NVIDIA GeForce RTX 2070 SUPER: OK (...)`
    pub fn line(&self) -> String {
        format!("{:<14} {}: {}", self.spec, self.name, self.text)
    }
}

/// Creates the backend a `--device` value names, the way the decoder does (native first, then CUDA, then wgpu).
fn create(spec: &str) -> Result<Arc<dyn Backend>, String> {
    let parsed = crate::devices::parse_spec(spec)?;
    let index = parsed.index.unwrap_or(0);
    let kind = parsed.kind.as_str();
    let wgpu_only = kind.strip_prefix("wgpu-");
    if wgpu_only.is_none() {
        if let Some(backend) = crate::backend::create_native_backend(kind, index)? {
            return Ok(backend);
        }
    }
    #[cfg(feature = "cuda")]
    if kind == "cuda" {
        return crate::backend::cuda::CudaBackend::new(index).map(|b| Arc::new(b) as Arc<dyn Backend>).map_err(|e| e.to_string());
    }
    let api = wgpu_only.unwrap_or(kind);
    #[cfg(feature = "rust-gpu")]
    if matches!(api, "vulkan" | "metal" | "dx12" | "opengl" | "gl" | "gpu" | "wgpu") {
        return crate::backend::wgpu_backend::WgpuBackend::new(api, index).map(|b| Arc::new(b) as Arc<dyn Backend>).map_err(|e| e.to_string());
    }
    let _ = api;
    Err(format!("the {kind} backend is not compiled into this binary (compiled: {})", crate::devices::compiled_backends().join(", ")))
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A deterministic pseudo-random sequence in `[-1, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn weights() -> Vec<u8> {
    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
    let mut out = Vec::with_capacity(ROWS * COLS / 256 * 144);
    let mut row = vec![0.0f32; COLS];
    for _ in 0..ROWS {
        for v in &mut row {
            *v = rng.next() * 0.5;
        }
        fllm_checkpoint::quantize_row(fllm_checkpoint::QType::Q4K, &row, &mut out);
    }
    out
}

fn inputs(batch: usize) -> Vec<f32> {
    let mut rng = Lcg(0x1234_5678_9abc_def1);
    (0..batch * COLS).map(|_| rng.next()).collect()
}

/// Calls to operations that fell back to the CPU since the check began (the minor, by-design ones excluded).
fn fallback_calls() -> Vec<(String, String, u64)> {
    crate::placement::report().ops.into_iter().filter(|o| o.0 != "sampling_logits").collect()
}

fn time_loop(mut f: impl FnMut() -> Result<(), String>) -> Result<f64, String> {
    let start = Instant::now();
    let mut n = 0usize;
    while n < MAX_ITERATIONS && (n == 0 || start.elapsed() < BUDGET) {
        f()?;
        n += 1;
    }
    Ok(start.elapsed().as_secs_f64() * 1e3 / n as f64)
}

struct DeviceRun {
    ms: (f64, f64),
    decode_out: Vec<f32>,
    prompt_out: Vec<f32>,
    note: Option<String>,
}

fn run_device(backend: &dyn Backend, w: &[u8], x1: &[f32], x64: &[f32]) -> Result<DeviceRun, String> {
    let mut stream: Box<dyn Stream> = backend.create_stream().map_err(|e| e.to_string())?;
    let weight = backend
        .upload_weight("gpu-check", w, crate::backend::QuantType::from_gguf_type_id(KIND), ROWS, COLS, None)
        .map_err(|e| e.to_string())?;
    let note = (weight.quant_type != crate::backend::QuantType::from_gguf_type_id(KIND)).then(|| format!("the device stores the weights as {:?} instead of Q4_K", weight.quant_type));
    let mut one = |x: &[f32], batch: usize| -> Result<(f64, Vec<f32>), String> {
        let mut xb: DeviceBuffer = backend.allocate_buffer(x.len() * 4).map_err(|e| e.to_string())?;
        stream.upload(&mut xb, &f32_bytes(x)).map_err(|e| e.to_string())?;
        let mut out: DeviceBuffer = backend.allocate_buffer(batch * ROWS * 4).map_err(|e| e.to_string())?;
        // the first call compiles pipelines / loads kernels: not timed
        stream.quantized_linear(&mut out, &xb, &weight, None).map_err(|e| e.to_string())?;
        stream.synchronize().map_err(|e| e.to_string())?;
        let ms = time_loop(|| {
            stream.quantized_linear(&mut out, &xb, &weight, None).map_err(|e| e.to_string())?;
            stream.synchronize().map_err(|e| e.to_string())
        })?;
        let mut bytes = vec![0u8; batch * ROWS * 4];
        stream.download(&mut bytes, &out).map_err(|e| e.to_string())?;
        Ok((ms, bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()))
    };
    let (decode_ms, decode_out) = one(x1, 1)?;
    let (prompt_ms, prompt_out) = one(x64, 64)?;
    Ok(DeviceRun { ms: (decode_ms, prompt_ms), decode_out, prompt_out, note })
}

/// Milliseconds per call (decode, prompt) and the two results.
type CpuRun = ((f64, f64), Vec<f32>, Vec<f32>);

fn run_cpu(w: &[u8], x1: &[f32], x64: &[f32]) -> Result<CpuRun, String> {
    let mut decode_out = Vec::new();
    let mut prompt_out = Vec::new();
    let decode_ms = time_loop(|| {
        decode_out = crate::cpu_quantized::linear(w, KIND, x1, ROWS, COLS, None).ok_or("the CPU kernels do not take Q4_K")?;
        Ok(())
    })?;
    let prompt_ms = time_loop(|| {
        prompt_out = crate::cpu_quantized::linear(w, KIND, x64, ROWS, COLS, None).ok_or("the CPU kernels do not take Q4_K")?;
        Ok(())
    })?;
    Ok(((decode_ms, prompt_ms), decode_out, prompt_out))
}

/// Largest error relative to the largest magnitude of the reference.
fn relative_error(have: &[f32], want: &[f32]) -> f64 {
    if have.len() != want.len() || have.iter().any(|v| !v.is_finite()) {
        return f64::INFINITY;
    }
    let scale = want.iter().fold(0.0f64, |m, v| m.max(f64::from(v.abs()))).max(1e-6);
    have.iter().zip(want).fold(0.0f64, |m, (h, w)| m.max((f64::from(*h) - f64::from(*w)).abs())) / scale
}

fn times(x: f64) -> String {
    if x >= 10.0 { format!("{x:.0}") } else { format!("{x:.1}") }
}

/// Checks the device `spec` (a `--device` value like `cuda:0` or `vulkan:1`).
pub fn check_device(spec: &str, name: &str) -> Verdict {
    let backend_id = spec.split(':').next().unwrap_or(spec).to_owned();
    let mut verdict = check_inner(spec, name, &backend_id);
    verdict.backend = backend_id;
    verdict
}

fn check_inner(spec: &str, name: &str, backend_id: &str) -> Verdict {
    let backend = match create(spec) {
        Ok(b) => b,
        Err(e) => return Verdict::bad(spec, name, backend_id, "falls_back", format!("FALLS BACK: the device cannot be used at all, every model would run on the CPU ({e})")),
    };
    let w = weights();
    let (x1, x64) = (inputs(1), inputs(64));
    let before: u64 = fallback_calls().iter().map(|o| o.2).sum();
    let run = match run_device(&*backend, &w, &x1, &x64) {
        Ok(r) => r,
        Err(e) => return Verdict::bad(spec, name, backend_id, "falls_back", format!("FALLS BACK: a quantized matrix product fails on this device, so the work goes to the CPU or stops ({e})")),
    };
    let after = fallback_calls();
    if after.iter().map(|o| o.2).sum::<u64>() > before {
        let (op, reason, _) = &after[0];
        return Verdict::bad(spec, name, backend_id, "falls_back", format!("FALLS BACK: the operation '{op}' runs on the CPU: {reason}"));
    }
    let (cpu_ms, cpu_decode, cpu_prompt) = match run_cpu(&w, &x1, &x64) {
        Ok(r) => r,
        Err(e) => return Verdict::bad(spec, name, backend_id, "error", format!("ERROR: the CPU reference failed ({e})")),
    };
    // the fast GPU kernels use f16 / int8 activations: a few thousandths of the largest value are normal, a percent is a wrong kernel
    let error = relative_error(&run.decode_out, &cpu_decode).max(relative_error(&run.prompt_out, &cpu_prompt));
    if error > 0.03 {
        return Verdict::bad(spec, name, backend_id, "wrong", format!("WRONG RESULT: the device result differs from the CPU result by {:.1} % of the largest value (a kernel is broken on this GPU/driver)", error * 100.0));
    }
    let decode = cpu_ms.0 / run.ms.0.max(1e-6);
    let prompt = cpu_ms.1 / run.ms.1.max(1e-6);
    let best = decode.max(prompt);
    let (ok, status, text) = if best >= 1.0 {
        (true, "ok", format!("OK ({} times faster than the CPU; decode {}x, prompt {}x)", times(best), times(decode), times(prompt)))
    } else {
        (false, "slow", format!("SLOWER THAN THE CPU ({}x at decode, {}x at prompt): the device works but does not help; check the driver, the power mode and that the display is not the only user of this GPU", times(decode), times(prompt)))
    };
    let text = match run.note {
        Some(n) => format!("{text} [{n}]"),
        None => text,
    };
    Verdict {
        spec: spec.to_owned(),
        name: name.to_owned(),
        backend: backend_id.to_owned(),
        ok,
        status,
        text,
        decode_speedup: Some(decode),
        prompt_speedup: Some(prompt),
        device_ms: Some(run.ms),
        cpu_ms: Some(cpu_ms),
    }
}

/// The devices to check: every accelerator of the inventory (or the one `only` names).
fn targets(only: Option<&str>) -> Result<Vec<(String, String)>, String> {
    let all: Vec<(String, String)> = crate::devices::inventory()
        .into_iter()
        .filter(|e| e.api != crate::devices::Api::Host && e.info.dev_type != crate::devices::DeviceType::Accelerator)
        .map(|e| (e.spec, e.info.name))
        .collect();
    match only {
        None | Some("all") => Ok(all),
        Some(spec) => {
            let wanted = spec.to_ascii_lowercase();
            let found: Vec<(String, String)> = all.into_iter().filter(|(s, _)| s.to_ascii_lowercase() == wanted).collect();
            if found.is_empty() {
                Err(format!("no GPU named {spec}; `fllm-rs devices` lists the --device values"))
            } else {
                Ok(found)
            }
        }
    }
}

/// Checks `only` (or every GPU) and returns one verdict per device.
pub fn check(only: Option<&str>) -> Result<Vec<Verdict>, String> {
    Ok(targets(only)?.into_iter().map(|(spec, name)| check_device(&spec, &name)).collect())
}

pub const HELP: &str = "usage: fllm-rs doctor --gpu [--device cuda:0|vulkan:1|all]
Runs a few seconds of real work (a quantized matrix product, one token and 64 tokens) on every GPU and on the CPU, and says per device
in one line whether the GPU does the work:
  OK (12 times faster than the CPU)       the GPU computes and is faster
  FALLS BACK: <reason>                    the device cannot be used, or an operation goes to the CPU
  SLOWER THAN THE CPU                     it works, but the CPU is faster for this product
  WRONG RESULT                            a kernel gives a different answer than the CPU
Exit code 1 when a device falls back or gives wrong results.";

/// `fllm-rs doctor --gpu`
pub fn run_cli(args: &[String]) -> Result<i32, String> {
    let only = args.iter().position(|a| a == "--device").and_then(|i| args.get(i + 1)).map(String::as_str);
    let targets = targets(only)?;
    if targets.is_empty() {
        println!("no GPU found: every model runs on the CPU (`fllm-rs devices` shows what the program can see; install the vendor driver)");
        return Ok(0);
    }
    let mut failed = false;
    for (spec, name) in targets {
        let verdict = check_device(&spec, &name);
        println!("{}", verdict.line());
        failed |= matches!(verdict.status, "falls_back" | "wrong");
    }
    Ok(i32::from(failed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_error_measure_flags_wrong_and_non_finite_results() {
        assert!(relative_error(&[1.0, 2.0], &[1.0, 2.0]) < 1e-9);
        assert!(relative_error(&[1.0, 2.5], &[1.0, 2.0]) > 0.1);
        assert!(relative_error(&[f32::NAN, 2.0], &[1.0, 2.0]).is_infinite());
        assert!(relative_error(&[1.0], &[1.0, 2.0]).is_infinite());
    }

    #[test]
    fn the_cpu_side_of_the_check_runs_and_the_weights_have_the_q4k_size() {
        let w = weights();
        assert_eq!(w.len(), ROWS * COLS / 256 * 144);
        let ((decode, prompt), a, b) = run_cpu(&w, &inputs(1), &inputs(64)).unwrap();
        assert!(decode > 0.0 && prompt > 0.0);
        assert_eq!((a.len(), b.len()), (ROWS, 64 * ROWS));
    }

    #[test]
    fn an_unknown_device_is_reported_not_guessed() {
        assert!(check(Some("nosuch:9")).is_err());
        let v = check_device("nosuch:9", "nothing");
        assert!(!v.ok && v.text.starts_with("FALLS BACK"), "{}", v.text);
    }
}
