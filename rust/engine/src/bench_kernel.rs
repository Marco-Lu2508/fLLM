//! `fllm-rs bench-kernel`: time one kernel on synthetic data (no model needed): `--op marlin_gemv` is a GPU kernel,
//! `--op cpu_iq` the CPU kernels of the lattice IQ formats against the row-decode path they replace.
//!
//! `--op marlin_gemv` (CUDA builds): the Marlin 4-bit GEMM and the small-M GEMV of `cuda/marlin_vllm/fllm_marlin_gemv.cu` on random weights
//! of one shape. Prints microseconds per call of the Marlin GEMM with and without its FP32 reduction scratch and of the GEMV, the
//! bandwidth the GEMV reaches and the largest difference between the two results (random bytes are a valid weight matrix for both).

pub const HELP: &str = "usage: fllm-rs bench-kernel --op marlin_gemv|cpu_iq [--m 1] [--n 2048] [--k 2048] [--iters 200] [--kind awq|gptq] [--group 128] [--device 0] [--iq all|ref|iq2_xxs|iq2_xs|iq3_xxs|iq3_s|iq2_s|q4_k|q3_k|iq4_xs]
  --op marlin_gemv  Marlin GEMM against the small-M GEMV (FLLM_MARLIN_SMALL_M) for an [m, k] x [k, n] product with 4-bit weights
  --op cpu_iq       CPU projection of --n rows by --k columns (a multiple of 256) for --m tokens in the lattice IQ formats (--iq; ref = Q4_K, Q3_K, IQ4_XS as a yardstick): the packed
                    integer kernels against the row-decode path (dequantize each row to f32, f32 dot). Prints rows/s, weight GB/s and the speed-up
  --m / --n / --k   product shape (m 1 to 4; n a multiple of 64; k a multiple of 128)
  --iters N         timed launches per kernel (default 200), back to back on one stream
Output: us per call of Marlin (FP32 reduce), Marlin (FP16 reduce, no scratch), the GEMV, GB/s of the GEMV over the weight bytes, and the largest
difference of the results relative to the largest result.";

#[derive(Debug, PartialEq)]
struct Args {
    op: String,
    m: i32,
    n: i32,
    k: i32,
    iters: i32,
    kind: i32,
    group: i32,
    device: i32,
    iq: String,
}

fn parse(args: &[String]) -> Result<Option<Args>, String> {
    let mut a = Args { op: String::new(), m: 1, n: 2048, k: 2048, iters: 200, kind: 0, group: 128, device: 0, iq: "all".into() };
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        if flag == "--help" || flag == "-h" {
            return Ok(None);
        }
        let value = args.get(i + 1).ok_or_else(|| format!("missing value for {flag}"))?;
        let number = |what: &str| value.parse::<i32>().map_err(|_| format!("{what} needs an integer, got '{value}'"));
        match flag {
            "--op" => a.op = value.clone(),
            "--m" => a.m = number("--m")?,
            "--n" => a.n = number("--n")?,
            "--k" => a.k = number("--k")?,
            "--iters" => a.iters = number("--iters")?,
            "--group" => a.group = number("--group")?,
            "--device" => a.device = number("--device")?,
            "--iq" => a.iq = value.clone(),
            "--kind" => {
                a.kind = match value.as_str() {
                    "awq" | "0" => 0,
                    "gptq" | "1" => 1,
                    other => return Err(format!("--kind is awq or gptq, got '{other}'")),
                }
            }
            other => return Err(format!("unknown flag {other}")),
        }
        i += 2;
    }
    if a.op.is_empty() {
        return Err("--op is required (marlin_gemv or cpu_iq)".into());
    }
    Ok(Some(a))
}

/// Entry point of `fllm-rs bench-kernel`.
pub fn run(args: &[String]) -> Result<(), String> {
    let Some(a) = parse(args)? else {
        println!("{HELP}");
        return Ok(());
    };
    match a.op.as_str() {
        "marlin_gemv" => marlin_gemv(&a),
        "cpu_iq" => cpu_iq(&a),
        other => Err(format!("unknown --op {other} (marlin_gemv, cpu_iq)")),
    }
}

#[cfg(feature = "cuda")]
fn marlin_gemv(a: &Args) -> Result<(), String> {
    extern "C" {
        fn fllm_marlin_gemv_bench(dev: i32, m: i32, n: i32, k: i32, kind: i32, group: i32, iters: i32, out: *mut f64) -> i32;
        fn fllm_cuda_last_error() -> *const std::ffi::c_char;
    }
    let mut out = [0f64; 8];
    // SAFETY: `out` has room for the eight values the routine writes.
    let rc = unsafe { fllm_marlin_gemv_bench(a.device, a.m, a.n, a.k, a.kind, a.group, a.iters, out.as_mut_ptr()) };
    match rc {
        0 => {}
        -1 => return Err(format!("shape m={} n={} k={} group={} is not supported by the GEMV (m 1..4, n % 64, k % 128, group 32/64/128)", a.m, a.n, a.k, a.group)),
        _ => {
            // SAFETY: the library returns a NUL-terminated buffer owned by thread-local storage.
            let why = unsafe { std::ffi::CStr::from_ptr(fllm_cuda_last_error()) }.to_string_lossy().into_owned();
            return Err(format!("benchmark failed ({rc}) {why}"));
        }
    }
    let weight_mb = out[4] / 1e6;
    println!(
        "marlin_gemv m={} n={} k={} {} group {}: weights {weight_mb:.2} MB, {} iterations",
        a.m,
        a.n,
        a.k,
        if a.kind == 0 { "awq" } else { "gptq" },
        a.group,
        a.iters
    );
    println!("  marlin (fp32 reduce)   {:8.2} us/call  {:7.0} GB/s", out[0], out[4] / (out[0] * 1e-6) / 1e9);
    println!("  marlin (fp16 reduce)   {:8.2} us/call  {:7.0} GB/s", out[6], out[4] / (out[6] * 1e-6) / 1e9);
    println!("  gemv                   {:8.2} us/call  {:7.0} GB/s", out[1], out[5]);
    println!("  speed-up gemv over marlin: {:.2}x", out[0] / out[1]);
    println!("  max |marlin - gemv| = {:.3e} (largest result {:.3e}, relative {:.2e})", out[2], out[3], if out[3] > 0.0 { out[2] / out[3] } else { f64::NAN });
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn marlin_gemv(_: &Args) -> Result<(), String> {
    Err("marlin_gemv needs a build with --features cuda".into())
}

/// CPU model name, best effort, for the benchmark header.
fn cpu_name() -> String {
    if let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") {
        if let Some(name) = text
            .lines()
            .find_map(|l| l.strip_prefix("model name").and_then(|r| r.split_once(':')).map(|(_, v)| v.trim().to_owned()))
        {
            return name;
        }
    }
    std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "unknown CPU".into())
}

/// `--op cpu_iq`: the packed integer kernels of `cpu_iqgrid` against the row-decode path of
/// `Linear::forward` (every output element dequantizes its weight row to f32 and takes an f32 dot).
fn cpu_iq(a: &Args) -> Result<(), String> {
    use crate::cpu_selftest::{random_weights, Lcg};
    use rayon::prelude::*;
    use std::time::Instant;
    let (rows, cols, batch) = (a.n.max(1) as usize, a.k.max(1) as usize, a.m.max(1) as usize);
    if !cols.is_multiple_of(256) {
        return Err("--k must be a multiple of 256".into());
    }
    // the five lattice formats, then K formats as a yardstick (`--iq ref` or by name)
    let formats: [(&str, u32); 8] = [
        ("iq2_xxs", 16),
        ("iq2_xs", 17),
        ("iq3_xxs", 18),
        ("iq3_s", 21),
        ("iq2_s", 22),
        ("q4_k", 12),
        ("q3_k", 11),
        ("iq4_xs", 23),
    ];
    let chosen: Vec<_> = formats
        .iter()
        .enumerate()
        .filter(|(i, (n, _))| (a.iq == "all" && *i < 5) || (a.iq == "ref" && *i >= 5) || a.iq == *n)
        .map(|(_, f)| f)
        .collect();
    if chosen.is_empty() {
        return Err(format!("--iq: unknown format '{}'", a.iq));
    }
    println!(
        "cpu_iq: {} | isa {} | {} threads | {rows} rows x {cols} cols, {batch} token(s), {} iterations",
        cpu_name(),
        crate::cpu_kernels::isa(),
        rayon::current_num_threads(),
        a.iters.max(1)
    );
    let mut rng = Lcg(0x1d2e3f);
    let x = rng.vector(batch * cols, 1.0);
    for (name, kind) in chosen {
        let weights = random_weights(*kind, rows, cols, &mut rng);
        let row_bytes = weights.len() / rows;
        let native = |x: &[f32]| crate::cpu_quantized::linear(&weights, *kind, x, rows, cols, None);
        let Some(first) = native(&x) else {
            println!("{name:8} no packed integer kernel on this CPU (the row-decode path runs)");
            continue;
        };
        let row_decode = || -> Result<Vec<f32>, String> {
            let mut out = vec![0f32; batch * rows];
            out.par_chunks_mut(64).enumerate().try_for_each(|(chunk, part)| -> Result<(), String> {
                for (j, y) in part.iter_mut().enumerate() {
                    let index = chunk * 64 + j;
                    let (token, row) = (index / rows, index % rows);
                    let values = crate::compute::dequantize(&weights[row * row_bytes..(row + 1) * row_bytes], *kind, cols)?;
                    *y = x[token * cols..(token + 1) * cols].iter().zip(&values).map(|(a, b)| a * b).sum();
                }
                Ok(())
            })?;
            Ok(out)
        };
        let old = row_decode()?;
        // 8-bit activations against f32 ones: the two differ by the activation rounding only
        let peak = old.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-9);
        let worst = first.iter().zip(&old).fold(0f32, |m, (a, b)| m.max((a - b).abs())) / peak;
        let iters = a.iters.max(1) as usize;
        let time = |runs: usize, f: &dyn Fn()| {
            f();
            let start = Instant::now();
            for _ in 0..runs {
                f();
            }
            start.elapsed().as_secs_f64() / runs as f64
        };
        let new_s = time(iters, &|| {
            std::hint::black_box(native(&x));
        });
        let old_s = time((iters / 10).max(2), &|| {
            std::hint::black_box(row_decode().ok());
        });
        let rate = |s: f64| (rows * batch) as f64 / s;
        let gbs = |s: f64| weights.len() as f64 / s / 1e9;
        println!(
            "{name:8} packed kernel {:9.3} ms  {:11.0} rows/s  {:6.2} GB/s | row decode {:9.3} ms  {:9.0} rows/s  {:6.3} GB/s | {:6.1}x | max diff {:.1e} of peak",
            new_s * 1e3,
            rate(new_s),
            gbs(new_s),
            old_s * 1e3,
            rate(old_s),
            gbs(old_s),
            old_s / new_s,
            worst
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn parses_the_documented_flags() {
        let a = parse(&args(&["--op", "marlin_gemv", "--m", "2", "--n", "4096", "--k", "1024", "--iters", "10", "--kind", "gptq", "--group", "64"]))
            .unwrap()
            .unwrap();
        assert_eq!(a, Args { op: "marlin_gemv".into(), m: 2, n: 4096, k: 1024, iters: 10, kind: 1, group: 64, device: 0, iq: "all".into() });
    }

    #[test]
    fn rejects_bad_input() {
        assert!(parse(&args(&["--m", "1"])).is_err());
        assert!(parse(&args(&["--op", "x", "--m"])).is_err());
        assert!(parse(&args(&["--op", "x", "--kind", "q4"])).is_err());
        assert!(parse(&args(&["--op", "x", "--bogus", "1"])).is_err());
        assert!(parse(&args(&["--help"])).unwrap().is_none());
    }
}
