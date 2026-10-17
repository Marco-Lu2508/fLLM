//! `fllm-rs bench-kernel`: time one GPU kernel on synthetic data (no model needed).
//!
//! `--op marlin_gemv` (CUDA builds): the Marlin 4-bit GEMM and the small-M GEMV of `cuda/marlin_vllm/fllm_marlin_gemv.cu` on random weights
//! of one shape. Prints microseconds per call of the Marlin GEMM with and without its FP32 reduction scratch and of the GEMV, the
//! bandwidth the GEMV reaches and the largest difference between the two results (random bytes are a valid weight matrix for both).

pub const HELP: &str = "usage: fllm-rs bench-kernel --op marlin_gemv [--m 1] [--n 2048] [--k 2048] [--iters 200] [--kind awq|gptq] [--group 128] [--device 0]
  --op marlin_gemv  Marlin GEMM against the small-M GEMV (FLLM_MARLIN_SMALL_M) for an [m, k] x [k, n] product with 4-bit weights
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
}

fn parse(args: &[String]) -> Result<Option<Args>, String> {
    let mut a = Args { op: String::new(), m: 1, n: 2048, k: 2048, iters: 200, kind: 0, group: 128, device: 0 };
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
        return Err("--op is required (marlin_gemv)".into());
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
        other => Err(format!("unknown --op {other} (marlin_gemv)")),
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
        assert_eq!(a, Args { op: "marlin_gemv".into(), m: 2, n: 4096, k: 1024, iters: 10, kind: 1, group: 64, device: 0 });
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
