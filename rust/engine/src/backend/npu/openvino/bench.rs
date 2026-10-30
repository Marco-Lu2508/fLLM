//! `fllm-rs npu-bench`: times the Intel NPU on synthetic projections (no model needed).
//!
//! For each weight variant (`q4_0:g32` exact int4, `q4_0:g128`, `q4_0:ch` channel-wise int4, `q8_0:g32`, `q8_0:ch`, `f16`) and each
//! batch it builds the projection graph, compiles it (with `--cache`, run the command twice to see the compile time with a warm blob
//! cache), runs it and compares the speed with the CPU kernels of the host decoder on the same Q4_0 / Q8_0 / F16 data.
//! The numbers decide the defaults of the offload planner.

use super::devices;
use super::model::build_linear_ir;
use super::quant::{convert, QuantPolicy};
use super::session::Runtime;
use crate::backend::QuantType;
use half::f16;
use std::time::Instant;

pub const HELP: &str = "usage: fllm-rs npu-bench [--n 2048] [--k 2048] [--batches 64,128,256,512] [--variants q4_0:i8,q4_0:ch,q4_0:g128,f16] [--iters 20] [--cache DIR] [--props K=V,K=V]
Times one projection (n outputs, k inputs) on the Intel NPU through OpenVINO against the CPU kernels: compile time, time per call and GOPS.
  --cache DIR   OpenVINO compiled-model cache (run twice to see the compile time with a warm cache); default: no cache
  --variants    weight formats on the NPU: q4_0:i8 (INT8 per channel, the default of the offload), q4_0:ch (INT4 per channel), q4_0:g128, q4_0:g32 (exact), q8_0:i8, f16";

fn variant(spec: &str) -> Result<(QuantType, QuantPolicy), String> {
    let (quant, group) = spec.split_once(':').unwrap_or((spec, ""));
    let quant = match quant {
        "q4_0" => QuantType::Q4_0,
        "q8_0" => QuantType::Q8_0,
        "f16" => QuantType::F16,
        other => return Err(format!("unknown variant format '{other}'")),
    };
    let policy = match group {
        "" => QuantPolicy::default(),
        "g32" => QuantPolicy { group: 32, lossless: true, eight: false },
        "ch" => QuantPolicy { group: 0, lossless: false, eight: false },
        "i8" => QuantPolicy::int8_channel(),
        g if g.starts_with('g') => QuantPolicy { group: g[1..].parse().map_err(|_| format!("bad group '{g}'"))?, lossless: false, eight: false },
        other => return Err(format!("bad variant suffix '{other}'")),
    };
    Ok((quant, policy))
}

fn synthetic(quant: QuantType, rows: usize, cols: usize) -> Vec<u8> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    match quant {
        QuantType::Q4_0 | QuantType::Q8_0 => {
            let payload = if quant == QuantType::Q4_0 { 16 } else { 32 };
            let mut raw = Vec::with_capacity(rows * cols / 32 * (payload + 2));
            for _ in 0..rows * cols / 32 {
                raw.extend_from_slice(&f16::from_f32(0.004 + (next() % 100) as f32 * 1e-4).to_le_bytes());
                raw.extend((0..payload).map(|_| next() as u8));
            }
            raw
        }
        _ => (0..rows * cols).flat_map(|_| f16::from_f32(((next() % 2000) as f32 - 1000.0) * 1e-4).to_le_bytes()).collect(),
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values[values.len() / 2]
}

pub fn run(args: &[String]) -> Result<(), String> {
    let (mut n, mut k, mut iters) = (2048usize, 2048usize, 20usize);
    let mut batches = vec![64usize, 128, 256, 512];
    let mut variants = vec!["q4_0:i8".to_owned(), "q4_0:ch".to_owned(), "q4_0:g128".to_owned(), "f16".to_owned()];
    let mut cache: Option<String> = None;
    let mut extra: Vec<(String, String)> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        if flag == "--help" || flag == "-h" {
            println!("{HELP}");
            return Ok(());
        }
        let value = args.get(i + 1).ok_or_else(|| format!("missing value for {flag}"))?;
        let number = |what: &str| value.parse::<usize>().map_err(|_| format!("{what} needs an integer, got '{value}'"));
        match flag {
            "--n" => n = number("--n")?,
            "--k" => k = number("--k")?,
            "--iters" => iters = number("--iters")?.max(1),
            "--batches" => batches = value.split(',').map(|b| b.trim().parse().map_err(|_| format!("bad batch '{b}'"))).collect::<Result<_, _>>()?,
            "--variants" => variants = value.split(',').map(|v| v.trim().to_owned()).collect(),
            "--cache" => cache = Some(value.clone()),
            "--props" => extra = value.split(',').filter_map(|p| p.split_once('=')).map(|(a, b)| (a.trim().to_owned(), b.trim().to_owned())).take(2).collect(),
            other => return Err(format!("unknown flag {other}")),
        }
        i += 2;
    }
    let runtime = Runtime::shared()?;
    let found = devices::enumerate();
    let device = found.first().ok_or("OpenVINO lists no NPU")?;
    println!("OpenVINO {} on {} ({}), platform {}, driver {}", super::session::runtime_version()?, device.name, device.full_name, device.architecture, device.driver);
    let mut props = extra;
    if let Some(dir) = &cache {
        std::fs::create_dir_all(dir).map_err(|e| format!("cache directory: {e}"))?;
        props.push(("CACHE_DIR".to_owned(), dir.clone()));
    }
    println!("projection {n} x {k} (n outputs, k inputs), {iters} timed calls per point, cache {}", cache.as_deref().unwrap_or("off"));
    println!("{:<10} {:>6} {:>9} {:>10} {:>9} {:>9} {:>8} {:>9}", "variant", "batch", "compile", "npu ms", "npu GOPS", "cpu ms", "cpu GOPS", "cpu/npu");
    let mut state = 12345u64;
    let mut random = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    };
    for spec in &variants {
        let (quant, policy) = variant(spec)?;
        let raw = synthetic(quant, n, k);
        let weight = convert(quant, &raw, n, k, &policy)?;
        for &batch in &batches {
            let ir = build_linear_ir(&weight, batch, "fllm_bench")?;
            let started = Instant::now();
            let compiled = runtime.compile_ir(&ir.xml, &ir.bin, &device.name, &props)?;
            let compile_ms = started.elapsed().as_secs_f64() * 1e3;
            let x: Vec<f32> = (0..batch * k).map(|_| random()).collect();
            let x16: Vec<u16> = x.iter().map(|v| f16::from_f32(*v).to_bits()).collect();
            let mut y16 = vec![0u16; batch * n];
            compiled.run(&x16, &mut y16)?;
            let mut times = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t = Instant::now();
                compiled.run(&x16, &mut y16)?;
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let npu_ms = median(&mut times);
            let ops = 2.0 * batch as f64 * n as f64 * k as f64;
            let mut cpu_times = Vec::new();
            for _ in 0..iters.min(10) {
                let t = Instant::now();
                if super::engine::host_linear(&raw, quant, n, k, &x, None).is_none() {
                    break;
                }
                cpu_times.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let cpu = (!cpu_times.is_empty()).then(|| median(&mut cpu_times));
            let (cpu_ms, cpu_gops, ratio) = match cpu {
                Some(ms) => (format!("{ms:.2}"), format!("{:.1}", ops / ms / 1e6), format!("{:.2}x", ms / npu_ms)),
                None => ("n/a".to_owned(), "n/a".to_owned(), "n/a".to_owned()),
            };
            println!("{:<10} {:>6} {:>7.0}ms {:>10.2} {:>9.1} {:>9} {:>8} {:>9}", spec, batch, compile_ms, npu_ms, ops / npu_ms / 1e6, cpu_ms, cpu_gops, ratio);
        }
    }
    Ok(())
}
