//! `fllm-rs selftest-parallel`: the multi-device code on the CPU. Every rank is a thread with its own sharded decoder and the ranks
//! talk through `LocalComm`; the logits of a step and the greedy generation must equal those of the ordinary single-device decoder.
//! The same code paths run on GPUs (only the communicator and the device kernels differ).
use super::config::{EpDispatch, ParallelConfig};
use super::ctx::ParCtx;
use super::fixture::{scratch_path, TinySpec};
use super::group::worker_loop;
use crate::comm::local::{run_ranks, LocalComm};
use crate::model::GgufModel;
use crate::rust_decoder::RustDecoder;
use std::sync::Arc;
use std::time::Instant;

pub const HELP: &str = "usage: fllm-rs selftest-parallel [--tp N] [--pp N] [--ep] [--ep-dispatch allreduce|alltoall] [--vocab-parallel-embedding]\n\
       [--case NAME | --model PATH] [--steps N] [--tolerance X] [--all] [--list]\n\
Runs the tensor / pipeline / expert parallel code on the CPU, one thread per rank, and checks that the logits and the greedy\n\
generation equal the single-device run. Without --model it uses tiny random models in many weight formats (--list shows them).\n\
--all runs the whole matrix of cases and layouts.";

/// Relative tolerance of the logits (largest difference over largest logit).
pub const TOLERANCE: f64 = 1e-4;

#[derive(Clone, Debug)]
pub struct RunSpec {
    pub tp: usize,
    pub pp: usize,
    pub ep: bool,
    pub dispatch: EpDispatch,
    pub vocab_parallel_embedding: bool,
    pub steps: usize,
    /// Layers of every pipeline stage (empty: an even split), as the layer-split planner would choose them.
    pub ranges: Vec<std::ops::Range<usize>>,
    /// Ranks that run their stage through the device path (the CPU backend in the role of a GPU: `init_backend`) instead of the host
    /// path; empty: all ranks use the host path. Proves the staged device forward pass without GPU hardware.
    pub accel: Vec<bool>,
}

impl RunSpec {
    pub fn new(tp: usize, pp: usize) -> Self {
        Self { tp, pp, ep: false, dispatch: EpDispatch::AllReduce, vocab_parallel_embedding: false, steps: 16, ranges: Vec::new(), accel: Vec::new() }
    }

    pub fn label(&self) -> String {
        let mut s = format!("tp={} pp={}", self.tp, self.pp);
        if self.ep {
            s += &format!(" ep={}", if self.dispatch == EpDispatch::AllToAll { "alltoall" } else { "allreduce" });
        }
        if self.vocab_parallel_embedding {
            s += " vocab-embed";
        }
        if !self.ranges.is_empty() {
            s += &format!(" layers {}", self.ranges.iter().map(|r| format!("{}..{}", r.start, r.end)).collect::<Vec<_>>().join("|"));
        }
        if !self.accel.is_empty() {
            s += &format!(" device-stages {}", self.accel.iter().map(|&a| if a { 'D' } else { 'H' }).collect::<String>());
        }
        s
    }

    fn config(&self) -> ParallelConfig {
        ParallelConfig {
            tp: self.tp,
            pp: self.pp,
            expert_parallel: self.ep,
            ep_dispatch: self.dispatch,
            vocab_parallel_embedding: self.vocab_parallel_embedding,
            stage_ranges: self.ranges.clone(),
            ..Default::default()
        }
    }
}

/// The logits of every step and the greedy tokens of the two sequences.
pub struct Trace {
    pub logits: Vec<Vec<f32>>,
    pub tokens: Vec<Vec<i64>>,
}

fn argmax(row: &[f32]) -> i64 {
    let mut best = 0;
    for (i, v) in row.iter().enumerate() {
        if *v > row[best] {
            best = i;
        }
    }
    best as i64
}

/// Prompt of 12 and 7 tokens (the first is prefilled in two chunks), then `steps` greedy decode steps of both sequences.
pub fn drive(decoder: &RustDecoder, model: &GgufModel, key: &mut [u16], value: &mut [u16], steps: usize) -> Result<Trace, String> {
    drive_forced(decoder, model, key, value, steps, None)
}

/// `drive`, but when `forced` is given the sequences continue with the tokens of that trace instead of the run's own greedy choice
/// (teacher forcing). A tiny random model has near-ties between its best tokens: with free generation a rounding difference that flips
/// one choice makes the two runs continue with different texts and their logits stop being comparable. The run still records its own
/// choice in `tokens`, so a differing choice stays visible.
pub fn drive_forced(
    decoder: &RustDecoder,
    model: &GgufModel,
    key: &mut [u16],
    value: &mut [u16],
    steps: usize,
    forced: Option<&Trace>,
) -> Result<Trace, String> {
    let pick = |s: usize, n: usize, own: i64| -> i64 { forced.and_then(|f| f.tokens.get(s)).and_then(|t| t.get(n)).copied().unwrap_or(own) };
    let vocab = model.vocab_size();
    let stride = model.table_stride();
    let mut rng = crate::cpu_selftest::Lcg(0xfeed);
    let mut prompt = |n: usize| -> Vec<i64> { (0..n).map(|_| (rng.next() as usize % vocab) as i64).collect() };
    let mut seqs: Vec<Vec<i64>> = vec![prompt(12), prompt(7)];
    let blocks = stride;
    let table = |s: usize| -> Vec<i64> { (0..stride).map(|b| (s * blocks + b) as i64).collect() };
    let mut logits_trace = Vec::new();
    let mut tokens: Vec<Vec<i64>> = vec![Vec::new(), Vec::new()];

    // step 0: chunk 1 of sequence 0 (no logits needed) together with all of sequence 1
    let mut ids = seqs[0][..8].to_vec();
    ids.extend_from_slice(&seqs[1]);
    let mut lengths: Vec<i64> = (1..=8).collect();
    lengths.extend(1..=7);
    let mut tables = Vec::new();
    for _ in 0..8 {
        tables.extend(table(0));
    }
    for _ in 0..7 {
        tables.extend(table(1));
    }
    let out = decoder.forward(&ids, &lengths, &tables, key, value, &[14], false)?;
    logits_trace.push(out.clone());
    tokens[1].push(argmax(&out[..vocab]));
    // step 1: chunk 2 of sequence 0
    let ids = seqs[0][8..].to_vec();
    let lengths: Vec<i64> = (9..=12).collect();
    let mut tables = Vec::new();
    for _ in 0..4 {
        tables.extend(table(0));
    }
    let out = decoder.forward(&ids, &lengths, &tables, key, value, &[3], false)?;
    logits_trace.push(out.clone());
    tokens[0].push(argmax(&out[..vocab]));
    seqs[0].push(pick(0, 0, tokens[0][0]));
    seqs[1].push(pick(1, 0, tokens[1][0]));
    for _ in 1..steps {
        let ids = vec![*seqs[0].last().unwrap_or(&0), *seqs[1].last().unwrap_or(&0)];
        let lengths = vec![seqs[0].len() as i64, seqs[1].len() as i64];
        let mut tables = table(0);
        tables.extend(table(1));
        let out = decoder.forward(&ids, &lengths, &tables, key, value, &[0, 1], false)?;
        logits_trace.push(out.clone());
        for s in 0..2 {
            let next = argmax(&out[s * vocab..(s + 1) * vocab]);
            let n = tokens[s].len();
            tokens[s].push(next);
            seqs[s].push(pick(s, n, next));
        }
    }
    Ok(Trace { logits: logits_trace, tokens })
}

pub fn run_single(model: &GgufModel, steps: usize) -> Result<Trace, String> {
    let decoder = RustDecoder::load(model, "cpu")?;
    let mut key = vec![0u16; model.key_buffer_len()?];
    let mut value = vec![0u16; model.value_buffer_len()?];
    drive(&decoder, model, &mut key, &mut value, steps)
}

pub fn run_parallel(model: &GgufModel, spec: &RunSpec) -> Result<Trace, String> {
    run_parallel_forced(model, spec, None)
}

/// `run_parallel` with the tokens of `forced` fed to both sequences (see `drive_forced`).
pub fn run_parallel_forced(model: &GgufModel, spec: &RunSpec, forced: Option<&Trace>) -> Result<Trace, String> {
    let cfg = Arc::new(spec.config());
    cfg.validate()?;
    super::plan::check_model(model.mapping(), model.config(), &cfg, false)?;
    let world = LocalComm::world(cfg.world_size());
    let results = run_ranks(&world, |rank, comm| {
        let model = model.duplicate()?;
        let ctx = Arc::new(ParCtx::new_host(cfg.clone(), rank, comm)?);
        let mut decoder = RustDecoder::load_sharded(&model, "cpu", ctx.clone())?;
        if spec.accel.get(rank).copied().unwrap_or(false) {
            decoder.init_backend(Arc::new(crate::backend::cpu::CpuBackend::new()))?;
            if !decoder.uses_accelerator() {
                return Err(format!("rank {rank}: the device path was not taken"));
            }
        }
        if rank == 0 {
            let mut key = vec![0u16; model.key_buffer_len()?];
            let mut value = vec![0u16; model.value_buffer_len()?];
            drive_forced(&decoder, &model, &mut key, &mut value, spec.steps, forced).map(Some)
        } else {
            worker_loop(&decoder, &ctx);
            Ok(None)
        }
    })?;
    results.into_iter().next().flatten().ok_or_else(|| "rank 0 returned no trace".to_owned())
}

/// Relative L2 error of one step: `|other - reference|_2 / |reference|_2`.
pub fn relative_error(reference: &[f32], other: &[f32]) -> f64 {
    if reference.len() != other.len() {
        return f64::INFINITY;
    }
    let norm: f64 = reference.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt().max(1e-30);
    let diff: f64 = reference.iter().zip(other).map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2)).sum::<f64>().sqrt();
    if diff.is_nan() {
        f64::INFINITY
    } else {
        diff / norm
    }
}

/// Tolerance of a recipe: 1e-4 when every weight is float (only the order of the sums differs between layouts); the quantized formats
/// add the noise of the CPU kernels' 8-bit activation quantization, whose per-vector scale differs when a vector is cut in slices.
pub fn tolerance_for(case: &str) -> f64 {
    match case {
        "llama-f16" | "gemma2-f32" => FLOAT_TOLERANCE,
        // the router of the tiny random model has near-ties: a last-bit difference between layouts can send a token to another expert
        "moe-shared-f32" => ROUTING_TOLERANCE,
        _ => QUANTIZED_TOLERANCE,
    }
}

/// A sum that rounds differently (another kernel, another slice) can tip the next layer's 8-bit activation rounding, which moves single
/// logits by a few percent; a real layout defect shows up as an error of order 1.
pub const QUANTIZED_TOLERANCE: f64 = 1e-1;
/// Expert routing with near-tied scores (measured up to 1.7e-1 on AVX-512 hosts); a broken layout is off by order 1.
pub const ROUTING_TOLERANCE: f64 = 3e-1;
/// Stages that run the device forward pass: the device kernels round differently from the host kernels (f16 weights: 1.1e-3 measured).
pub const DEVICE_TOLERANCE: f64 = 5e-3;
/// Float weights: the sums are added in a different order (and the f16 kernels round differently per slice).
pub const FLOAT_TOLERANCE: f64 = 1e-3;

#[derive(Debug)]
pub struct CaseResult {
    pub case: String,
    pub layout: String,
    pub max_relative_error: f64,
    pub exact_steps: usize,
    pub steps: usize,
    pub tokens_equal: bool,
    pub millis: f64,
}

impl CaseResult {
    pub fn ok(&self, tolerance: f64) -> bool {
        self.tokens_equal && self.max_relative_error <= tolerance
    }
}

/// The greedy tokens agree, or every differing choice was a near-tie of the reference: its logits for the two tokens are closer than
/// the tolerance of the recipe (relative to the norm of the logit row). Only meaningful when both runs saw the same context, i.e. when
/// `other` was driven with the tokens of `reference`.
fn tokens_match_or_tie(reference: &Trace, other: &Trace, tolerance: f64) -> bool {
    if reference.tokens == other.tokens {
        return true;
    }
    let vocab = reference.logits.first().map_or(0, Vec::len);
    if vocab == 0 || reference.tokens.len() != other.tokens.len() {
        return false;
    }
    // step 0 produced token 0 of sequence 1 and step 1 token 0 of sequence 0; every later step one token of each sequence
    let mut slots: Vec<(usize, usize, &[f32])> = Vec::new();
    for (step, logits) in reference.logits.iter().enumerate() {
        match step {
            0 => slots.push((1, 0, &logits[..vocab])),
            1 => slots.push((0, 0, &logits[..vocab])),
            _ => {
                for s in 0..2 {
                    if let Some(row) = logits.get(s * vocab..(s + 1) * vocab) {
                        slots.push((s, step - 1, row));
                    }
                }
            }
        }
    }
    slots.into_iter().all(|(s, n, row)| {
        let (a, b) = (reference.tokens[s].get(n), other.tokens[s].get(n));
        match (a, b) {
            (Some(a), Some(b)) if a == b => true,
            (Some(&a), Some(&b)) => {
                let norm = row.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt().max(1e-30);
                (f64::from(row[a as usize]) - f64::from(row[b as usize])).abs() <= tolerance * norm
            }
            _ => false,
        }
    })
}

pub fn compare(case: &str, spec: &RunSpec, reference: &Trace, other: &Trace, millis: f64) -> CaseResult {
    let mut worst = 0.0f64;
    let mut exact = 0;
    for (a, b) in reference.logits.iter().zip(&other.logits) {
        worst = worst.max(relative_error(a, b));
        if a == b {
            exact += 1;
        }
    }
    if reference.logits.len() != other.logits.len() {
        worst = f64::INFINITY;
    }
    CaseResult {
        case: case.to_owned(),
        layout: spec.label(),
        max_relative_error: worst,
        exact_steps: exact,
        steps: reference.logits.len(),
        tokens_equal: tokens_match_or_tie(reference, other, tolerance_for(case)),
        millis,
    }
}

/// Run one layout on a model file and compare it with the single-device run (`reference` is reused across layouts).
pub fn check_layout(case: &str, model: &GgufModel, reference: &Trace, spec: &RunSpec) -> Result<CaseResult, String> {
    let started = Instant::now();
    let trace = run_parallel_forced(model, spec, Some(reference))?;
    Ok(compare(case, spec, reference, &trace, started.elapsed().as_secs_f64() * 1e3))
}

/// The layouts every recipe is run with.
pub fn layouts_for(case: &str) -> Vec<RunSpec> {
    let mut specs = vec![RunSpec::new(2, 1), RunSpec::new(4, 1), RunSpec::new(1, 2), RunSpec::new(2, 2)];
    let moe = case.starts_with("moe");
    match case {
        "llama-q4k" => {
            specs.extend([RunSpec::new(8, 1), RunSpec::new(1, 4), RunSpec { vocab_parallel_embedding: true, ..RunSpec::new(4, 1) }]);
        }
        "qwen3-q8" | "gemma2-f32" | "llama-nvfp4" | "llama-q4zp" => specs.push(RunSpec::new(8, 1)),
        "llama-f16" => specs.push(RunSpec::new(1, 4)),
        _ => {}
    }
    if case == "gemma2-f32" {
        specs.push(RunSpec::new(1, 4));
    }
    if case == "qwen2-q4z" {
        specs.push(RunSpec::new(1, 4));
    }
    if moe {
        specs.retain(|s| s.tp <= 4);
        for dispatch in [EpDispatch::AllReduce, EpDispatch::AllToAll] {
            specs.push(RunSpec { ep: true, dispatch, ..RunSpec::new(4, 1) });
        }
        specs.push(RunSpec { ep: true, dispatch: EpDispatch::AllToAll, ..RunSpec::new(2, 2) });
        specs.push(RunSpec { ep: true, dispatch: EpDispatch::AllReduce, ..RunSpec::new(8, 1) });
    }
    specs
}

fn print_result(r: &CaseResult, tolerance: f64) {
    println!(
        "{:<5} {:<16} {:<28} rel err (L2) {:.2e}  exact {}/{} steps  tokens {}  {:.0} ms",
        if r.ok(tolerance) { "PASS" } else { "FAIL" },
        r.case,
        r.layout,
        r.max_relative_error,
        r.exact_steps,
        r.steps,
        if r.tokens_equal { "identical" } else { "DIFFER" },
        r.millis
    );
}

pub fn run(args: &[String]) -> Result<(), String> {
    let mut spec = RunSpec::new(2, 1);
    let mut case: Option<String> = None;
    let mut model_path: Option<String> = None;
    let mut all = false;
    let mut tolerance: Option<f64> = None;
    let mut i = 0;
    let value = |i: &mut usize, flag: &str| -> Result<String, String> {
        *i += 1;
        args.get(*i).cloned().ok_or_else(|| format!("{flag} needs a value"))
    };
    while i < args.len() {
        match args[i].as_str() {
            "--tp" | "--tensor-parallel-size" => spec.tp = value(&mut i, "--tp")?.parse().map_err(|_| "invalid --tp")?,
            "--pp" | "--pipeline-parallel-size" => spec.pp = value(&mut i, "--pp")?.parse().map_err(|_| "invalid --pp")?,
            "--ep" | "--expert-parallel" => spec.ep = true,
            "--ep-dispatch" => {
                spec.dispatch = match value(&mut i, "--ep-dispatch")?.as_str() {
                    "allreduce" => EpDispatch::AllReduce,
                    "alltoall" => EpDispatch::AllToAll,
                    other => return Err(format!("unknown --ep-dispatch {other}")),
                }
            }
            "--vocab-parallel-embedding" => spec.vocab_parallel_embedding = true,
            "--case" => case = Some(value(&mut i, "--case")?),
            "--model" => model_path = Some(value(&mut i, "--model")?),
            "--steps" => spec.steps = value(&mut i, "--steps")?.parse().map_err(|_| "invalid --steps")?,
            "--tolerance" => tolerance = Some(value(&mut i, "--tolerance")?.parse().map_err(|_| "invalid --tolerance")?),
            "--all" => all = true,
            "--list" => {
                for name in TinySpec::CASES {
                    println!("{name}");
                }
                return Ok(());
            }
            "--help" | "-h" => {
                println!("{HELP}");
                return Ok(());
            }
            other => return Err(format!("unknown option {other}\n{HELP}")),
        }
        i += 1;
    }
    let mut failures = 0;
    let mut total = 0;
    if let Some(path) = model_path {
        let model = GgufModel::load(&path, 64, 2, 1)?;
        let reference = run_single(&model, spec.steps)?;
        let result = check_layout(&path, &model, &reference, &spec)?;
        let limit = tolerance.unwrap_or(QUANTIZED_TOLERANCE);
        print_result(&result, limit);
        total += 1;
        failures += usize::from(!result.ok(limit));
    } else {
        let names: Vec<&str> = match &case {
            Some(c) => vec![TinySpec::CASES.iter().find(|n| *n == c).copied().ok_or_else(|| format!("unknown case {c} (--list shows them)"))?],
            None if all => TinySpec::CASES.to_vec(),
            None => vec!["llama-q4k"],
        };
        for name in names {
            let recipe = TinySpec::named(name).ok_or("unknown recipe")?;
            let path = scratch_path(name);
            recipe.write(&path)?;
            let outcome = (|| -> Result<(), String> {
                let model = GgufModel::load(path.to_str().ok_or("path")?, 64, 2, 1)?;
                let reference = run_single(&model, spec.steps)?;
                let layouts = if all { layouts_for(name) } else { vec![spec.clone()] };
                for layout in layouts {
                    let layout = RunSpec { steps: spec.steps, ..layout };
                    total += 1;
                    match check_layout(name, &model, &reference, &layout) {
                        Ok(result) => {
                            let limit = tolerance.unwrap_or_else(|| tolerance_for(name));
                            print_result(&result, limit);
                            failures += usize::from(!result.ok(limit));
                        }
                        Err(e) => {
                            println!("FAIL  {name:<16} {:<28} {e}", layout.label());
                            failures += 1;
                        }
                    }
                }
                Ok(())
            })();
            let _ = std::fs::remove_file(&path);
            outcome?;
        }
    }
    println!("{} of {total} parallel checks passed", total - failures);
    if failures > 0 {
        Err(format!("{failures} parallel checks failed"))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_case(name: &str, layouts: &[RunSpec]) {
        let recipe = TinySpec::named(name).unwrap();
        let path = scratch_path(name);
        recipe.write(&path).unwrap();
        let model = GgufModel::load(path.to_str().unwrap(), 64, 2, 1).unwrap();
        let _ = std::fs::remove_file(&path);
        let reference = run_single(&model, 16).unwrap();
        assert_eq!(reference.tokens[0].len(), 16);
        let mut report = Vec::new();
        for layout in layouts {
            let result = check_layout(name, &model, &reference, layout).unwrap_or_else(|e| panic!("{name} {}: {e}", layout.label()));
            // a device stage computes with the device kernels (here the CPU backend's): a rounding of its own on top of the layout's
            let limit = if layout.accel.contains(&true) { tolerance_for(name).max(DEVICE_TOLERANCE) } else { tolerance_for(name) };
            print_result(&result, limit);
            if !result.ok(limit) {
                report.push(format!("{name} {}: error {:.2e}, tokens equal {}", layout.label(), result.max_relative_error, result.tokens_equal));
            }
        }
        assert!(report.is_empty(), "{}", report.join("\n"));
    }

    fn spec_with(pp: usize, ranges: &[std::ops::Range<usize>], accel: &[bool]) -> RunSpec {
        RunSpec { ranges: ranges.to_vec(), accel: accel.to_vec(), ..RunSpec::new(1, pp) }
    }

    /// Layer-split over devices: every stage runs the device forward pass (here on the CPU backend), the activations travel through
    /// host memory, and the result equals the single-device host run. Even and uneven splits, device stages next to host stages.
    #[test]
    fn device_stages_match_the_single_device_run() {
        run_case(
            "llama-q4k",
            &[
                spec_with(2, &[], &[true, true]),
                spec_with(2, &[0..1, 1..4], &[true, true]),
                spec_with(2, &[0..3, 3..4], &[true, false]),
                spec_with(2, &[0..2, 2..4], &[false, true]),
                spec_with(3, &[0..1, 1..3, 3..4], &[true, false, true]),
                spec_with(4, &[], &[true, true, true, true]),
            ],
        );
        run_case("qwen3-q8", &[spec_with(2, &[0..1, 1..4], &[true, true]), spec_with(3, &[0..2, 2..3, 3..4], &[true, true, false])]);
        run_case("llama-f16", &[spec_with(2, &[], &[true, true]), spec_with(4, &[0..1, 1..2, 2..3, 3..4], &[true, false, true, false])]);
        run_case("gemma2-f32", &[spec_with(2, &[], &[true, true]), spec_with(2, &[0..1, 1..4], &[true, false])]);
        run_case("qwen2-q4z", &[spec_with(2, &[0..3, 3..4], &[true, true])]);
        run_case("llama-nvfp4", &[spec_with(2, &[], &[true, true])]);
    }

    /// Mixtures of experts split by layers with uneven ranges run on the host stages (the CPU backend has no expert kernels, so a device
    /// stage of an MoE model can only be tried on a real GPU backend).
    #[test]
    fn uneven_layer_ranges_match_for_mixture_of_experts() {
        run_case("moe-q4k", &[spec_with(2, &[0..1, 1..4], &[]), spec_with(3, &[0..2, 2..3, 3..4], &[])]);
        run_case("moe-shared-f32", &[spec_with(2, &[0..3, 3..4], &[])]);
    }

    #[test]
    fn llama_q4k_matches_for_tp_pp_and_both() {
        run_case("llama-q4k", &layouts_for("llama-q4k"));
    }

    #[test]
    fn qwen3_q8_matches() {
        run_case("qwen3-q8", &layouts_for("qwen3-q8"));
    }

    #[test]
    fn qwen2_q4z_with_bias_matches() {
        run_case("qwen2-q4z", &layouts_for("qwen2-q4z"));
    }

    #[test]
    fn gemma2_with_window_softcap_and_tied_head_matches() {
        run_case("gemma2-f32", &layouts_for("gemma2-f32"));
    }

    #[test]
    fn nvfp4_matches() {
        run_case("llama-nvfp4", &layouts_for("llama-nvfp4"));
    }

    #[test]
    fn gptq_with_activation_order_matches() {
        run_case("llama-q4zp", &layouts_for("llama-q4zp"));
    }

    #[test]
    fn f16_weights_match() {
        run_case("llama-f16", &layouts_for("llama-f16"));
    }

    #[test]
    fn moe_with_tensor_and_expert_parallelism_matches() {
        run_case("moe-q4k", &layouts_for("moe-q4k"));
    }

    #[test]
    fn moe_with_shared_expert_matches() {
        run_case("moe-shared-f32", &layouts_for("moe-shared-f32"));
    }

    /// Diagnostic: which ingredient of the qwen2-q4z recipe breaks tensor parallelism of 4 (prints, does not assert).
    #[test]
    fn qwen2_q4z_variants_localise_the_tp4_failure() {
        let base = TinySpec::named("qwen2-q4z").unwrap();
        let variants: Vec<(&str, TinySpec)> = vec![
            ("no bias", TinySpec { qkv_bias: false, ..base.clone() }),
            ("f32 head", TinySpec { formats: super::super::fixture::Formats { head: super::super::fixture::F32, ..base.formats }, ..base.clone() }),
            ("q4z128 -> q4z32", TinySpec { formats: super::super::fixture::Formats { attn: super::super::fixture::Q4Z32, ffn: super::super::fixture::Q4Z32, expert: super::super::fixture::Q4Z32, ..base.formats }, ..base.clone() }),
        ];
        for (label, spec) in variants {
            let path = scratch_path("variant");
            spec.write(&path).unwrap();
            let model = GgufModel::load(path.to_str().unwrap(), 64, 2, 1).unwrap();
            let _ = std::fs::remove_file(&path);
            let reference = run_single(&model, 16).unwrap();
            let result = check_layout(label, &model, &reference, &RunSpec::new(4, 1)).unwrap();
            print_result(&result, QUANTIZED_TOLERANCE);
        }
    }

    #[test]
    fn impossible_layouts_are_refused_with_the_tensor_name_and_the_sizes_that_work() {
        let recipe = TinySpec::named("moe-q4k").unwrap();
        let path = scratch_path("refuse");
        recipe.write(&path).unwrap();
        let model = GgufModel::load(path.to_str().unwrap(), 64, 2, 1).unwrap();
        let _ = std::fs::remove_file(&path);
        // 8 ranks cut the 1024-wide experts into 128 values, below the 256-value Q4_K block
        let error = run_parallel(&model, &RunSpec::new(8, 1)).err().expect("must fail");
        assert!(error.contains("ffn_down_exps") && error.contains("1, 2, 4"), "{error}");
        // 16 heads do not divide by 3
        let error = run_parallel(&model, &RunSpec::new(3, 1)).err().expect("must fail");
        assert!(error.contains("16 attention heads") && error.contains("sizes that work"), "{error}");
        // expert parallelism needs a tensor group
        let spec = RunSpec { ep: true, ..RunSpec::new(1, 2) };
        assert!(run_parallel(&model, &spec).is_err());
        // more stages than layers
        assert!(run_parallel(&model, &RunSpec::new(1, 8)).err().expect("must fail").contains("layers"));
    }

    #[test]
    fn the_production_launcher_runs_the_same_group_with_worker_threads() {
        let recipe = TinySpec::named("llama-q4k").unwrap();
        let path = scratch_path("launch");
        recipe.write(&path).unwrap();
        let model = GgufModel::load(path.to_str().unwrap(), 64, 2, 1).unwrap();
        let _ = std::fs::remove_file(&path);
        let reference = run_single(&model, 16).unwrap();
        let cfg = ParallelConfig { tp: 2, pp: 2, ..Default::default() };
        let decoder = super::super::group::launch(&model, "cpu", cfg).unwrap();
        let mut key = vec![0u16; model.key_buffer_len().unwrap()];
        let mut value = vec![0u16; model.value_buffer_len().unwrap()];
        let trace = drive(&decoder, &model, &mut key, &mut value, 16).unwrap();
        drop(decoder);
        let result = compare("launch", &RunSpec::new(2, 2), &reference, &trace, 0.0);
        assert!(result.ok(tolerance_for("llama-q4k")), "{result:?}");
    }

    #[test]
    fn the_distributed_argmax_equals_the_argmax_of_the_full_logits() {
        // uses the sharded head of a 4-rank group: compare ids from vocabulary slices with the full-vocabulary argmax
        let recipe = TinySpec::named("llama-q4k").unwrap();
        let path = scratch_path("argmax");
        recipe.write(&path).unwrap();
        let model = GgufModel::load(path.to_str().unwrap(), 64, 2, 1).unwrap();
        let _ = std::fs::remove_file(&path);
        let cfg = Arc::new(ParallelConfig { tp: 4, ..Default::default() });
        let world = LocalComm::world(4);
        let vocab = model.vocab_size();
        let results = run_ranks(&world, |rank, comm| {
            let model = model.duplicate()?;
            let ctx = Arc::new(ParCtx::new_host(cfg.clone(), rank, comm)?);
            let decoder = RustDecoder::load_sharded(&model, "cpu", ctx)?;
            // fake local logits: a known pattern over the global vocabulary, restricted to this rank's slice
            let mine = crate::parallel::plan::vocab_range(vocab, 4, 1, rank);
            let rows = 3;
            let mut local = Vec::new();
            let mut full_ids = Vec::new();
            for row in 0..rows {
                let full: Vec<f32> = (0..vocab).map(|v| (((v * 37 + row * 101) % 211) as f32) * 0.5).collect();
                full_ids.push(argmax(&full) as i32);
                local.extend_from_slice(&full[mine.clone()]);
            }
            Ok((decoder.par_argmax(&local, rows)?, full_ids))
        })
        .unwrap();
        for (ids, expected) in results {
            assert_eq!(ids, expected);
        }
    }
}
