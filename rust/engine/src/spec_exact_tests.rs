//! Is a verification pass bit-identical to decoding one token at a time?
//!
//! Speculative decoding checks several draft tokens in one forward pass with several rows. If a projection kernel adds the terms of a
//! row in another order for a batch of rows than for one row, the logits of the batched pass differ in the last bits from the ones
//! plain decoding computes, and a near-tie between the two best tokens can then be decided the other way.
//! This module measures that on the projection kernels.

use crate::cpu_selftest::{random_weights, Lcg};
use half::{bf16, f16};

/// Token `t` of a batched projection against the same token alone: the number of tokens whose outputs differ in any bit.
fn differing_tokens(run: &dyn Fn(&[f32]) -> Option<Vec<f32>>, rows: usize, cols: usize, batch: usize, rng: &mut Lcg) -> Option<usize> {
    let x = rng.vector(batch * cols, 3.0);
    let all = run(&x)?;
    let mut bad = 0;
    for t in 0..batch {
        let one = run(&x[t * cols..(t + 1) * cols])?;
        if all[t * rows..(t + 1) * rows].iter().zip(&one).any(|(a, b)| a.to_bits() != b.to_bits()) {
            bad += 1;
        }
    }
    Some(bad)
}

fn float_bytes(kind: u32, count: usize, rng: &mut Lcg) -> Vec<u8> {
    let values = rng.vector(count, 0.3);
    match kind {
        0 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        1 => values.iter().flat_map(|v| f16::from_f32(*v).to_le_bytes()).collect(),
        _ => values.iter().flat_map(|v| bf16::from_f32(*v).to_le_bytes()).collect(),
    }
}

/// Without the guard: prints which kernels differ (a measurement, nothing asserted). With it: every kernel must give each token its
/// single-token bits.
#[test]
fn projection_kernels_give_a_token_the_same_bits_alone_and_in_a_batch() {
    println!("default kernels:");
    probe(false);
    println!("batch-invariant kernels:");
    probe(true);
}

fn probe(invariant: bool) {
    let _guard = invariant.then(crate::cpu_kernels::BatchInvariant::enter);
    let mut rng = Lcg(0x5eed);
    let mut report = Vec::new();
    let kinds: [u32; 22] = [0, 1, 30, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 21, 22, 20, 23, 39, 42];
    for kind in kinds {
        for (rows, cols) in [(13usize, 512usize), (64, 2048), (257, 768)] {
            let weights = if matches!(kind, 0 | 1 | 30) { float_bytes(kind, rows * cols, &mut rng) } else { random_weights(kind, rows, cols, &mut rng) };
            let run = |x: &[f32]| -> Option<Vec<f32>> {
                if matches!(kind, 0 | 1 | 30) {
                    crate::cpu_float::linear(&weights, kind, x, rows, cols, None)
                } else if kind == 8 {
                    crate::cpu_kernels::q8_linear(&weights, x, rows, cols, None)
                } else {
                    crate::cpu_quantized::linear(&weights, kind, x, rows, cols, None)
                }
            };
            for batch in [2usize, 3, 4, 6, 7, 8, 16] {
                match differing_tokens(&run, rows, cols, batch, &mut rng) {
                    Some(0) => {}
                    Some(n) => report.push(format!("kind {kind} rows {rows} cols {cols} batch {batch}: {n} of {batch} tokens differ")),
                    None => {
                        report.push(format!("kind {kind}: no kernel on this CPU"));
                        break;
                    }
                }
            }
        }
    }
    println!("isa {} avx512 feature {:?}", crate::cpu_kernels::isa(), cfg!(feature = "avx512"));
    // one line per (kind, batch): the shapes that differ
    let mut summary = std::collections::BTreeMap::<String, usize>::new();
    for line in &report {
        let key = match line.find(" rows ") {
            Some(at) => format!("{} batch {}", &line[..at], line.split(" batch ").nth(1).and_then(|s| s.split(':').next()).unwrap_or("?")),
            None => line.clone(),
        };
        *summary.entry(key).or_default() += 1;
    }
    for (key, shapes) in &summary {
        println!("DIFF {key}: {shapes} of 3 shapes");
    }
    if invariant {
        assert!(report.iter().all(|l| l.contains("no kernel")), "batched projections differ from single-token ones:\n{}", report.join("\n"));
    }
}

/// Large enough that the exact pass takes the row-block path (`cpu_quantized::exact_by_row_blocks`, from 8 MB of weights): every token
/// must still get the bits it gets alone, for each format that path serves.
#[test]
fn the_row_block_path_of_the_exact_pass_keeps_the_bits() {
    let mut rng = Lcg(0xb10c);
    let cols = 2048usize;
    for kind in [12u32, 14, 11, 10, 2, 3, 6, 7, 20, 23] {
        // every format above is at least 8 MB at this size (the block path starts at 8 MB); not a multiple of the block height, so the
        // last block is short
        let rows = 13000usize;
        let weights = random_weights(kind, rows, cols, &mut rng);
        let bias = rng.vector(rows, 1.0);
        let run = |x: &[f32]| -> Option<Vec<f32>> { crate::cpu_quantized::linear(&weights, kind, x, rows, cols, Some(&bias)) };
        let _guard = crate::cpu_kernels::BatchInvariant::enter();
        for batch in [2usize, 7] {
            let x = rng.vector(batch * cols, 3.0);
            let Some(all) = run(&x) else { break };
            for t in 0..batch {
                let one = run(&x[t * cols..(t + 1) * cols]).unwrap();
                assert!(
                    all[t * rows..(t + 1) * rows].iter().zip(&one).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "kind {kind} batch {batch} token {t} differs from the same token alone"
                );
            }
        }
    }
}

/// The guard is per thread. A pass hands it to the pool workers that run its projections, and it must not reach a neighbour: another
/// thread that holds no guard keeps the default kernels while this one runs an exact pass, and the other way round. The test asks
/// `batch_invariant()` itself, so it does not depend on whether a kernel gives other bits on the host it runs on.
#[test]
fn the_exact_choice_belongs_to_the_pass_that_made_it() {
    use crate::cpu_kernels::{batch_invariant, BatchInvariant};
    if std::env::var_os("FLLM_BATCH_INVARIANT").is_some_and(|v| !v.is_empty() && v != "0") {
        return; // the environment forces the exact kernels for every call: there is no choice to isolate
    }
    assert!(!batch_invariant(), "no guard is alive on this thread at the start");
    // 1. a guard held by another thread does not reach this thread
    let (entered, entered_rx) = std::sync::mpsc::channel();
    let (release, release_rx) = std::sync::mpsc::channel::<()>();
    let holder = std::thread::spawn(move || {
        let _guard = BatchInvariant::enter();
        let own = batch_invariant();
        entered.send(()).unwrap();
        release_rx.recv().unwrap();
        own
    });
    entered_rx.recv().unwrap();
    let seen_here = batch_invariant();
    release.send(()).unwrap();
    assert!(holder.join().unwrap(), "the thread that holds the guard sees it");
    assert!(!seen_here, "a guard held by another thread must not switch this thread to the exact kernels");
    // 2. a guard entered here does not reach a pool worker by itself ...
    let pool = rayon::ThreadPoolBuilder::new().num_threads(3).build().unwrap();
    let _guard = BatchInvariant::enter();
    assert!(batch_invariant(), "this thread holds the guard");
    assert!(!pool.install(batch_invariant), "a pool worker does not see this thread's guard unless the pass hands it over");
    // ... and sees it where the pass hands it over (what `RustDecoder::forward_*` does inside `install`)
    let invariant = batch_invariant();
    let inside = pool.install(|| {
        let _handed = invariant.then(BatchInvariant::enter);
        batch_invariant()
    });
    assert!(inside, "inside a pool that was handed the guard, the workers choose the exact kernels");
    // 3. and the hand-over ends with the pass: afterwards the worker is back to the default
    assert!(!pool.install(batch_invariant), "the worker does not keep the guard after the pass");
}

/// The Q8_0 group projection in its exact pass walks blocks of rows too (`cpu_kernels::q8_linear_group_into`): three siblings of
/// different heights, above the 8 MiB threshold, with a bias on one of them. Every token must get the bits it gets alone.
#[test]
fn the_q8_row_block_path_of_the_exact_pass_keeps_the_bits() {
    let mut rng = Lcg(0x08c0);
    let cols = 2048usize;
    let heights = [9000usize, 4100, 5200];
    let weights: Vec<Vec<u8>> = heights.iter().map(|&rows| random_weights(8, rows, cols, &mut rng)).collect();
    let bias = rng.vector(heights[1], 1.0);
    let projections: Vec<crate::cpu_kernels::Q8Projection<'_>> = heights
        .iter()
        .zip(&weights)
        .enumerate()
        .map(|(i, (&rows, w))| crate::cpu_kernels::Q8Projection { weights: w, rows, bias: (i == 1).then_some(bias.as_slice()) })
        .collect();
    let _guard = crate::cpu_kernels::BatchInvariant::enter();
    for batch in [2usize, 7] {
        let x = rng.vector(batch * cols, 3.0);
        let all = crate::cpu_kernels::q8_linear_group(&projections, &crate::cpu_kernels::Q8Input::new(&x, cols)).expect("a Q8 kernel");
        // the single projection (o_proj, down_proj) has its own loop
        let single = crate::cpu_kernels::q8_linear(&weights[0], &x, heights[0], cols, None).expect("a Q8 kernel");
        for t in 0..batch {
            let one = crate::cpu_kernels::q8_linear(&weights[0], &x[t * cols..(t + 1) * cols], heights[0], cols, None).unwrap();
            assert!(
                single[t * heights[0]..(t + 1) * heights[0]].iter().zip(&one).all(|(a, b)| a.to_bits() == b.to_bits()),
                "single projection batch {batch} token {t} differs from the same token alone"
            );
        }
        for t in 0..batch {
            let one = crate::cpu_kernels::q8_linear_group(&projections, &crate::cpu_kernels::Q8Input::new(&x[t * cols..(t + 1) * cols], cols)).unwrap();
            for (m, rows) in heights.iter().enumerate() {
                assert!(
                    all[m][t * rows..(t + 1) * rows].iter().zip(&one[m]).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "member {m} batch {batch} token {t} differs from the same token alone"
                );
            }
        }
    }
}

/// The float group projection (F32, BF16, F16) in its exact pass walks blocks of rows as well (`cpu_float::exact_by_row_blocks`).
#[test]
fn the_float_row_block_path_of_the_exact_pass_keeps_the_bits() {
    let mut rng = Lcg(0xf10a);
    let cols = 1024usize;
    let heights = [4200usize, 1500, 2100];
    for kind in [0u32, 1, 30] {
        let weights: Vec<Vec<u8>> = heights.iter().map(|&rows| float_bytes(kind, rows * cols, &mut rng)).collect();
        let bias = rng.vector(heights[1], 1.0);
        let projections: Vec<crate::cpu_float::Projection<'_>> = heights
            .iter()
            .zip(&weights)
            .enumerate()
            .map(|(i, (&rows, w))| crate::cpu_float::Projection { weights: w, rows, bias: (i == 1).then_some(bias.as_slice()) })
            .collect();
        let _guard = crate::cpu_kernels::BatchInvariant::enter();
        for batch in [2usize, 7] {
            let x = rng.vector(batch * cols, 3.0);
            let all = crate::cpu_float::linear_group(&projections, kind, &x, cols).expect("a float kernel");
            for t in 0..batch {
                let one = crate::cpu_float::linear_group(&projections, kind, &x[t * cols..(t + 1) * cols], cols).unwrap();
                for (m, rows) in heights.iter().enumerate() {
                    assert!(
                        all[m][t * rows..(t + 1) * rows].iter().zip(&one[m]).all(|(a, b)| a.to_bits() == b.to_bits()),
                        "kind {kind} member {m} batch {batch} token {t} differs from the same token alone"
                    );
                }
            }
        }
    }
}

/// What the batch-invariant variants cost: the time of one projection for one token, and for the 7 rows of a verification pass (six draft
/// tokens) with the default and the invariant kernels. Prints the figures; a verification pass is worth it while its cost stays well
/// below 7 single-token steps.
#[test]
fn cost_of_the_invariant_kernels_in_a_verification_pass() {
    let mut rng = Lcg(77);
    let cols = 4096usize;
    println!("COST isa {} threads {}, projections of 320 MB x {cols} columns (more than the cache of the build host)", crate::cpu_kernels::isa(), rayon::current_num_threads());
    for kind in [12u32, 14, 11, 10, 23, 8, 6, 2, 18, 0, 30] {
        let probe = if matches!(kind, 0 | 1 | 30) { float_bytes(kind, 4 * cols, &mut rng) } else { random_weights(kind, 4, cols, &mut rng) };
        let rows = 320_000_000 / (probe.len() / 4) / 8 * 8;
        let weights = if matches!(kind, 0 | 1 | 30) { float_bytes(kind, rows * cols, &mut rng) } else { random_weights(kind, rows, cols, &mut rng) };
        let run = |x: &[f32]| -> Option<Vec<f32>> {
            if matches!(kind, 0 | 1 | 30) {
                crate::cpu_float::linear(&weights, kind, x, rows, cols, None)
            } else if kind == 8 {
                crate::cpu_kernels::q8_linear(&weights, x, rows, cols, None)
            } else {
                crate::cpu_quantized::linear(&weights, kind, x, rows, cols, None)
            }
        };
        let time = |batch: usize, invariant: bool| -> Option<f64> {
            let _guard = invariant.then(crate::cpu_kernels::BatchInvariant::enter);
            let x = rng_vector(batch * cols, kind);
            run(&x)?;
            // the fastest of 25 runs: the build host is shared with other jobs
            let mut best = f64::MAX;
            for _ in 0..25 {
                let started = std::time::Instant::now();
                run(&x)?;
                best = best.min(started.elapsed().as_secs_f64() * 1000.0);
            }
            Some(best)
        };
        let (Some(one), Some(default), Some(invariant)) = (time(1, false), time(7, false), time(7, true)) else { continue };
        println!("COST kind {kind}: 1 row {one:.2} ms, 7 rows default {default:.2} ms ({:.2}x), 7 rows invariant {invariant:.2} ms ({:.2}x)", default / one, invariant / one);
    }
}

fn rng_vector(len: usize, seed: u32) -> Vec<f32> {
    Lcg(u64::from(seed) + 1).vector(len, 3.0)
}

// ---- whole models: greedy output with scripted wrong drafts against plain greedy output ----

use crate::spec_state_tests::{check_jobs, Job};
use crate::tiny_models::{build_gguf, scratch, Arch, Encoding};

/// `count` jobs with prompts of 3 to 20 tokens (so the token indexer of the attention layer of Qwen3.8-Flash-Next has more blocks than it
/// selects) and 16 to 40 new tokens (the test models have a context of 64), from a fixed generator.
fn many_jobs(count: usize, seed: u64) -> Vec<Job> {
    const IDS: [&str; 12] = ["j0", "j1", "j2", "j3", "j4", "j5", "j6", "j7", "j8", "j9", "j10", "j11"];
    let mut rng = Lcg(seed);
    (0..count)
        .map(|i| {
            let len = 3 + (rng.next() % 18) as usize;
            let mut prompt = vec![1i64];
            prompt.extend((1..len).map(|_| 3 + (rng.next() % 97) as i64));
            Job { id: IDS[i % IDS.len()], prompt, tokens: 16 + (rng.next() % 25) as usize }
        })
        .collect()
}

fn model(label: &str, arch: Arch, encoding: Encoding, seed: u64) -> std::path::PathBuf {
    build_gguf(&scratch(label), arch, encoding, seed)
}

/// The logits of the positions after the prompt when the tokens `forced` are fed `chunk` rows at a time (the rows of a verification pass)
/// instead of one by one. `out[i]` predicts `forced[i + 1]`.
fn chunked_logits(dec: &crate::rust_decoder::RustDecoder, model: &crate::GgufModel, prompt: &[i64], forced: &[i64], chunk: usize, invariant: bool) -> Vec<Vec<f32>> {
    let mut key = vec![0u16; model.key_buffer_len().unwrap()];
    let mut value = vec![0u16; model.value_buffer_len().unwrap()];
    let table: Vec<i64> = (0..model.table_stride() as i64).collect();
    let vocab = model.config().vocab;
    let rows = prompt.len();
    let tables: Vec<i64> = (0..rows).flat_map(|_| table.iter().copied()).collect();
    let lengths: Vec<i64> = (1..=rows as i64).collect();
    dec.forward(prompt, &lengths, &tables, &mut key, &mut value, &[rows - 1], false).unwrap();
    let mut out = Vec::new();
    let mut at = 0;
    while at + 1 < forced.len() {
        let end = (at + chunk).min(forced.len() - 1);
        let ids = &forced[at..end];
        let lengths: Vec<i64> = (0..ids.len()).map(|i| (rows + at + i + 1) as i64).collect();
        let tables: Vec<i64> = (0..ids.len()).flat_map(|_| table.iter().copied()).collect();
        let wanted: Vec<usize> = (0..ids.len()).collect();
        let _guard = invariant.then(crate::cpu_kernels::BatchInvariant::enter);
        let logits = dec.forward(ids, &lengths, &tables, &mut key, &mut value, &wanted, false).unwrap();
        out.extend(logits.chunks(vocab).map(<[f32]>::to_vec));
        at = end;
    }
    out
}

/// How far the logits of a verification pass (6 rows) lie from the ones of single-token decoding, and how often that would have
/// changed the greedy token. Prints one line per model.
fn noise(label: &str, gguf: &std::path::Path, invariant: bool) -> (f32, usize, usize) {
    let model = crate::GgufModel::load(gguf.to_str().unwrap(), 64, 1, 2).unwrap();
    let dec = crate::rust_decoder::RustDecoder::load(&model, "cpu").unwrap();
    let prompt = [1i64, 5, 9, 3, 17, 40, 22, 8, 61, 14, 77, 2, 30];
    let plain = crate::tiny_models::decode(&dec, &model, &prompt, 48, None).unwrap();
    let batched = chunked_logits(&dec, &model, &prompt, &plain.tokens, 6, invariant);
    let (mut worst, mut identical, mut flips, mut positions) = (0f32, 0usize, 0usize, 0usize);
    for (i, got) in batched.iter().enumerate() {
        let want = &plain.logits[i + 1];
        let (d, _) = crate::tiny_models::relative_difference(got, want);
        worst = worst.max(d);
        positions += 1;
        identical += usize::from(got.iter().zip(want).all(|(a, b)| a.to_bits() == b.to_bits()));
        flips += usize::from(crate::tiny_models::argmax(got) != crate::tiny_models::argmax(want));
    }
    let mode = if invariant { "batch-invariant kernels" } else { "default kernels" };
    println!("NOISE {label} ({mode}): {positions} positions, {identical} bit-identical, largest relative logit difference {worst:.2e}, {flips} greedy tokens changed");
    (worst, identical, flips)
}

#[test]
fn noise_of_a_verification_pass_on_tiny_models() {
    for (label, arch, encoding, seed) in [
        ("qwen35 f32", Arch::Qwen35, Encoding::F32, 0x51_0011),
        ("qwen35 q8_0", Arch::Qwen35, Encoding::Quant("q8_0"), 0x51_0012),
        ("qwen35 q4_k_m", Arch::Qwen35, Encoding::Quant("q4_k_m"), 0x51_0013),
        ("qwen35 q2_k", Arch::Qwen35, Encoding::Quant("q2_k"), 0x51_0014),
        ("qwen4 f32", Arch::Qwen4, Encoding::F32, 0x51_0015),
        ("qwen4 q4_k_m", Arch::Qwen4, Encoding::Quant("q4_k_m"), 0x51_0016),
        ("llama f32", Arch::Llama, Encoding::F32, 0x51_0019),
        ("llama q8_0", Arch::Llama, Encoding::Quant("q8_0"), 0x51_001a),
        ("qwen3moe q8_0", Arch::Qwen3Moe, Encoding::Quant("q8_0"), 0x51_001b),
        ("qwen3 q8_0", Arch::Qwen3, Encoding::Quant("q8_0"), 0x51_001c),
        ("llama q4_k_m", Arch::Llama, Encoding::Quant("q4_k_m"), 0x51_0017),
        ("rwkv7 f32", Arch::Rwkv7, Encoding::F32, 0x51_0018),
    ] {
        let gguf = model(&format!("sx-noise-{}", label.replace(' ', "-")), arch, encoding, seed);
        noise(label, &gguf, false);
        let (worst, identical, flips) = noise(label, &gguf, true);
        // the CPU build host runs the AVX2 tier; the invariant variants must give exactly the bits of single-token decoding there
        assert_eq!((identical, flips), (47, 0), "{label}: worst relative logit difference {worst:e}");
    }
}

/// Every job on its own, plain and with scripted drafts that are wrong at every position in turn: the tokens must be the same. This is
/// the guarantee of `docs/speculative-decoding.md`: a request gives the same text with and without speculation.
fn check_alone(label: &str, gguf: &std::path::Path, vocab: i64, jobs: &[Job]) {
    let (changed, rejected) = changed_alone(gguf, vocab, jobs);
    assert_eq!(changed, 0, "{label}: speculative decoding changed the output of {changed} of {} requests", jobs.len());
    assert!(rejected >= 2 * jobs.len(), "{label}: only {rejected} rejections were scripted");
}

/// How many of the `jobs` (each served alone) give other tokens with scripted speculation than without, and how many draft chunks were
/// scripted to be rejected.
fn changed_alone(gguf: &std::path::Path, vocab: i64, jobs: &[Job]) -> (usize, usize) {
    use crate::spec_state_tests::{runtime, serve};
    let (mut changed, mut rejected) = (0, 0);
    for job in jobs {
        let one = std::slice::from_ref(job);
        let plain = serve(&mut runtime(gguf), one, None, vocab).0.remove(0);
        let mut rt = runtime(gguf);
        let (tokens, plan) = serve(&mut rt, one, Some(std::slice::from_ref(&plain)), vocab);
        changed += usize::from(tokens[0] != plain);
        rejected += plan.rejected_at.iter().sum::<usize>();
    }
    (changed, rejected)
}

/// The verification pass with the prompt kernels (what the engine did before) and with the batch-invariant ones, on the same requests.
#[test]
fn verification_with_prompt_kernels_changes_output_and_the_invariant_kernels_do_not() {
    for (label, arch, encoding, seed, jobs) in [
        ("qwen35 q2_k", Arch::Qwen35, Encoding::Quant("q2_k"), 0x51_0005u64, many_jobs(12, 19)),
        ("qwen35 q4_k_m", Arch::Qwen35, Encoding::Quant("q4_k_m"), 0x51_0003, many_jobs(12, 13)),
        ("qwen4 q4_k_m", Arch::Qwen4, Encoding::Quant("q4_k_m"), 0x51_0002, many_jobs(12, 11)),
        ("qwen4 f32", Arch::Qwen4, Encoding::F32, 0x51_0001, many_jobs(12, 7)),
        ("llama q4_k_m", Arch::Llama, Encoding::Quant("q4_k_m"), 0x51_0007, many_jobs(12, 29)),
    ] {
        let gguf = model(&format!("sx-ba-{}", label.replace(' ', "-")), arch, encoding, seed);
        let before = crate::speculative::with_exact_verification(false, || changed_alone(&gguf, 100, &jobs));
        let after = crate::speculative::with_exact_verification(true, || changed_alone(&gguf, 100, &jobs));
        println!("BEFORE/AFTER {label}: {} requests; prompt kernels in the verification pass change the text of {}, the batch-invariant kernels change {}", jobs.len(), before.0, after.0);
        assert_eq!(after.0, 0, "{label}");
    }
}

/// `check_alone`, then all jobs together (the sequences share every batch) with the batch-invariant kernels on for the whole run, which
/// also checks the state rollback of several sequences at once.
fn check_all(label: &str, gguf: &std::path::Path, jobs: &[Job]) {
    check_alone(label, gguf, 100, jobs);
    let _invariant = crate::cpu_kernels::BatchInvariant::enter();
    check_jobs(&format!("{label}, together"), gguf, 100, jobs);
}

#[test]
fn qwen4_f32_speculation_matches_plain_greedy() {
    check_all("qwen4 f32", &model("sx-q4-f32", Arch::Qwen4, Encoding::F32, 0x51_0001), &many_jobs(6, 7));
}

#[test]
fn qwen4_q4_k_m_speculation_matches_plain_greedy() {
    check_all("qwen4 q4_k_m", &model("sx-q4-q4km", Arch::Qwen4, Encoding::Quant("q4_k_m"), 0x51_0002), &many_jobs(6, 11));
}

#[test]
fn qwen35_q4_k_m_speculation_matches_plain_greedy() {
    check_all("qwen35 q4_k_m", &model("sx-q35-q4km", Arch::Qwen35, Encoding::Quant("q4_k_m"), 0x51_0003), &many_jobs(8, 13));
}

#[test]
fn qwen35_q8_0_speculation_matches_plain_greedy() {
    check_all("qwen35 q8_0", &model("sx-q35-q80", Arch::Qwen35, Encoding::Quant("q8_0"), 0x51_0004), &many_jobs(8, 17));
}

#[test]
fn qwen35_q2_k_speculation_matches_plain_greedy() {
    check_all("qwen35 q2_k", &model("sx-q35-q2k", Arch::Qwen35, Encoding::Quant("q2_k"), 0x51_0005), &many_jobs(8, 19));
}

#[test]
fn qwen3moe_q4_k_m_speculation_matches_plain_greedy() {
    check_all("qwen3moe q4_k_m", &model("sx-q3moe-q4km", Arch::Qwen3Moe, Encoding::Quant("q4_k_m"), 0x51_0006), &many_jobs(6, 23));
}

#[test]
fn llama_q4_k_m_speculation_matches_plain_greedy() {
    check_all("llama q4_k_m", &model("sx-llama-q4km", Arch::Llama, Encoding::Quant("q4_k_m"), 0x51_0007), &many_jobs(6, 29));
}
