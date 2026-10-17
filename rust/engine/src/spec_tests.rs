//! End-to-end checks of speculative decoding on the CPU with a tiny synthetic model (the fixture of `import_tests`): every speculation
//! source must reproduce the greedy output of plain decoding exactly, and stochastic requests must stay valid, reproducible and
//! accepted at the rate the construction predicts.

use crate::import_tests::{build_pair, convert, scratch, Kind};
use crate::runtime::InferenceRuntime;
use crate::sampling::{SamplingParams, TokenLogprobs};
use crate::speculative::SuffixConfig;
use crate::GgufModel;
use fllm_checkpoint::OutputDType;
use std::path::PathBuf;

const PROMPT: [i64; 12] = [1, 5, 9, 3, 17, 40, 5, 9, 3, 17, 40, 5];

fn fixture() -> PathBuf {
    static DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    let dir = DIR.get_or_init(|| {
        let (quant, reference) = (scratch("spec-q"), scratch("spec-r"));
        build_pair(Kind::Fp8Block(32), 4242, &quant, &reference);
        let gguf = convert(&quant, OutputDType::Preserve);
        let kept = scratch("spec-kept").join("tiny.gguf");
        std::fs::copy(&gguf, &kept).unwrap();
        let _ = std::fs::remove_dir_all(quant);
        let _ = std::fs::remove_dir_all(reference);
        kept
    });
    dir.clone()
}

fn load() -> GgufModel {
    GgufModel::load(fixture().to_str().unwrap(), 64, 4, 2).unwrap()
}

fn runtime(lookup: usize) -> InferenceRuntime {
    let mut rt = InferenceRuntime::with_speculative(load(), 4, 64, Vec::new(), lookup, 8).unwrap();
    rt.attach_compute("cpu").unwrap();
    rt
}

fn generate(rt: &mut InferenceRuntime, id: &str, tokens: usize, params: SamplingParams) -> (Vec<i64>, Vec<Option<TokenLogprobs>>) {
    rt.submit_with_sampling(id.into(), PROMPT.to_vec(), tokens, params).unwrap();
    let (mut out, mut lps) = (Vec::new(), Vec::new());
    let mut guard = 0;
    while out.len() < tokens {
        guard += 1;
        assert!(guard < 10_000, "no progress");
        for e in rt.step().unwrap() {
            if e.request_id == id {
                out.push(e.token_id);
                lps.push(e.logprobs);
            }
        }
    }
    (out, lps)
}

fn greedy_logprobs() -> SamplingParams {
    SamplingParams { logprobs: Some(3), ..Default::default() }
}

#[test]
fn every_speculation_source_reproduces_greedy_decoding() {
    let (plain, plain_lp) = generate(&mut runtime(0), "plain", 40, greedy_logprobs());

    // prompt lookup (n-gram) in the scheduler
    let mut rt = runtime(4);
    let (tokens, lps) = generate(&mut rt, "ngram", 40, greedy_logprobs());
    assert_eq!(tokens, plain, "n-gram speculation changed the output");
    assert_close_logprobs(&lps, &plain_lp, "ngram");
    let (_, _, stats) = rt.speculative_status();
    assert!(stats.steps > 0, "the tiny model loops, so n-gram lookup must find drafts");

    // suffix matching
    let mut rt = runtime(0);
    rt.enable_suffix(5, SuffixConfig::default()).unwrap();
    let (tokens, lps) = generate(&mut rt, "suffix", 40, greedy_logprobs());
    assert_eq!(tokens, plain, "suffix speculation changed the output");
    assert_close_logprobs(&lps, &plain_lp, "suffix");
    assert!(rt.speculative_status().2.steps > 0);

    // a draft model: the target itself, so everything must be accepted
    let mut rt = runtime(0);
    rt.enable_draft(load(), 4).unwrap();
    rt.attach_compute("cpu").unwrap();
    let (tokens, lps) = generate(&mut rt, "draft", 40, greedy_logprobs());
    assert_eq!(tokens, plain, "draft-model speculation changed the output");
    assert_close_logprobs(&lps, &plain_lp, "draft");
    let stats = rt.speculative_status().2;
    assert!(stats.acceptance_rate() > 0.9, "identical draft and target must agree: {stats:?}");
    assert!(stats.mean_emitted() > 2.0, "{stats:?}");
}

fn assert_close_logprobs(got: &[Option<TokenLogprobs>], want: &[Option<TokenLogprobs>], what: &str) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let (g, w) = (g.as_ref().unwrap(), w.as_ref().unwrap());
        assert!((g.logprob - w.logprob).abs() < 1e-3, "{what}: token {i} logprob {} vs {}", g.logprob, w.logprob);
    }
}

#[test]
fn stochastic_requests_speculate_stay_reproducible_and_accept_a_matching_draft() {
    let params = SamplingParams { temperature: 0.9, top_k: 20, top_p: 0.95, seed: Some(1234), ..Default::default() };
    let mut rt = runtime(0);
    rt.enable_draft(load(), 4).unwrap();
    rt.attach_compute("cpu").unwrap();
    let (a, _) = generate(&mut rt, "a", 40, params.clone());
    let stats = rt.speculative_status().2;
    // the draft draws from exactly the target's filtered distribution: p == q, so every token is accepted
    assert!(stats.steps > 0 && stats.acceptance_rate() > 0.97, "{stats:?}");
    assert!(a.iter().all(|&t| (0..64).contains(&t)));

    // same seed, same output (new runtime, new sequence ids)
    let mut rt2 = runtime(0);
    rt2.enable_draft(load(), 4).unwrap();
    rt2.attach_compute("cpu").unwrap();
    let (b, _) = generate(&mut rt2, "b", 40, params.clone());
    assert_eq!(a, b, "seeded speculative sampling must be reproducible");

    // a different seed gives a different sequence
    let other = SamplingParams { seed: Some(99), ..params.clone() };
    let (c, _) = generate(&mut rt2, "c", 40, other);
    assert_ne!(a, c);

    // sampling with n-gram drafts (point-mass q) and logprobs of the emitted tokens
    let mut rt3 = runtime(4);
    let with_lp = SamplingParams { logprobs: Some(2), ..params.clone() };
    let (d, lps) = generate(&mut rt3, "d", 40, with_lp);
    assert_eq!(d.len(), 40);
    assert!(lps.iter().all(|l| l.as_ref().is_some_and(|l| l.logprob <= 0.0 && l.top.len() == 2)));
}

#[test]
fn a_request_can_opt_out_of_speculation() {
    let mut rt = runtime(4);
    let params = SamplingParams { speculative: false, ..Default::default() };
    let (tokens, _) = generate(&mut rt, "off", 30, params);
    assert_eq!(tokens.len(), 30);
    assert_eq!(rt.speculative_status().2.steps, 0);
    // the engine still speculates for requests that did not opt out
    generate(&mut rt, "on", 30, SamplingParams::default());
    assert!(rt.speculative_status().2.steps > 0);
}

#[test]
fn speculation_with_min_tokens_allow_lists_and_bad_words_matches_plain_decoding() {
    let (free, _) = generate(&mut runtime(0), "free", 30, SamplingParams::default());
    let banned = free[3];
    let params = SamplingParams {
        bad_word_ids: vec![vec![banned]],
        min_tokens: 5,
        stop_token_ids: vec![],
        ..Default::default()
    };
    let (plain, _) = generate(&mut runtime(0), "plain", 30, params.clone());
    assert!(!plain.contains(&banned));
    let mut rt = runtime(4);
    let (spec, _) = generate(&mut rt, "spec", 30, params);
    assert_eq!(spec, plain);
    let allowed = SamplingParams { allowed_token_ids: vec![5, 9, 17], ..Default::default() };
    let (plain, _) = generate(&mut runtime(0), "plain2", 20, allowed.clone());
    assert!(plain.iter().all(|t| [5, 9, 17].contains(t)));
    let (spec, _) = generate(&mut runtime(4), "spec2", 20, allowed);
    assert_eq!(spec, plain);
}

#[test]
fn stateless_models_have_no_stateful_speculation_guard_in_the_way() {
    // enabling suffix twice or combining sources is refused with a clear message
    let mut rt = runtime(0);
    rt.enable_suffix(4, SuffixConfig::default()).unwrap();
    assert!(rt.enable_draft(load(), 4).unwrap_err().contains("mutually exclusive"));
    assert!(rt.enable_mtp().unwrap_err().contains("mutually exclusive"));
}

/// A runtime whose KV pool is too small for the sequences it serves, so the scheduler must preempt.
fn tight_runtime(mode: &str) -> InferenceRuntime {
    let mut model = load();
    model.set_pool_blocks(model.pool_blocks().min(9));
    let mut rt = InferenceRuntime::with_speculative(model, 4, 64, Vec::new(), 0, 0).unwrap();
    rt.attach_compute("cpu").unwrap();
    let cfg = rt.scheduler_config_mut();
    cfg.preemption = mode.into();
    cfg.swap_blocks = 64;
    cfg.watermark_blocks = 0;
    rt
}

fn serve_together(rt: &mut InferenceRuntime, count: usize, tokens: usize) -> Vec<Vec<i64>> {
    for i in 0..count {
        let mut prompt = PROMPT.to_vec();
        prompt.rotate_left(i);
        rt.submit_with_sampling(format!("r{i}"), prompt, tokens, SamplingParams::default()).unwrap();
    }
    let mut out = vec![Vec::new(); count];
    let mut guard = 0;
    while out.iter().any(|o| o.len() < tokens) {
        guard += 1;
        assert!(guard < 20_000, "no progress");
        for e in rt.step().unwrap() {
            let i: usize = e.request_id[1..].parse().unwrap();
            out[i].push(e.token_id);
        }
    }
    out
}

#[test]
fn preemption_by_swap_and_by_recompute_give_the_tokens_of_an_unpressured_run() {
    // reference: every request alone in a roomy pool
    let mut expected = Vec::new();
    for i in 0..4 {
        let mut prompt = PROMPT.to_vec();
        prompt.rotate_left(i);
        let mut rt = runtime(0);
        rt.submit_with_sampling("solo".into(), prompt, 36, SamplingParams::default()).unwrap();
        let mut tokens = Vec::new();
        while tokens.len() < 36 {
            tokens.extend(rt.step().unwrap().into_iter().map(|e| e.token_id));
        }
        expected.push(tokens);
    }
    for mode in ["recompute", "swap"] {
        let mut rt = tight_runtime(mode);
        let got = serve_together(&mut rt, 4, 36);
        let (preemptions, out_blocks, in_blocks) = rt.preemption_stats();
        assert!(preemptions > 0, "{mode}: the pool is too small for three sequences, something must be preempted");
        assert_eq!(got, expected, "{mode}: preemption changed the output");
        if mode == "swap" {
            assert!(out_blocks > 0 && in_blocks == out_blocks, "swap traffic out {out_blocks} in {in_blocks}");
        } else {
            assert_eq!((out_blocks, in_blocks), (0, 0));
        }
    }
}

#[test]
fn a_cancelled_swapped_request_releases_its_swap_space() {
    let mut rt = tight_runtime("swap");
    for i in 0..4 {
        rt.submit_with_sampling(format!("r{i}"), PROMPT.to_vec(), 36, SamplingParams::default()).unwrap();
    }
    for _ in 0..200 {
        rt.step().unwrap();
        if rt.preemption_stats().1 > 0 {
            break;
        }
    }
    assert!(rt.preemption_stats().1 > 0, "nothing was swapped out");
    for i in 0..4 {
        rt.cancel_request(&format!("r{i}")).unwrap();
    }
    assert_eq!(rt.active_requests(), 0);
    // the engine keeps working afterwards
    let (tokens, _) = generate(&mut rt, "after", 5, SamplingParams::default());
    assert_eq!(tokens.len(), 5);
}

#[test]
fn forking_choices_works_through_the_block_copy_of_the_runtime() {
    let mut rt = runtime(0);
    assert!(rt.supports_shared_choices() && rt.kv_block_ops_available());
    let params = SamplingParams { choices: 3, temperature: 0.8, seed: Some(5), ..Default::default() };
    rt.submit_with_sampling("n3".into(), PROMPT.to_vec(), 6, params).unwrap();
    let mut per_choice = std::collections::BTreeMap::<usize, Vec<i64>>::new();
    let mut guard = 0;
    while per_choice.values().map(Vec::len).sum::<usize>() < 18 {
        guard += 1;
        assert!(guard < 1000);
        for e in rt.step().unwrap() {
            per_choice.entry(e.choice_index).or_default().push(e.token_id);
        }
    }
    assert_eq!(per_choice.len(), 3);
}

#[test]
fn the_device_sampling_job_encodes_penalties_bias_controls_and_masks() {
    let mut model = load();
    model.set_pool_blocks(8);
    let vocab = model.vocab_size();
    let mut rt = InferenceRuntime::with_speculative(model, 2, 64, vec![2], 0, 0).unwrap();
    rt.attach_compute("cpu").unwrap();
    let params = SamplingParams {
        temperature: 0.7,
        top_k: 9,
        repetition_penalty: 1.2,
        logit_bias: vec![(7, 1.5), (7, 0.5), (3, -2.0)],
        min_tokens: 4,
        allowed_token_ids: vec![1, 33],
        bad_word_ids: vec![vec![5], vec![9, 3]],
        logprobs: Some(3),
        seed: Some(4),
        ..Default::default()
    };
    let sid = rt.submit_with_sampling("job".into(), PROMPT.to_vec(), 8, params).unwrap();
    let job = rt.device_sample_job(&[sid]).unwrap();
    assert_eq!(job.rows.len(), 1);
    let row = job.rows[0];
    assert_eq!((row.top_k, row.seed, row.draw, row.logprobs_mode, row.top_n), (9, 4, 0, 1, 3));
    // the distinct prompt tokens with their counts
    assert!(job.pen.contains(&(5, 3)) && job.pen.contains(&(40, 2)) && job.pen.contains(&(1, 1)));
    assert_eq!(row.pen_len as usize, job.pen.len());
    // bias merged per token; EOS (2) forbidden by min_tokens; the single-token bad word 5; the two-token word 9 3 is not active yet
    let bias: std::collections::HashMap<i32, f32> = job.bias.iter().copied().collect();
    assert_eq!(bias[&7], 2.0);
    assert_eq!(bias[&3], -2.0);
    assert_eq!(bias[&2], f32::NEG_INFINITY);
    assert_eq!(bias[&5], f32::NEG_INFINITY);
    assert_eq!(row.mask_row, 0);
    assert_eq!(job.mask.len(), vocab.div_ceil(32));
    assert_eq!(job.mask[0], 1 << 1);
    assert_eq!(job.mask[1], 1 << 1);
    // mirostat and DRY stay on the host
    let host_only = SamplingParams { mirostat: 2, temperature: 1.0, ..Default::default() };
    let sid2 = rt.submit_with_sampling("host".into(), PROMPT.to_vec(), 8, host_only).unwrap();
    assert!(rt.device_sample_job(&[sid2]).is_none());
}

#[test]
fn processed_log_probabilities_are_reported_on_the_plain_and_the_speculative_path() {
    // greedy: the processed distribution is a point mass, so the chosen token has probability one
    let greedy = SamplingParams { logprobs: Some(3), logprobs_processed: true, ..Default::default() };
    let (tokens, lps) = generate(&mut runtime(0), "g", 12, greedy);
    for (t, lp) in tokens.iter().zip(&lps) {
        let lp = lp.as_ref().unwrap();
        assert_eq!(lp.logprob, 0.0);
        assert_eq!(lp.top, vec![(*t as u32, 0.0)]);
    }
    // sampled with top-k 4: the listed alternatives are the survivors and sum to one, with and without speculation
    let sampled = SamplingParams { temperature: 0.9, top_k: 4, logprobs: Some(8), logprobs_processed: true, seed: Some(3), ..Default::default() };
    let mut speculative = runtime(0);
    speculative.enable_draft(load(), 3).unwrap();
    speculative.attach_compute("cpu").unwrap();
    for rt in [&mut runtime(0), &mut speculative] {
        let (tokens, lps) = generate(rt, "s", 20, sampled.clone());
        for (t, lp) in tokens.iter().zip(&lps) {
            let lp = lp.as_ref().unwrap();
            assert_eq!(lp.top.len(), 4);
            assert!((lp.top.iter().map(|x| x.1.exp()).sum::<f64>() - 1.0).abs() < 1e-4);
            assert!(lp.top.iter().any(|x| x.0 == *t as u32), "the sampled token is one of the survivors");
        }
    }
    assert!(speculative.speculative_status().2.steps > 0);
}

/// Runs a grammar-constrained greedy request to its end (EOS or the token limit) and returns the tokens.
fn constrained(rt: &mut InferenceRuntime, id: &str, tokens: usize) -> Vec<i64> {
    let model = load();
    let tokenizer = crate::tokenizer::TokenizerBundle::from_gguf(model.mapping()).unwrap();
    let factory = tokenizer.grammar_factory().unwrap();
    let grammar = crate::structured::request_grammar(&serde_json::json!({"grammar": "root ::= (\"t5\" | \"t9\" | \"t6\" | \"t7\")+"}))
        .unwrap()
        .unwrap();
    let matcher = crate::structured::matcher(&factory, grammar).unwrap();
    let sid = rt.submit_with_sampling(id.into(), PROMPT.to_vec(), tokens, SamplingParams::default()).unwrap();
    rt.set_constraint(sid, matcher).unwrap();
    let mut out = Vec::new();
    let mut guard = 0;
    loop {
        guard += 1;
        assert!(guard < 10_000, "no progress");
        let mut done = false;
        for e in rt.step().unwrap() {
            if e.request_id == id {
                out.push(e.token_id);
                done |= e.request_finished;
            }
        }
        if done {
            return out;
        }
    }
}

#[test]
fn grammar_constrained_requests_speculate_and_stay_inside_the_grammar() {
    let eos = vec![2];
    let make = |lookup: usize| {
        let mut rt = InferenceRuntime::with_speculative(load(), 4, 64, eos.clone(), lookup, 8).unwrap();
        rt.attach_compute("cpu").unwrap();
        rt
    };
    let plain = constrained(&mut make(0), "plain", 30);
    assert!(plain.len() > 5);
    assert!(plain.iter().all(|t| [5, 9, 6, 7, 2].contains(t)), "outside the grammar: {plain:?}");
    // n-gram drafts and a draft model: the same tokens, and speculation really ran
    let mut ngram = make(4);
    assert_eq!(constrained(&mut ngram, "ngram", 30), plain);
    assert!(ngram.speculative_status().2.steps > 0);
    let mut drafted = make(0);
    drafted.enable_draft(load(), 3).unwrap();
    drafted.attach_compute("cpu").unwrap();
    assert_eq!(constrained(&mut drafted, "draft", 30), plain);
    assert!(drafted.speculative_status().2.steps > 0);
}
