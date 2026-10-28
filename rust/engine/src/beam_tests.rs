//! Beam search and multi-step scheduling on the tiny random model (CPU). This module is a child of `runtime`, so it can look at
//! the block tables and reference counts directly.

use super::*;
use crate::beam::{BeamParams, EarlyStopping, Hypothesis, BUSY, REFUSED};
use crate::sampling::SamplingParams;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

const PROMPT: [i64; 20] = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29];

fn gguf() -> PathBuf {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("fllm-beam-tests-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.gguf");
        crate::tiny_model::write_gguf(&path, &crate::tiny_model::Options::default()).unwrap();
        path
    })
    .clone()
}

fn runtime_with(eos: Vec<i64>, lookup: usize) -> InferenceRuntime {
    let model = GgufModel::load(gguf().to_str().unwrap(), 128, 8, 2).unwrap();
    let mut rt = InferenceRuntime::with_speculative(model, 8, 128, eos, lookup, 8).unwrap();
    rt.attach_compute("cpu").unwrap();
    rt
}

fn runtime(eos: Vec<i64>) -> InferenceRuntime {
    runtime_with(eos, 0)
}

struct Finished {
    tokens: Vec<i64>,
    reason: Option<&'static str>,
}

/// Runs the runtime until the given requests are done; returns the tokens of each in order.
fn drive(rt: &mut InferenceRuntime, ids: &[&str]) -> Vec<Finished> {
    let mut out: Vec<Finished> = ids.iter().map(|_| Finished { tokens: Vec::new(), reason: None }).collect();
    let mut guard = 0;
    while rt.active_requests() > 0 {
        guard += 1;
        assert!(guard < 20_000, "no progress");
        for event in rt.step().unwrap() {
            let index = ids.iter().position(|id| *id == event.request_id).expect("event of an unknown request");
            out[index].tokens.push(event.token_id);
            if event.finished {
                out[index].reason = event.finish_reason;
            }
        }
    }
    out
}

fn greedy(rt: &mut InferenceRuntime, id: &str, prompt: &[i64], max_tokens: usize) -> Finished {
    rt.submit_with_sampling(id.into(), prompt.to_vec(), max_tokens, SamplingParams::default()).unwrap();
    drive(rt, &[id]).pop().unwrap()
}

fn beam(rt: &mut InferenceRuntime, id: &str, prompt: &[i64], max_tokens: usize, params: BeamParams) -> Vec<Hypothesis> {
    rt.run_beam_search(id, prompt.to_vec(), max_tokens, SamplingParams::default(), params).unwrap()
}

fn free_after_clearing_cache(rt: &mut InferenceRuntime) -> usize {
    rt.clear_prefix_cache().unwrap();
    rt.available_blocks()
}

#[test]
fn beam_width_one_equals_greedy_decoding() {
    let mut rt = runtime(Vec::new());
    let plain = greedy(&mut rt, "g", &PROMPT, 24);
    assert_eq!(plain.tokens.len(), 24);
    let found = beam(&mut rt, "b", &PROMPT, 24, BeamParams::new(1, 1));
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].tokens, plain.tokens, "width 1 must be greedy decoding");
    assert_eq!(found[0].finish_reason, "length");
}

#[test]
fn beam_width_one_equals_greedy_decoding_with_an_end_token() {
    // pick a token that greedy decoding produces after a few steps and call it the end token
    let probe = greedy(&mut runtime(Vec::new()), "p", &PROMPT, 12);
    let position = (2..12).find(|&i| !probe.tokens[..i].contains(&probe.tokens[i])).expect("a first occurrence");
    let eos = probe.tokens[position];
    let mut rt = runtime(vec![eos]);
    let plain = greedy(&mut rt, "g", &PROMPT, 24);
    assert_eq!(plain.reason, Some("stop"));
    assert_eq!(plain.tokens.len(), position + 1);
    let found = beam(&mut rt, "b", &PROMPT, 24, BeamParams::new(1, 1));
    assert_eq!(found[0].tokens, plain.tokens, "the end token is part of the hypothesis, like in the plain path");
    assert_eq!(found[0].finish_reason, "stop");
}

#[test]
fn a_wider_beam_scores_at_least_as_well_as_greedy_and_its_numbers_are_right() {
    let mut rt = runtime(Vec::new());
    let params = |width: usize, returned: usize| BeamParams { length_penalty: 0.0, ..BeamParams::new(width, returned) };
    let narrow = beam(&mut rt, "n", &PROMPT, 30, params(1, 1));
    let wide = beam(&mut rt, "w", &PROMPT, 30, params(4, 4));
    assert_eq!(wide.len(), 4);
    assert!(
        wide[0].logprob >= narrow[0].logprob - 1e-9,
        "best of 4 beams ({}) must not score below greedy ({})",
        wide[0].logprob,
        narrow[0].logprob
    );
    for pair in wide.windows(2) {
        assert!(pair[0].score >= pair[1].score, "hypotheses come out best first");
    }
    let distinct: std::collections::HashSet<_> = wide.iter().map(|h| h.tokens.clone()).collect();
    assert_eq!(distinct.len(), 4, "four different hypotheses");
    eprintln!(
        "beam scores (length_penalty 0): greedy {:.6}, width 4: {:?}",
        narrow[0].logprob,
        wide.iter().map(|h| h.logprob).collect::<Vec<_>>()
    );

    // The reported log-probabilities are checked against teacher forcing in a fresh runtime that shares nothing with the beams:
    // for every position, the prompt plus the hypothesis so far, and the next token forced by `allowed_token_ids`.
    let mut fresh = runtime(Vec::new());
    for (rank, hypothesis) in wide.iter().enumerate() {
        let mut total = 0.0;
        for (index, &token) in hypothesis.tokens.iter().enumerate() {
            let mut prompt = PROMPT.to_vec();
            prompt.extend_from_slice(&hypothesis.tokens[..index]);
            let sampling = SamplingParams { logprobs: Some(0), allowed_token_ids: vec![token as usize], ..Default::default() };
            let id = format!("tf{rank}-{index}");
            fresh.submit_with_sampling(id.clone(), prompt, 1, sampling).unwrap();
            let mut seen = None;
            while fresh.active_requests() > 0 {
                for event in fresh.step().unwrap() {
                    assert_eq!(event.token_id, token);
                    seen = event.logprobs.map(|l| l.logprob);
                }
            }
            let forced = seen.expect("a log-probability");
            assert!(
                (forced - hypothesis.token_logprobs[index]).abs() < 1e-3,
                "hypothesis {rank}, token {index}: beam says {}, teacher forcing says {forced}",
                hypothesis.token_logprobs[index]
            );
            total += forced;
        }
        assert!((total - hypothesis.logprob).abs() < 1e-2, "the hypothesis score is the sum of its token log-probabilities");
    }
}

#[test]
fn beam_search_is_deterministic_and_leaves_no_blocks_behind() {
    let mut rt = runtime(Vec::new());
    let before = free_after_clearing_cache(&mut rt);
    let first = beam(&mut rt, "a", &PROMPT, 40, BeamParams::new(4, 2));
    assert_eq!(first.len(), 2);
    assert_eq!(free_after_clearing_cache(&mut rt), before, "every block is given back");
    let second = beam(&mut rt, "b", &PROMPT, 40, BeamParams::new(4, 2));
    assert_eq!(first, second);
    assert_eq!(free_after_clearing_cache(&mut rt), before);
    assert_eq!(rt.active_requests(), 0);
}

#[test]
fn forking_shares_full_blocks_and_copies_the_tail() {
    let mut rt = runtime(Vec::new());
    let block_size = rt.model.block_size();
    assert!(block_size <= 16, "the test prompt must cross a block boundary");
    rt.submit_with_sampling("f".into(), PROMPT.to_vec(), 4, SamplingParams::default()).unwrap();
    let leader = 1;
    // run until the prompt is computed (the logits are pending)
    let mut guard = 0;
    while !rt.pending_logits.contains_key(&leader) {
        guard += 1;
        assert!(guard < 100);
        rt.step().unwrap();
    }
    let free_before = rt.available_blocks();
    let child = rt.fork_sequence(leader).unwrap();
    let parent_blocks = rt.scheduler.bm.get_table(leader).unwrap().blocks.clone();
    let child_blocks = rt.scheduler.bm.get_table(child).unwrap().blocks.clone();
    let full = PROMPT.len() / block_size;
    assert!(full >= 1);
    assert_eq!(parent_blocks[..full], child_blocks[..full], "completely computed blocks are shared");
    for &block in &child_blocks[..full] {
        assert_eq!(rt.scheduler.bm.allocator.refcounts[block], 2 + usize::from(rt.scheduler.cfg.prefix_cache), "the table of each sequence, plus the prefix cache");
    }
    if !PROMPT.len().is_multiple_of(block_size) {
        assert_ne!(parent_blocks[full], child_blocks[full], "the tail block is copied");
        assert_eq!(free_before - rt.available_blocks(), 1, "a fork costs one block, not a copy of the prompt");
    }
    // the copy equals the original, byte for byte
    if !PROMPT.len().is_multiple_of(block_size) {
        let a = rt.kv_export_block(parent_blocks[full]).unwrap();
        let b = rt.kv_export_block(child_blocks[full]).unwrap();
        assert_eq!(a, b);
    }
    // giving the fork up returns exactly what it took
    rt.cancel_request("f").unwrap();
    assert_eq!(free_after_clearing_cache(&mut rt), rt.model.pool_blocks() - 1);
}

#[test]
fn the_beam_budget_is_checked_up_front() {
    let mut rt = runtime(Vec::new());
    let submit = |rt: &mut InferenceRuntime, id: &str, width: usize, max_tokens: usize, sampling: SamplingParams| {
        rt.submit_beam(id.into(), PROMPT.to_vec(), max_tokens, sampling, BeamParams::new(width, 1))
    };
    let error = submit(&mut rt, "w9", 9, 8, SamplingParams::default()).unwrap_err();
    assert!(error.starts_with(REFUSED), "{error}");
    let error = submit(&mut rt, "lp", 2, 8, SamplingParams { logprobs: Some(1), ..Default::default() }).unwrap_err();
    assert!(error.starts_with(REFUSED), "{error}");
    let error = submit(&mut rt, "rp", 2, 8, SamplingParams { repetition_penalty: 1.3, ..Default::default() }).unwrap_err();
    assert!(error.starts_with(REFUSED), "{error}");
    let error = rt
        .submit_beam("bad".into(), PROMPT.to_vec(), 8, SamplingParams::default(), BeamParams::new(2, 3))
        .unwrap_err();
    assert!(error.starts_with(REFUSED), "returned > width: {error}");
    // 8 beams of 100 tokens do not fit the pool of this runtime, whatever else is running
    let usable = rt.model.pool_blocks() - 1;
    let error = submit(&mut rt, "big", 8, 100, SamplingParams::default()).unwrap_err();
    assert!(error.starts_with(REFUSED) && error.contains(&usable.to_string()), "{error}");
    assert_eq!(rt.active_requests(), 0, "a refusal leaves nothing behind");
    // slots: 8 in total, so 5 + 5 cannot both be held
    submit(&mut rt, "one", 5, 8, SamplingParams::default()).unwrap();
    let error = submit(&mut rt, "two", 5, 8, SamplingParams::default()).unwrap_err();
    assert!(error.starts_with(BUSY), "{error}");
    rt.cancel_request("one").unwrap();
    assert_eq!(free_after_clearing_cache(&mut rt), rt.model.pool_blocks() - 1);
}

#[test]
fn a_stop_string_check_ends_hypotheses_with_stop() {
    let mut rt = runtime(Vec::new());
    let plain = beam(&mut rt, "p", &PROMPT, 20, BeamParams::new(1, 1));
    let stop_token = plain[0].tokens[3];
    let check: crate::beam::StopCheck = Arc::new(move |tokens: &[i64]| tokens.last() == Some(&stop_token));
    let params = BeamParams { stop_check: Some(check), ..BeamParams::new(1, 1) };
    let found = beam(&mut rt, "s", &PROMPT, 20, params);
    assert_eq!(found[0].finish_reason, "stop");
    assert!(found[0].tokens.len() <= 4, "greedy reaches the stop token at index 3: {:?}", found[0].tokens);
    assert_eq!(found[0].tokens.last(), Some(&stop_token));
}

#[test]
fn early_stopping_variants_agree_on_the_winner_when_no_end_token_exists() {
    let mut rt = runtime(Vec::new());
    let mut results = Vec::new();
    for rule in [EarlyStopping::Heuristic, EarlyStopping::Always, EarlyStopping::Never] {
        let params = BeamParams { early_stopping: rule, ..BeamParams::new(3, 1) };
        results.push(beam(&mut rt, "e", &PROMPT, 16, params));
    }
    assert_eq!(results[0], results[1]);
    assert_eq!(results[0], results[2]);
}

#[test]
fn beam_searches_and_plain_requests_run_side_by_side() {
    let mut solo = runtime(Vec::new());
    let alone_plain = greedy(&mut solo, "g", &PROMPT[..7], 20).tokens;
    let alone_beam = beam(&mut solo, "b", &PROMPT, 20, BeamParams::new(3, 2));
    // the same three, interleaved: two searches (3 + 3 slots) and a plain request
    let mut rt = runtime(Vec::new());
    rt.submit_with_sampling("g".into(), PROMPT[..7].to_vec(), 20, SamplingParams::default()).unwrap();
    let mut first = BeamParams::new(3, 2);
    first.keep_result = true;
    rt.submit_beam("b1".into(), PROMPT.to_vec(), 20, SamplingParams::default(), first.clone()).unwrap();
    rt.submit_beam("b2".into(), PROMPT.to_vec(), 20, SamplingParams::default(), first).unwrap();
    let mut plain = Vec::new();
    let mut guard = 0;
    while rt.active_requests() > 0 {
        guard += 1;
        assert!(guard < 20_000, "no progress");
        for event in rt.step().unwrap() {
            if event.request_id == "g" {
                plain.push(event.token_id);
            }
        }
        for request in rt.beam_outbox_requests() {
            rt.take_beam_events(&request, usize::MAX);
        }
    }
    assert_eq!(plain, alone_plain, "a beam search next to it does not change a plain request");
    assert_eq!(rt.beam_results.get("b1").map(|r| r.len()), Some(2));
    // identical tokens; the sums may differ in the last bits because the rows of a bigger batch are summed in another order
    for name in ["b1", "b2"] {
        let got = &rt.beam_results[name];
        assert_eq!(got.len(), alone_beam.len());
        for (a, b) in got.iter().zip(&alone_beam) {
            assert_eq!(a.tokens, b.tokens);
            assert_eq!(a.finish_reason, b.finish_reason);
            assert!((a.logprob - b.logprob).abs() < 1e-4, "{name}: {} vs {}", a.logprob, b.logprob);
        }
    }
    assert_eq!(free_after_clearing_cache(&mut rt), rt.model.pool_blocks() - 1);
}

#[test]
fn the_events_of_a_beam_search_carry_rank_reason_and_the_last_marker() {
    let mut rt = runtime(Vec::new());
    rt.submit_beam("e".into(), PROMPT.to_vec(), 10, SamplingParams::default(), BeamParams::new(3, 2)).unwrap();
    let mut guard = 0;
    while rt.beam_outbox_requests().is_empty() {
        guard += 1;
        assert!(guard < 1_000);
        assert!(rt.step().unwrap().is_empty(), "nothing is streamed while the search runs");
    }
    assert_eq!(rt.active_requests(), 1, "the request stays active until its events are taken");
    let mut events = Vec::new();
    loop {
        let batch = rt.take_beam_events("e", 7);
        if batch.is_empty() {
            break;
        }
        events.extend(batch);
    }
    assert_eq!(rt.active_requests(), 0);
    assert_eq!(events.iter().filter(|e| e.request_finished).count(), 1);
    assert!(events.last().unwrap().request_finished);
    for choice in 0..2 {
        let mine: Vec<_> = events.iter().filter(|e| e.choice_index == choice).collect();
        assert_eq!(mine.len(), 10);
        assert_eq!(mine.iter().filter(|e| e.finished).count(), 1);
        assert_eq!(mine.last().unwrap().finish_reason, Some("length"));
    }
}

#[test]
fn cancelling_a_beam_search_gives_everything_back() {
    let mut rt = runtime(Vec::new());
    let before = free_after_clearing_cache(&mut rt);
    rt.submit_beam("c".into(), PROMPT.to_vec(), 30, SamplingParams::default(), BeamParams::new(4, 4)).unwrap();
    for _ in 0..12 {
        rt.step().unwrap();
    }
    assert!(rt.available_blocks() < before);
    rt.cancel_request("c").unwrap();
    assert_eq!(rt.active_requests(), 0);
    assert_eq!(free_after_clearing_cache(&mut rt), before);
    assert!(rt.beam_seq.is_empty() && rt.beams.is_empty());
}

// ---------------------------------------------------------------- multi-step scheduling

/// Runs requests `(prompt, max_tokens, seed)` to the end; returns, per request, its tokens and finish reason, plus the batch stats.
type Job<'a> = (&'a [i64], usize, Option<u64>);
type Outcome = (Vec<i64>, Option<&'static str>);

fn run_batch(rt: &mut InferenceRuntime, jobs: &[Job]) -> (Vec<Outcome>, BatchStats) {
    let ids: Vec<String> = (0..jobs.len()).map(|i| format!("r{i}")).collect();
    for (id, (prompt, max, seed)) in ids.iter().zip(jobs) {
        let sampling = match seed {
            Some(seed) => SamplingParams { temperature: 1.0, top_k: 40, seed: Some(*seed), ..Default::default() },
            None => SamplingParams::default(),
        };
        rt.submit_with_sampling(id.clone(), prompt.to_vec(), *max, sampling).unwrap();
    }
    let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    let done = drive(rt, &refs);
    (done.into_iter().map(|f| (f.tokens, f.reason)).collect(), rt.batch_stats())
}

#[test]
fn multi_step_output_equals_single_step_output_token_for_token() {
    let jobs: Vec<(&[i64], usize, Option<u64>)> = vec![
        (&PROMPT[..9], 31, None),
        (&PROMPT[..14], 17, None),
        (&PROMPT[..], 40, Some(7)),
        (&PROMPT[..5], 23, Some(99)),
    ];
    // the end token: something sequence 0 produces, so one request stops early
    let probe = greedy(&mut runtime(Vec::new()), "p", &PROMPT[..9], 12);
    let eos = probe.tokens[6];
    for eos_ids in [Vec::new(), vec![eos]] {
        let mut single = runtime(eos_ids.clone());
        let (expected, single_stats) = run_batch(&mut single, &jobs);
        for rounds in [2usize, 4, 16] {
            let mut multi = runtime(eos_ids.clone());
            assert_eq!(multi.set_multi_step(rounds).unwrap(), rounds);
            let (got, stats) = run_batch(&mut multi, &jobs);
            assert_eq!(got, expected, "multi-step {rounds} (end tokens {eos_ids:?}) changed the output");
            assert_eq!(stats.forward_calls, single_stats.forward_calls, "the same forward passes, only fewer scheduler rounds");
            assert_eq!(stats.decode_rows, single_stats.decode_rows);
            assert!(stats.replayed_rounds > 0, "multi-step {rounds} never reused a batch");
            eprintln!(
                "multi-step {rounds}, end tokens {}: {} forward passes, {} reused batches, scheduler {} ns (single-step {} ns)",
                eos_ids.len(),
                stats.forward_calls,
                stats.replayed_rounds,
                stats.scheduler_ns,
                single_stats.scheduler_ns
            );
        }
        assert!(expected.iter().any(|(tokens, reason)| *reason == Some("length") && !tokens.is_empty()));
    }
    // a stop at the end token really stops: nothing follows it
    let mut multi = runtime(vec![eos]);
    multi.set_multi_step(8).unwrap();
    let (got, _) = run_batch(&mut multi, &jobs[..1]);
    assert_eq!(got[0].1, Some("stop"));
    assert_eq!(got[0].0.last(), Some(&eos));
    assert_eq!(got[0].0.iter().filter(|t| **t == eos).count(), 1);
}

#[test]
fn multi_step_never_generates_past_max_tokens() {
    let mut rt = runtime(Vec::new());
    rt.set_multi_step(16).unwrap();
    for max in [1usize, 2, 3, 5] {
        let (got, _) = run_batch(&mut rt, &[(&PROMPT[..6], max, None), (&PROMPT[..8], max + 1, None)]);
        assert_eq!(got[0].0.len(), max);
        assert_eq!(got[1].0.len(), max + 1);
        assert!(got.iter().all(|(_, reason)| *reason == Some("length")));
    }
}

#[test]
fn speculative_decoding_and_beam_search_keep_the_single_step() {
    let jobs: Vec<(&[i64], usize, Option<u64>)> = vec![(&PROMPT[..], 30, None)];
    let mut spec = runtime_with(Vec::new(), 4);
    spec.set_multi_step(4).unwrap();
    let (spec_out, stats) = run_batch(&mut spec, &jobs);
    assert_eq!(stats.replayed_rounds, 0, "lookup speculation falls back to single steps");
    let (plain_out, _) = run_batch(&mut runtime(Vec::new()), &jobs);
    assert_eq!(spec_out, plain_out);

    let mut rt = runtime(Vec::new());
    rt.set_multi_step(4).unwrap();
    let found = beam(&mut rt, "b", &PROMPT, 12, BeamParams::new(2, 1));
    assert_eq!(found.len(), 1);
    assert_eq!(rt.batch_stats().replayed_rounds, 0, "beam search never replays a batch");
}

#[test]
fn multi_step_is_off_by_default_and_validated() {
    let mut rt = runtime(Vec::new());
    assert_eq!(rt.scheduler.cfg.multi_step, 1);
    assert!(rt.set_multi_step(0).is_err());
    assert!(rt.set_multi_step(17).is_err());
    let (_, stats) = run_batch(&mut rt, &[(&PROMPT[..], 10, None)]);
    assert_eq!(stats.replayed_rounds, 0);
}
