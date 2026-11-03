//! Speculative decoding on models with recurrent layers (Gated DeltaNet, Mamba 1 and 2, RWKV 6 and 7, Zamba2): a draft that is wrong at
//! chosen positions must give exactly the tokens of plain greedy decoding, because the rejected draft tokens are taken out of the
//! recurrent state again (snapshot before the verification pass, restore and re-run of the accepted prefix after it).
//!
//! The drafts are scripted from the greedy output of a plain run: a chunk is the true continuation of the sequence, cut to a length
//! and corrupted at one position (or not at all). So the test sees chunks that are accepted completely, chunks rejected at the first, a
//! middle and the last position, and a sequence that goes on for many more tokens after each rollback. A second sequence runs
//! alongside the first one, in the same batches.

use crate::hybrid_tests::{falcon_h1_specs, jamba_specs, scratch as hybrid_scratch, write_checkpoint, zamba2_specs};
use crate::runtime::InferenceRuntime;
use crate::sampling::SamplingParams;
use crate::tiny_models::{build_gguf, scratch, Arch, Encoding};
use crate::GgufModel;
use fllm_checkpoint::{convert_checkpoint, ConvertOptions, OutputDType};
use serde_json::Value;
use std::path::{Path, PathBuf};

/// `(draft length, position of the wrong token)`; the schedule repeats. Position 0 is the first token after the verified one.
const SCHEDULE: [(usize, Option<usize>); 9] =
    [(3, None), (4, Some(0)), (4, Some(2)), (2, Some(1)), (4, None), (4, Some(3)), (1, Some(0)), (3, Some(2)), (4, None)];

struct Job {
    id: &'static str,
    prompt: Vec<i64>,
    tokens: usize,
}

fn jobs() -> Vec<Job> {
    vec![
        Job { id: "a", prompt: vec![1, 5, 9, 3, 17, 40], tokens: 40 },
        Job { id: "b", prompt: vec![1, 7, 21, 30, 4, 12, 8], tokens: 33 },
    ]
}

fn runtime(gguf: &Path) -> InferenceRuntime {
    let model = GgufModel::load(gguf.to_str().unwrap(), 64, 4, 2).unwrap();
    let mut rt = InferenceRuntime::with_speculative(model, 4, 64, Vec::new(), 0, 0).unwrap();
    rt.attach_compute("cpu").unwrap();
    rt
}

/// Counts of what the scripted drafts promise: tokens proposed, tokens that are right, chunks of every kind.
#[derive(Default, Debug)]
struct Plan {
    proposed: u64,
    accepted: u64,
    complete_chunks: usize,
    rejected_at: [usize; 4],
}

/// Serves all `jobs` together. With `reference` the script feeds the wrong-at-a-position drafts of `SCHEDULE`; the plan says what was fed.
fn serve(rt: &mut InferenceRuntime, jobs: &[Job], reference: Option<&[Vec<i64>]>, vocab: i64) -> (Vec<Vec<i64>>, Plan) {
    let params = SamplingParams { ignore_eos: true, ..Default::default() };
    let sids: Vec<i64> =
        jobs.iter().map(|j| rt.submit_with_sampling(j.id.into(), j.prompt.clone(), j.tokens, params.clone()).unwrap()).collect();
    let mut out = vec![Vec::<i64>::new(); jobs.len()];
    let mut rounds = vec![0usize; jobs.len()];
    let mut plan = Plan::default();
    let mut guard = 0;
    while out.iter().zip(jobs).any(|(o, j)| o.len() < j.tokens) {
        guard += 1;
        assert!(guard < 5_000, "no progress");
        if let Some(reference) = reference {
            for (i, job) in jobs.iter().enumerate() {
                let done = out[i].len();
                // the first token comes with the prompt pass; the pass after that carries the first draft
                if done == 0 || done >= job.tokens {
                    continue;
                }
                let (len, wrong) = SCHEDULE[(rounds[i] + i * 4) % SCHEDULE.len()];
                rounds[i] += 1;
                let len = len.min(job.tokens - done - 1);
                if len == 0 {
                    continue;
                }
                // the logits of the verified token (the last one emitted) predict `reference[done]`
                let mut draft: Vec<i64> = reference[i][done..done + len].to_vec();
                plan.proposed += len as u64;
                match wrong.filter(|w| *w < len) {
                    Some(w) => {
                        let mut v = (draft[w] + 11) % vocab;
                        while v == draft[w] || v == 2 {
                            v = (v + 1) % vocab;
                        }
                        draft[w] = v;
                        plan.accepted += w as u64;
                        if w < plan.rejected_at.len() {
                            plan.rejected_at[w] += 1;
                        }
                    }
                    None => {
                        plan.accepted += len as u64;
                        plan.complete_chunks += 1;
                    }
                }
                rt.set_test_draft(sids[i], draft);
            }
        }
        for e in rt.step().unwrap() {
            let i = jobs.iter().position(|j| j.id == e.request_id).unwrap();
            if out[i].len() < jobs[i].tokens {
                out[i].push(e.token_id);
            }
        }
    }
    (out, plan)
}

/// Plain greedy decoding of every job on its own, then the same jobs served together with the scripted drafts.
fn check(label: &str, gguf: &Path, vocab: i64) {
    let jobs = jobs();
    let reference: Vec<Vec<i64>> = jobs.iter().map(|j| serve(&mut runtime(gguf), std::slice::from_ref(j), None, vocab).0.remove(0)).collect();
    assert!(reference.iter().all(|r| r.windows(2).any(|w| w[0] != w[1])), "{label}: the plain output is constant, the test would prove little");

    let mut rt = runtime(gguf);
    let (tokens, plan) = serve(&mut rt, &jobs, Some(&reference), vocab);
    let stats = rt.speculative_status().2;
    for (i, job) in jobs.iter().enumerate() {
        assert_eq!(tokens[i], reference[i], "{label}: speculative decoding changed the output of sequence {}", job.id);
    }
    // the scripted chunks really went through the verification pass
    assert_eq!((stats.proposed, stats.accepted), (plan.proposed, plan.accepted), "{label}: {plan:?} {stats:?}");
    assert!(plan.complete_chunks >= 4, "{label}: {plan:?}");
    assert!(plan.rejected_at.iter().all(|n| *n >= 2), "{label}: rejections at the first, second, third and fourth position: {plan:?}");
    assert!(stats.accepted < stats.proposed && stats.accepted > 0, "{label}: {stats:?}");

    // the same runtime serves a new sequence after all those rollbacks (its states were not left behind in a wrong place)
    let (again, _) = serve(&mut rt, &jobs[..1], Some(&reference[..1]), vocab);
    assert_eq!(again[0], reference[0], "{label}: a later sequence differs");
}

fn from_specs(label: &str, config: Value, specs: &[crate::hybrid_tests::Spec], seed: u64) -> PathBuf {
    let dir = hybrid_scratch(label);
    write_checkpoint(&dir, config, specs, seed);
    let gguf = dir.join("model.gguf");
    convert_checkpoint(&dir, &gguf, ConvertOptions { output_dtype: OutputDType::F32, overwrite: false, spec_head: None })
        .unwrap_or_else(|e| panic!("conversion failed: {e}"));
    gguf
}

fn tiny(label: &str, arch: Arch) -> PathBuf {
    build_gguf(&scratch(label), arch, Encoding::F32, 0x1234_5678_9abc_def1)
}

#[test]
fn qwen35_deltanet_speculation_rolls_back_to_plain_greedy() {
    check("qwen35", &tiny("spec-q35", Arch::Qwen35), 100);
}

#[test]
fn rwkv7_speculation_rolls_back_to_plain_greedy() {
    check("rwkv7", &tiny("spec-rwkv7", Arch::Rwkv7), 100);
}

#[test]
fn rwkv6_speculation_rolls_back_to_plain_greedy() {
    check("rwkv6", &tiny("spec-rwkv6", Arch::Rwkv6), 100);
}

#[test]
fn mamba2_attention_hybrid_speculation_rolls_back_to_plain_greedy() {
    let (config, specs) = falcon_h1_specs(true);
    check("falcon-h1", &from_specs("spec-fh1", config, &specs, 0x5eed_0101), 40);
}

#[test]
fn mamba1_moe_hybrid_speculation_rolls_back_to_plain_greedy() {
    let (config, specs) = jamba_specs();
    check("jamba", &from_specs("spec-jamba", config, &specs, 0x5eed_0102), 40);
}

#[test]
fn zamba2_speculation_rolls_back_to_plain_greedy() {
    let (config, specs) = zamba2_specs();
    check("zamba2", &from_specs("spec-zamba2", config, &specs, 0x5eed_0103), 40);
}

#[test]
fn a_draft_model_with_recurrent_layers_matches_plain_greedy_decoding() {
    // the target drafts for itself: every proposal is right, so this exercises the draft model's own state (history replay in chunks,
    // continuing after accepted drafts) and, with the wrong tokens injected above, the target. Then a second, differing draft model.
    let gguf = tiny("spec-draft", Arch::Qwen35);
    let jobs = jobs();
    let reference: Vec<Vec<i64>> = jobs.iter().map(|j| serve(&mut runtime(&gguf), std::slice::from_ref(j), None, 100).0.remove(0)).collect();
    let other = build_gguf(&scratch("spec-draft-other"), Arch::Qwen35, Encoding::F32, 0xfeed_beef_0bad_cafe);
    for (label, draft) in [("same weights", &gguf), ("other weights", &other)] {
        let mut rt = runtime(&gguf);
        rt.enable_draft(GgufModel::load(draft.to_str().unwrap(), 64, 4, 2).unwrap(), 4).unwrap();
        rt.attach_compute("cpu").unwrap();
        let (tokens, _) = serve(&mut rt, &jobs, None, 100);
        assert_eq!(tokens, reference, "draft model with {label}: the output changed");
        let stats = rt.speculative_status().2;
        assert!(stats.steps > 0, "{label}: {stats:?}");
        if label == "same weights" {
            assert!(stats.acceptance_rate() > 0.9, "{stats:?}");
        } else {
            assert!(stats.accepted < stats.proposed, "different weights must be rejected sometimes: {stats:?}");
        }
    }
}
