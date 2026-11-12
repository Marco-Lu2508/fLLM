//! A model with recurrent layers (Gated DeltaNet, RWKV, Mamba, and the hybrids with attention) reads a prompt that starts like an earlier
//! one from a checkpoint of its state instead of from the start. The result has to be the result of reading everything again, also when
//! several prompts that share the start run at the same time (they share the cached KV block that the state would otherwise hang on),
//! when the start is shared but the prompts differ after it, and when the checkpoints are evicted.

use crate::hybrid_tests::{falcon_h1_specs, jamba_specs, scratch as hybrid_scratch, write_checkpoint, zamba2_specs};
use crate::runtime::InferenceRuntime;
use crate::sampling::SamplingParams;
use crate::tiny_models::{build_gguf, scratch, Arch, Encoding};
use crate::GgufModel;
use fllm_checkpoint::{convert_checkpoint, ConvertOptions, OutputDType};
use std::path::{Path, PathBuf};

fn runtime(gguf: &Path) -> InferenceRuntime {
    let model = GgufModel::load(gguf.to_str().unwrap(), 128, 4, 2).unwrap();
    let mut rt = InferenceRuntime::with_speculative(model, 4, 64, Vec::new(), 0, 0).unwrap();
    rt.attach_compute("cpu").unwrap();
    rt
}

/// A prompt of `len` tokens made from the seed (no end-of-sequence token, different seeds differ everywhere).
fn prompt(seed: i64, len: usize, vocab: i64) -> Vec<i64> {
    (0..len as i64).map(|i| 3 + (seed * 31 + i * 7 + (i * i) % 11) % (vocab - 5)).collect()
}

/// Serves the prompts together, greedy, `tokens` each, and returns the tokens of every prompt.
fn serve(rt: &mut InferenceRuntime, prompts: &[Vec<i64>], tokens: usize) -> Vec<Vec<i64>> {
    serve_hinted(rt, prompts, tokens, 0)
}

/// `serve` for requests that announce that their first `share` tokens are shared with other requests (a system prompt).
fn serve_hinted(rt: &mut InferenceRuntime, prompts: &[Vec<i64>], tokens: usize, share: usize) -> Vec<Vec<i64>> {
    let params = SamplingParams { ignore_eos: true, share_prefix_tokens: share, ..Default::default() };
    for (i, p) in prompts.iter().enumerate() {
        rt.submit_with_sampling(format!("r{i}"), p.clone(), tokens, params.clone()).unwrap();
    }
    let mut out = vec![Vec::<i64>::new(); prompts.len()];
    let mut guard = 0;
    while out.iter().any(|o| o.len() < tokens) {
        guard += 1;
        assert!(guard < 5_000, "no progress");
        for e in rt.step().unwrap() {
            let i: usize = e.request_id[1..].parse().unwrap();
            if out[i].len() < tokens {
                out[i].push(e.token_id);
            }
        }
    }
    out
}

fn states(rt: &InferenceRuntime) -> (u64, u64, u64, u64) {
    let s = &rt.prefix_cache_report()["state_checkpoints"];
    assert_eq!(s["enabled"], true, "{s}");
    (s["hits"].as_u64().unwrap(), s["tokens_skipped"].as_u64().unwrap(), s["entries"].as_u64().unwrap(), s["evicted"].as_u64().unwrap())
}

fn check(label: &str, gguf: &Path, vocab: i64) {
    let a = prompt(1, 70, vocab);
    let mut shares_start = a[..56].to_vec(); // 3.5 blocks of 16: the start is shared, the rest differs
    shares_start.extend(prompt(5, 14, vocab));
    let other = prompt(2, 70, vocab);

    // what reading every prompt from the start gives (one fresh runtime each: nothing to restore from)
    let fresh = |p: &Vec<i64>| serve(&mut runtime(gguf), std::slice::from_ref(p), 14).remove(0);
    let (want_a, want_shares, want_other) = (fresh(&a), fresh(&shares_start), fresh(&other));
    assert!(want_a.windows(2).any(|w| w[0] != w[1]), "{label}: constant output proves nothing");
    assert_ne!(want_a, want_shares, "{label}: the prompts must lead to different tokens");

    let mut rt = runtime(gguf);
    assert_eq!(serve(&mut rt, std::slice::from_ref(&a), 14).remove(0), want_a, "{label}: first read");
    let (hits, _, entries, _) = states(&rt);
    assert_eq!((hits, entries), (0, 1), "{label}: one checkpoint after the first prompt, nothing restored yet");

    // the same prompt again: the checkpoint at the last full block is restored, only the end is read
    assert_eq!(serve(&mut rt, std::slice::from_ref(&a), 14).remove(0), want_a, "{label}: the same prompt again");
    let (hits, skipped, _, _) = states(&rt);
    assert_eq!(hits, 1, "{label}");
    assert!(skipped >= 48, "{label}: the checkpoint covers the first full blocks of the prompt, not {skipped} tokens");

    // a prompt that shares only a start has no checkpoint at that point (the checkpoint sits at the end of the first prompt): read in full
    assert_eq!(serve(&mut rt, std::slice::from_ref(&shares_start), 14).remove(0), want_shares, "{label}: shared start, different end");
    assert_eq!(states(&rt).0, 1, "{label}: a start that ends before the checkpoint must not restore it");

    // a prompt that extends the first one (a chat turn): restored, and the answer is the one of a full read
    let mut turn = a.clone();
    turn.extend_from_slice(&want_a);
    turn.extend(prompt(9, 12, vocab));
    let want_turn = fresh(&turn);
    assert_eq!(serve(&mut rt, std::slice::from_ref(&turn), 14).remove(0), want_turn, "{label}: a longer prompt that starts with the first");
    assert_eq!(states(&rt).0, 2, "{label}");

    // three at once that all start from the same checkpoint: they share the cached KV block, each needs its own state
    let (before_hits, ..) = states(&rt);
    let got = serve(&mut rt, &[a.clone(), a.clone(), other.clone()], 14);
    assert_eq!((&got[0], &got[1], &got[2]), (&want_a, &want_a, &want_other), "{label}: sequences that restore the same checkpoint at once");
    assert!(states(&rt).0 >= before_hits + 2, "{label}: both copies of the prompt restore it");

    // requests that announce a shared start (a system prompt): a new conversation with another end restores the state at that start
    let mut rt = runtime(gguf);
    assert_eq!(serve_hinted(&mut rt, std::slice::from_ref(&a), 14, 56).remove(0), want_a, "{label}: first prompt with a hint");
    assert_eq!(states(&rt).2, 2, "{label}: a checkpoint at the end of the shared start and one at the end of the prompt");
    assert_eq!(serve_hinted(&mut rt, std::slice::from_ref(&shares_start), 14, 56).remove(0), want_shares, "{label}: shared start restored");
    let (hits, skipped, ..) = states(&rt);
    assert_eq!(hits, 1, "{label}: the second conversation starts from the checkpoint at the shared start");
    assert_eq!(skipped, 48, "{label}: 56 shared tokens are 3 full blocks of 16");

    // and again after all of them are gone: the state of a finished sequence did not stay behind in the shared block
    let mut rt = runtime(gguf);
    serve(&mut rt, std::slice::from_ref(&a), 14);
    serve(&mut rt, &[a.clone(), a.clone(), other.clone()], 14);
    assert_eq!(serve(&mut rt, std::slice::from_ref(&a), 14).remove(0), want_a, "{label}: after the concurrent ones");
}

fn tiny(label: &str, arch: Arch) -> PathBuf {
    build_gguf(&scratch(label), arch, Encoding::F32, 0x1234_5678_9abc_def1)
}

fn from_specs(label: &str, config: serde_json::Value, specs: &[crate::hybrid_tests::Spec], seed: u64) -> PathBuf {
    let dir = hybrid_scratch(label);
    write_checkpoint(&dir, config, specs, seed);
    let gguf = dir.join("model.gguf");
    convert_checkpoint(&dir, &gguf, ConvertOptions { output_dtype: OutputDType::F32, overwrite: false, spec_head: None })
        .unwrap_or_else(|e| panic!("conversion failed: {e}"));
    gguf
}

#[test]
fn qwen35_deltanet_attention_hybrid_restores_a_prompt_start() {
    check("qwen35", &tiny("ckpt-q35", Arch::Qwen35), 100);
}

#[test]
fn rwkv7_restores_a_prompt_start() {
    check("rwkv7", &tiny("ckpt-rwkv7", Arch::Rwkv7), 100);
}

#[test]
fn rwkv6_restores_a_prompt_start() {
    check("rwkv6", &tiny("ckpt-rwkv6", Arch::Rwkv6), 100);
}

#[test]
fn mamba2_attention_hybrid_restores_a_prompt_start() {
    let (config, specs) = falcon_h1_specs(true);
    check("falcon-h1", &from_specs("ckpt-fh1", config, &specs, 0x5eed_0201), 40);
}

#[test]
fn mamba1_moe_hybrid_restores_a_prompt_start() {
    let (config, specs) = jamba_specs();
    check("jamba", &from_specs("ckpt-jamba", config, &specs, 0x5eed_0202), 40);
}

#[test]
fn zamba2_restores_a_prompt_start() {
    let (config, specs) = zamba2_specs();
    check("zamba2", &from_specs("ckpt-zamba2", config, &specs, 0x5eed_0203), 40);
}

#[test]
fn clearing_the_prefix_cache_forgets_the_checkpoints_and_the_next_prompt_reads_everything() {
    let gguf = tiny("ckpt-clear", Arch::Qwen35);
    let a = prompt(1, 70, 100);
    let want = serve(&mut runtime(&gguf), std::slice::from_ref(&a), 10).remove(0);
    let mut rt = runtime(&gguf);
    serve(&mut rt, std::slice::from_ref(&a), 10);
    assert_eq!(states(&rt).2, 1);
    rt.clear_prefix_cache().unwrap();
    assert_eq!(states(&rt).2, 0);
    assert_eq!(serve(&mut rt, std::slice::from_ref(&a), 10).remove(0), want);
    assert_eq!(states(&rt).0, 0, "nothing was left to restore");
}
