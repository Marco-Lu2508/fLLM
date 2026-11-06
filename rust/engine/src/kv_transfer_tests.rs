//! Disaggregated prefill / decode on the tiny random model (CPU): the KV blocks of a prompt prefilled by one runtime are exported, sent through
//! the wire format, imported by a second runtime, and that runtime then generates from them exactly as if it had prefilled the prompt itself.
//! This module is a child of `runtime`, so it can look at the scheduler's counters and the allocator.

use super::*;
use crate::kv_transfer::{DirTransport, KvChain, KvTransport};
use crate::sampling::SamplingParams;
use std::path::PathBuf;
use std::sync::OnceLock;

fn gguf() -> PathBuf {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("fllm-kvt-tests-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.gguf");
        crate::tiny_model::write_gguf(&path, &crate::tiny_model::Options::default()).unwrap();
        path
    })
    .clone()
}

fn runtime() -> InferenceRuntime {
    let model = GgufModel::load(gguf().to_str().unwrap(), 128, 8, 2).unwrap();
    let mut rt = InferenceRuntime::with_speculative(model, 8, 128, Vec::new(), 0, 8).unwrap();
    rt.attach_compute("cpu").unwrap();
    rt
}

fn prompt(n: usize) -> Vec<i64> {
    (0..n as i64).map(|i| 10 + (i * 7) % 90).collect()
}

fn generate(rt: &mut InferenceRuntime, id: &str, prompt: &[i64], max_tokens: usize) -> Vec<i64> {
    if let Err(e) = rt.submit_with_sampling(id.into(), prompt.to_vec(), max_tokens, SamplingParams::default()) {
        panic!("submit failed for {} tokens: {e}", prompt.len());
    }
    let mut tokens = Vec::new();
    let mut guard = 0;
    while rt.active_requests() > 0 {
        guard += 1;
        assert!(guard < 20_000, "no progress");
        for event in rt.step().unwrap_or_else(|e| panic!("step failed: {e}")) {
            tokens.push(event.token_id);
        }
    }
    tokens
}

fn free_after_clearing_cache(rt: &mut InferenceRuntime) -> usize {
    rt.clear_prefix_cache().unwrap();
    rt.available_blocks()
}

#[test]
fn a_decode_runtime_generates_from_imported_blocks_like_a_runtime_that_prefilled() {
    let long = prompt(60); // several blocks, below the model's context of 128
    let mut prefill = runtime();
    let reference = generate(&mut prefill, "ref", &long, 8);
    assert_eq!(reference.len(), 8);

    // the prefill side exports what its prefix cache holds for this prompt
    let block = prefill.scheduler.cfg.block_size;
    let chain = prefill.kv_export_prompt(&long).unwrap();
    assert_eq!(chain.blocks.len(), long.len() / block, "every full block of the prompt is cached");
    assert!(chain.blocks.iter().all(|b| b.tokens.len() == block && !b.bytes.is_empty()));

    // through the wire format and a directory hand-over
    let dir = std::env::temp_dir().join(format!("fllm-kvt-inbox-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    DirTransport.send(dir.to_str().unwrap(), &chain.encode()).unwrap();
    let received = DirTransport::drain(&dir).unwrap();
    assert_eq!(received.len(), 1);
    let _ = std::fs::remove_dir_all(&dir);
    let received = KvChain::decode(&received[0]).unwrap();
    assert_eq!(received, chain);

    // the decode side imports it and generates
    let mut decode = runtime();
    let before = decode.available_blocks();
    let (imported, present) = decode.kv_import_chain(&received).unwrap();
    assert_eq!((imported, present), (chain.blocks.len(), 0));
    assert_eq!(decode.available_blocks(), before - imported, "the cache holds exactly one reference per imported block");
    let hits_before = decode.scheduler.prefix_cache_hits;
    let tokens = generate(&mut decode, "dec", &long, 8);
    assert_eq!(tokens, reference, "generation from imported KV differs from generation after a local prefill");
    assert_eq!(decode.scheduler.prefix_cache_hits, hits_before + 1, "the request did not use the imported blocks");

    // a control: a runtime that never saw the chain computes the same tokens without a cache hit
    let mut control = runtime();
    assert_eq!(generate(&mut control, "ctl", &long, 8), reference);
    assert_eq!(control.scheduler.prefix_cache_hits, 0);

    // importing the same chain again changes nothing
    let again = decode.kv_import_chain(&received).unwrap();
    assert_eq!(again, (0, chain.blocks.len()));

    // no block leaks: once the caches are emptied both runtimes have all their blocks back
    assert_eq!(free_after_clearing_cache(&mut decode), free_after_clearing_cache(&mut control));
    assert_eq!(free_after_clearing_cache(&mut prefill), free_after_clearing_cache(&mut control));
}

#[test]
fn a_shared_prefix_is_extended_not_duplicated() {
    let mut prefill = runtime();
    let block = prefill.scheduler.cfg.block_size;
    let long = prompt(((64 / block).clamp(2, 6)) * block + 1);
    generate(&mut prefill, "a", &long, 2);
    let chain = prefill.kv_export_prompt(&long).unwrap();
    assert!(chain.blocks.len() >= 2, "{} blocks of {block}", chain.blocks.len());
    let known = chain.blocks.len() - 1;
    let head = KvChain { blocks: chain.blocks[..known].to_vec(), ..chain.clone() };
    let mut decode = runtime();
    assert_eq!(decode.kv_import_chain(&head).unwrap(), (known, 0));
    // the full chain: the first blocks are known, the last one is new
    assert_eq!(decode.kv_import_chain(&chain).unwrap(), (1, known));
}

#[test]
fn a_chain_from_a_different_cache_layout_is_refused() {
    let long = prompt(32);
    let mut prefill = runtime();
    generate(&mut prefill, "a", &long, 2);
    let chain = prefill.kv_export_prompt(&long).unwrap();
    let mut decode = runtime();
    let mut other = chain.clone();
    other.fingerprint.push_str("-other");
    assert!(decode.kv_import_chain(&other).unwrap_err().contains("different cache layout"));
    let mut wrong_block = chain.clone();
    wrong_block.block_size += 1;
    assert!(decode.kv_import_chain(&wrong_block).is_err());
    let mut salted = chain.clone();
    salted.salt = 5;
    assert!(decode.kv_import_chain(&salted).unwrap_err().contains("default prefix identity"));
    let mut torn = chain;
    torn.blocks[0].tokens.pop();
    assert!(decode.kv_import_chain(&torn).is_err());
    // nothing was registered by the refused chains
    assert_eq!(decode.scheduler.cached_prompt_chain(&long).iter().filter(|c| c.2.is_some()).count(), 0);
}

#[test]
fn nothing_is_exported_for_a_prompt_that_was_not_prefilled() {
    let mut rt = runtime();
    let chain = rt.kv_export_prompt(&prompt(40)).unwrap();
    assert!(chain.blocks.is_empty());
}

/// A runtime whose disk tier is the given directory (the tier is read from the environment at construction; a test must not change the
/// environment of the process, other tests run beside it).
fn runtime_with_disk_tier(dir: &std::path::Path) -> InferenceRuntime {
    let mut rt = runtime();
    let tier = std::sync::Arc::new(crate::mem::kv_tier::KvTier::at(dir, 64 << 20));
    rt.spill_tx = spawn_spill_writer(std::sync::Arc::clone(&tier));
    rt.kv_tier = tier;
    rt
}

/// The disk tier across a restart: a second runtime that finds the blocks of the first one on disk must serve the shared prefix from
/// them (a prefix-cache hit on the first request) and give exactly the tokens a runtime without any tier computes. A wrong block would
/// show up as different tokens. The same thing was measured on a real model: 1408 of 1414 prompt tokens from disk, 7.7 s -> 0.6 s.
#[test]
fn a_restarted_runtime_serves_the_prefix_from_the_disk_tier_and_gives_the_same_tokens() {
    let long = prompt(60);
    let dir = std::env::temp_dir().join(format!("fllm-kvt-disk-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let mut first = runtime_with_disk_tier(&dir);
    let block = first.scheduler.cfg.block_size;
    let reference = generate(&mut first, "one", &long, 8);
    // the spill writer is a thread behind a queue: wait for the blocks of the prompt and of the answer (a block that fills up while
    // the answer is generated is registered too, so that the next turn of a conversation finds it)
    let want = (long.len() + reference.len() - 1) / block;
    for _ in 0..200 {
        first.spill_pending();
        if first.kv_tier.entries().len() >= want {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    assert!(first.kv_tier.entries().len() >= want, "{} of {want} blocks reached the disk", first.kv_tier.entries().len());
    drop(first);

    let mut second = runtime_with_disk_tier(&dir);
    assert_eq!(second.warm_prefix_cache(128).unwrap(), want, "every block of the chain is read back");
    second.kv_warmed = true;
    let hits = second.scheduler.prefix_cache_hits;
    let tokens = generate(&mut second, "two", &long, 8);
    assert_eq!(tokens, reference, "generation from blocks read back from disk differs from a local prefill");
    assert_eq!(second.scheduler.prefix_cache_hits, hits + 1, "the request did not use the blocks read back from disk");

    // a control without a tier computes the same tokens and has no hit
    let mut control = runtime();
    assert_eq!(generate(&mut control, "ctl", &long, 8), reference);
    assert_eq!(control.scheduler.prefix_cache_hits, 0);

    // a block whose file was damaged is not read back as KV: the chain stops before it and the request still gives the right tokens
    let damaged = {
        let mut third = runtime_with_disk_tier(&dir);
        let entries = third.kv_tier.entries();
        // the second block of the chain is the one whose parent is the root block
        let root = entries.iter().find(|e| e.parent == 0).expect("the root block").hash;
        let victim = entries.iter().find(|e| e.parent == root).expect("a second block").hash;
        let path = dir.join(format!("{victim:016x}.kv"));
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.truncate(bytes.len() - 5);
        std::fs::write(&path, bytes).unwrap();
        third.warm_prefix_cache(128).unwrap();
        third.kv_warmed = true;
        assert_eq!(generate(&mut third, "three", &long, 8), reference, "a truncated block changed the output");
        third
    };
    drop(damaged);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A chat client sends the whole conversation again every turn, so the next prompt starts with the previous prompt *and the answer the
/// model gave to it*. The blocks that fill up while the answer is generated must therefore be registered in the prefix cache too, not
/// only the blocks of the prompt: otherwise every turn recomputes the previous answer. (The decode step used to commit its token
/// without registering a block that had just become full; only prefill chunks and verified drafts did.)
#[test]
fn the_blocks_filled_while_generating_serve_the_next_turn() {
    let mut rt = runtime();
    let block = rt.scheduler.cfg.block_size;
    let first = prompt(2 * block + 3);
    let answer = generate(&mut rt, "turn1", &first, 5 * block);
    assert_eq!(answer.len(), 5 * block);
    // the next turn: the previous prompt, the answer, and a new question
    let mut second = first.clone();
    second.extend_from_slice(&answer);
    second.extend_from_slice(&[11, 12, 13]);
    let hits_before = rt.scheduler.prefix_cache_hits;
    rt.submit_with_sampling("turn2".into(), second.clone(), 2, SamplingParams::default()).unwrap();
    let mut cached = 0;
    while rt.active_requests() > 0 {
        for event in rt.step().unwrap() {
            cached = cached.max(event.cached_prompt_tokens);
        }
    }
    assert_eq!(rt.scheduler.prefix_cache_hits, hits_before + 1);
    // all full blocks of prompt and answer may be reused, but the last token of the prompt is always computed
    let reusable = (first.len() + answer.len()) / block * block;
    assert_eq!(cached, reusable, "the second turn reused {cached} tokens of {reusable} that were computed in the first");
}
