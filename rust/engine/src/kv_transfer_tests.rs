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
