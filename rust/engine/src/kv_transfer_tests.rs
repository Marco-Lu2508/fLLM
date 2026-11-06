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

/// Runs one request under `salt` and returns how many prompt tokens came from the cache.
fn cached_under_salt(rt: &mut InferenceRuntime, id: &str, prompt: &[i64], salt: u64) -> usize {
    rt.submit_with_sampling(id.into(), prompt.to_vec(), 2, SamplingParams::default()).unwrap();
    rt.set_request_prefix_salt(id, salt);
    let mut cached = 0;
    while rt.active_requests() > 0 {
        for event in rt.step().unwrap() {
            cached = cached.max(event.cached_prompt_tokens);
        }
    }
    cached
}

/// A request with a `cache_salt` must find the blocks that an earlier request with the same salt left, and nobody else's. The chain of
/// hashes starts at `seed ^ salt` when a prompt is looked up; it has to start there when the blocks are registered, too.
#[test]
fn a_salted_request_reuses_the_blocks_of_its_own_salt_and_of_nobody_else() {
    let mut rt = runtime();
    let block = rt.scheduler.cfg.block_size;
    let long = prompt(4 * block + 3);
    assert_eq!(cached_under_salt(&mut rt, "a1", &long, 77), 0, "nothing is cached yet");
    assert_eq!(cached_under_salt(&mut rt, "a2", &long, 77), 4 * block, "the same salt reuses its own blocks");
    assert_eq!(cached_under_salt(&mut rt, "b1", &long, 78), 0, "another salt must not see them");
    assert_eq!(cached_under_salt(&mut rt, "n1", &long, 0), 0, "nor the requests without a salt");
    assert_eq!(cached_under_salt(&mut rt, "n2", &long, 0), 4 * block, "and those reuse their own");
}

/// The other direction of the isolation: what a salted request left behind must not be visible to a request without a salt either.
#[test]
fn what_a_salted_request_left_is_not_served_to_an_unsalted_one() {
    let mut rt = runtime();
    let block = rt.scheduler.cfg.block_size;
    let long = prompt(4 * block + 3);
    assert_eq!(cached_under_salt(&mut rt, "s1", &long, 77), 0);
    assert_eq!(cached_under_salt(&mut rt, "u1", &long, 0), 0, "an unsalted request was served the blocks of a salted one");
}

/// The disk tier is read back by a server that does not know the salt of a request, so the blocks of a salted request stay out of it
/// (they would be written under a chain that no unsalted lookup can ever reach, and take room from blocks that can be read back).
#[test]
fn the_blocks_of_a_salted_request_do_not_go_to_the_disk_tier() {
    let dir = std::env::temp_dir().join(format!("fllm-kvt-salt-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut rt = runtime_with_disk_tier(&dir);
    let block = rt.scheduler.cfg.block_size;
    let long = prompt(3 * block + 3);
    assert_eq!(cached_under_salt(&mut rt, "salted", &long, 5), 0);
    for _ in 0..40 {
        rt.spill_pending();
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(rt.kv_tier.entries().is_empty(), "{} blocks of a salted request reached the disk", rt.kv_tier.entries().len());
    // the control: the same prompt without a salt is written
    let mut other = prompt(3 * block + 3);
    other[0] += 1;
    generate(&mut rt, "plain", &other, 2);
    for _ in 0..200 {
        rt.spill_pending();
        if !rt.kv_tier.entries().is_empty() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    assert!(!rt.kv_tier.entries().is_empty(), "the control wrote nothing, so the test above proves nothing");
    drop(rt);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Fills the prefix cache with other prompts until the first block of `first` has been dropped (the pool is small; the oldest leaves go
/// first, so when the first block is gone the whole chain is).
fn push_out_of_the_cache(rt: &mut InferenceRuntime, first: u64, length: usize) {
    let mut other = 0usize;
    while rt.scheduler.prefix_block_of(first).is_some() {
        other += 1;
        assert!(other < 400, "the cache never dropped the prompt; the test cannot say anything");
        let mut p = prompt(length);
        p[0] = 10 + (other % 90) as i64;
        p[1] = 10 + (other / 90 % 90) as i64;
        generate(rt, &format!("push{other}"), &p, 2);
    }
}

fn wait_for_the_disk(rt: &mut InferenceRuntime, blocks: usize) {
    for _ in 0..400 {
        rt.spill_pending();
        if rt.kv_tier.entries().len() >= blocks {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    panic!("{} of {blocks} blocks reached the disk", rt.kv_tier.entries().len());
}

fn ask(rt: &mut InferenceRuntime, id: &str, prompt: &[i64], max_tokens: usize) -> (Vec<i64>, usize) {
    rt.submit_with_sampling(id.into(), prompt.to_vec(), max_tokens, SamplingParams::default()).unwrap();
    let (mut tokens, mut cached) = (Vec::new(), 0);
    while rt.active_requests() > 0 {
        for event in rt.step().unwrap() {
            tokens.push(event.token_id);
            cached = cached.max(event.cached_prompt_tokens);
        }
    }
    (tokens, cached)
}

/// The cache drops blocks when the pool is full; the disk tier still has them. A request for such a prompt gets them back from disk
/// before it is admitted (no restart involved), and generates what a full prefill generates.
#[test]
fn a_prefix_the_cache_dropped_under_pressure_is_read_back_from_disk() {
    let dir = std::env::temp_dir().join(format!("fllm-kvt-reload-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut rt = runtime_with_disk_tier(&dir);
    rt.kv_warmed = true; // not the restart path: the server keeps running
    let block = rt.scheduler.cfg.block_size;
    let a = prompt(4 * block + 3);
    let reference = generate(&mut rt, "a", &a, 4);
    let first = rt.scheduler.cached_prompt_chain(&a)[0].0;
    wait_for_the_disk(&mut rt, 4);
    push_out_of_the_cache(&mut rt, first, 4 * block + 3);
    assert!(rt.scheduler.prefix_block_of(first).is_none());
    let before = rt.kv_reloaded;
    let (tokens, cached) = ask(&mut rt, "again", &a, 4);
    assert_eq!(tokens, reference, "generation from blocks read back from disk differs from a full prefill");
    assert_eq!(cached, 4 * block, "the dropped blocks were not read back");
    assert_eq!(rt.kv_reloaded - before, 4);
    assert_eq!(rt.kv_tier_stats()["reloaded_blocks"], rt.kv_reloaded);
    drop(rt);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A file under the right hash that carries other tokens is not the block that is meant: nothing is read back, the request is computed.
#[test]
fn a_block_file_with_other_tokens_is_not_read_back() {
    let dir = std::env::temp_dir().join(format!("fllm-kvt-reload-bad-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut rt = runtime_with_disk_tier(&dir);
    rt.kv_warmed = true;
    let block = rt.scheduler.cfg.block_size;
    let a = prompt(4 * block + 3);
    let reference = generate(&mut rt, "a", &a, 4);
    let first = rt.scheduler.cached_prompt_chain(&a)[0].0;
    wait_for_the_disk(&mut rt, 4);
    push_out_of_the_cache(&mut rt, first, 4 * block + 3);
    // the right KV under the right hash, with the wrong tokens
    let stored = rt.kv_tier.peek(first).expect("the first block on disk");
    let payload = rt.kv_tier.load(first).unwrap();
    let mut wrong = stored.tokens.clone();
    wrong[0] += 1;
    assert!(rt.kv_tier.spill(first, stored.parent, &wrong, &payload));
    let (tokens, cached) = ask(&mut rt, "again", &a, 4);
    assert_eq!(tokens, reference);
    assert_eq!(cached, 0, "a block file with other tokens was taken");
    drop(rt);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Reading blocks back must not use up the free pool: the request that waits needs a block of its own for the part it computes, and
/// blocks that a running sequence needs cannot be evicted. With two free blocks left only one is taken, whatever the chain is long.
#[test]
fn reading_blocks_back_leaves_room_for_the_request_that_waits() {
    let dir = std::env::temp_dir().join(format!("fllm-kvt-reload-room-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut rt = runtime_with_disk_tier(&dir);
    rt.kv_warmed = true;
    let block = rt.scheduler.cfg.block_size;
    let a = prompt(4 * block + 3);
    generate(&mut rt, "a", &a, 4);
    let first = rt.scheduler.cached_prompt_chain(&a)[0].0;
    wait_for_the_disk(&mut rt, 4);
    push_out_of_the_cache(&mut rt, first, 4 * block + 3);
    // nothing evictable: the cache is empty, so only the free blocks count (the disk still has the chain)
    rt.clear_prefix_cache().unwrap();
    // squeeze the pool: hold every free block but two
    let mut held = Vec::new();
    while rt.available_blocks() > 2 {
        held.push(rt.scheduler.alloc_free_block().unwrap());
    }
    rt.submit_with_sampling("waits".into(), a.clone(), 2, SamplingParams::default()).unwrap();
    let before = rt.kv_reloaded;
    rt.reload_for_waiting();
    assert_eq!(rt.kv_reloaded - before, 1, "two free blocks: one may be taken, one is left for the request");
    assert_eq!(rt.available_blocks(), 1, "the pool was used up");
    for block in held {
        rt.scheduler.free_block(block);
    }
    while rt.active_requests() > 0 {
        rt.step().unwrap();
    }
    drop(rt);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The pool of a busy server is full of cached blocks nobody uses; the few free blocks alone would allow almost no reload (measured on a
/// real model: 6 of 35 blocks). The reload makes room by dropping the oldest unused cache blocks, and keeps one block free.
#[test]
fn a_full_cache_makes_room_for_the_blocks_that_are_read_back() {
    let dir = std::env::temp_dir().join(format!("fllm-kvt-reload-full-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut rt = runtime_with_disk_tier(&dir);
    rt.kv_warmed = true;
    let block = rt.scheduler.cfg.block_size;
    let a = prompt(4 * block + 3);
    let reference = generate(&mut rt, "a", &a, 4);
    let first = rt.scheduler.cached_prompt_chain(&a)[0].0;
    wait_for_the_disk(&mut rt, 4);
    push_out_of_the_cache(&mut rt, first, 4 * block + 3);
    // the cache holds blocks of other prompts that nobody uses; leave only one block free
    let mut held = Vec::new();
    while rt.available_blocks() > 1 {
        held.push(rt.scheduler.alloc_free_block().unwrap());
    }
    assert!(rt.scheduler.pc.as_ref().unwrap().len() >= 4, "the cache must have something to give up");
    rt.submit_with_sampling("again".into(), a.clone(), 4, SamplingParams::default()).unwrap();
    let before = rt.kv_reloaded;
    rt.reload_for_waiting();
    assert_eq!(rt.kv_reloaded - before, 4, "the full cache did not make room");
    assert!(rt.available_blocks() >= 1, "the reserve was used up");
    for block in held {
        rt.scheduler.free_block(block);
    }
    let mut tokens = Vec::new();
    let mut cached = 0;
    while rt.active_requests() > 0 {
        for event in rt.step().unwrap() {
            tokens.push(event.token_id);
            cached = cached.max(event.cached_prompt_tokens);
        }
    }
    assert_eq!(tokens, reference);
    assert_eq!(cached, 4 * block);
    drop(rt);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A prompt the disk has never seen must not cost the cache anything: no block is dropped to make room for blocks that do not exist.
#[test]
fn a_prompt_the_disk_does_not_have_does_not_empty_the_cache() {
    let dir = std::env::temp_dir().join(format!("fllm-kvt-reload-new-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut rt = runtime_with_disk_tier(&dir);
    rt.kv_warmed = true;
    let block = rt.scheduler.cfg.block_size;
    for k in 0..6usize {
        let mut p = prompt(4 * block + 3);
        p[0] = 10 + k as i64;
        generate(&mut rt, &format!("fill{k}"), &p, 2);
    }
    let mut held = Vec::new();
    while rt.available_blocks() > 1 {
        held.push(rt.scheduler.alloc_free_block().unwrap());
    }
    let cached_before = rt.scheduler.pc.as_ref().unwrap().len();
    let mut new = prompt(4 * block + 3);
    new[0] = 99;
    new[1] = 98;
    rt.submit_with_sampling("new".into(), new, 2, SamplingParams::default()).unwrap();
    rt.reload_for_waiting();
    assert_eq!(rt.kv_reloaded, 0);
    assert_eq!(rt.scheduler.pc.as_ref().unwrap().len(), cached_before, "blocks were dropped for a prompt the disk does not have");
    for block in held {
        rt.scheduler.free_block(block);
    }
    drop(rt);
    let _ = std::fs::remove_dir_all(&dir);
}
