# KV tiering: memory and disk, writes and reads both work

`mem/kv_tier.rs` exists and is tested: one file per block, named after the block's chain hash, `FLLM_KV_DISK_DIR` /
`FLLM_KV_DISK_BYTES`, an LRU budget, atomic writes (temporary name + rename, so a crash cannot leave a half block under a
valid name), and a header check that throws away foreign or truncated files instead of reading garbage as KV.

**The write path is wired.** When the prefix cache registers a block (`Scheduler::insert_full_blocks`), the scheduler
queues `(chain_hash, block)` in `pending_spills`; the runtime drains that queue at the start of `step()` and writes the KV
with `kv_export_block`. The block is alive at that moment — the cache holds a reference — so its KV is exactly what its hash
claims. Live check with `FLLM_KV_DISK_DIR=/tmp/kvdisk`: one chat request produced three files of 4120 bytes (24-byte header
plus a 4096-byte block).

Write-through was chosen over write-on-evict for exactly one reason: at eviction the block is already on the free list, and
a block that can be reused at any moment must not be read.

The file carries what a rebuild needs: `FLLMKV02 | payload_len | hash | parent_hash | token_count | tokens… | KV`. Live
output from one request, four blocks in one chain:

```
FLLMKV02 block=fcd9942a8e0fa217 parent=0000000000000000 tokens=16 payload=4096
FLLMKV02 block=3c8f3fefe704c31e parent=fcd9942a8e0fa217 tokens=16 payload=4096
FLLMKV02 block=0548d93b79d17bb0 parent=3c8f3fefe704c31e tokens=16 payload=4096
FLLMKV02 block=ee0a3bf3d74020fc parent=0548d93b79d17bb0 tokens=16 payload=4096
```

**Reading back works.** `Runtime::warm_prefix_cache` runs once, on the first step: it groups the stored blocks by parent,
walks each chain from its root, imports the KV into freshly allocated blocks and registers the whole path with
`PrefixCache::insert_path` — a chain, not single blocks side by side, because the radix tree matches along a path.

Live proof across a restart, same prompt, 76 prompt tokens:

```
INFO runtime: warmed 4 prefix blocks from the disk tier
cached: 64 / 76
```

64 tokens (four blocks) came out of the disk cache on the first request of a fresh process.

**And the imported KV is correct.** The strongest check available without a real model is that the same prompt produces the
same tokens before and after a restart, because a wrong block would show up as different text:

```
run 1 (cold, computes the prompt):   [Ewt=m\no%.=m
run 2 (restart, warmed 2 blocks):    [Ewt=m\no%.=m      cached: 32 / 47
```

Identical, with a third of the prompt taken from disk. What is still open is the
management side: there is no eviction policy for the disk tier beyond the byte budget, and no `DELETE` for it. What is missing is a bridge, and here is exactly
where it has to go — checked in the code, not guessed.

## Why a bridge and not a call

- Blocks are freed in `Scheduler::evict_cache`: it has the block id, and `bm.block_hashes[block]` is the chain hash. That is
  the one place where "this block will never be needed from RAM again" is known.
- Exporting the KV needs `Runtime::kv_export_block(block) -> Vec<u8>` and importing needs `kv_import_block(block, data)` —
  both live on the decoder, which the scheduler does not own.
- The prefix lookup (`PrefixCache::match_blocks` inside `Scheduler::try_admit`) is where a disk hit would have to be turned
  into a claimed block, and it runs in the same direction: the scheduler cannot reach the decoder.

So the two halves have to talk through something. Two ways, both small:

1. **Queue, one direction each.** The scheduler pushes `(hash, block)` into `pending_spills` when it evicts; the runtime
   drains that queue right after the scheduling call and spills the KV while the block is still untouched. For the read
   side the runtime asks first: `tier.has(hash)` for the hashes it is about to admit. This keeps the ownership as it is and
   needs no trait object.
2. **A trait the runtime implements** (`has`, `load`, `spill`) that the scheduler holds as `Option<Arc<dyn KvDisk>>`. Cleaner
   call sites, but the runtime would have to hand a handle of itself to the scheduler it owns.

Option 1 is the smaller change: one `Vec<(u64, i64)>` in `Scheduler`, one drain in `Runtime`, and `match_blocks` gets an
optional `import` callback that returns a freshly imported block id for a hash.

## The trap worth naming

Spilling has to happen **before the block is reused**, not after the batch that needs the space. A block that is on the free
list is still intact — but only until the next allocation writes over it. That is why the drain has to sit right after the
scheduling call and not, say, at the end of the request.

## Correctness rules that are enforced

- A block is written while the cache holds a reference to it, never after it was freed.
- A chain is only rebuilt when its root is on disk: without the root the hash chain cannot be verified, so a middle-of-chain
  block is useless and is left alone.
- If the KV import or the registration fails, the block goes straight back to the allocator.

## What is proven today

Seven tests in `mem/kv_tier.rs`: roundtrip, miss without a file, foreign file ignored and removed, truncated payload
refused, LRU budget drops the oldest, a block larger than the budget is not written, and "no directory means no tier".

`runtime/kv_transfer_tests.rs` adds the end-to-end check on the tiny model: a restarted runtime reads every block of the chain
back, serves the first request from them (a prefix-cache hit) and gives exactly the tokens of a runtime without any tier; a
runtime without a tier has no hit; a block file shortened by five bytes is not read as KV and the output stays the same. With the
read-back switched off the test fails on "every block of the chain is read back".

**On a real model** (Qwen2.5-0.5B Q8_0, CPU, one process killed and a second one started on the same directory, output cache off):

```
process 1 (cold):     7.7 s, prompt 1414 tokens, cached 0,    88 files on disk
process 2 (restart):  0.6 s, prompt 1414 tokens, cached 1408
text equal: True      "warmed 88 prefix blocks from the disk tier"
```

13 times faster for the first request after a restart, and the same text.

Nothing in this document needs a GPU.

## Reading back while the server runs

A restart is not the only time the cache has lost a block: when the pool is full the prefix cache drops its oldest leaves, and a long
conversation or a busy server drops prefixes it will be asked for again. The tier writes every block when it is registered, so a dropped
block is still on disk as long as the byte budget keeps it. `Runtime::reload_for_waiting` runs before every scheduling round: for each
request that waits it walks the chain of the prompt as `try_admit` looks it up, finds the first block the cache does not hold, and reads
the following ones from the tier (`KvTier::load_for`, which also checks the tokens the file carries against the prompt, so a file under the
right hash with other tokens is not taken), imports them and registers the path. The request is then admitted with the whole prefix cached.

Limits, on purpose: at most 256 blocks a round; the request keeps one free block (plus the watermark) for what it computes. The pool of a
busy server is full of cached blocks nobody uses, so the free blocks alone allow almost nothing (the first real run read back 6 of 35
blocks); the reload therefore drops the oldest unused cache blocks to make room (`Scheduler::reclaim_cache`, what admitting the request
would do anyway), but only when the disk really has the blocks (`KvTier::has`), so a prompt the server has never seen costs the cache
nothing. Blocks that a running sequence uses cannot be evicted. Only chains without a `cache_salt` (a salted block is
not written, since a restarted server cannot know the salt); media prompts are skipped. The read runs on the worker thread between two
rounds, so a slow disk delays the round; the cost is a file read per block against a prefill of the same block.

Tests (`kv_transfer_tests.rs`): a prefix the cache dropped under pressure is read back and the output equals a full prefill; a file with
other tokens under the right hash is not taken; with two free blocks only one is taken; a full cache makes room; a prompt the disk does not
have drops nothing. `reloaded_blocks` in `GET /admin/kv-tier` (the tier statistics) counts what was taken.

Measured on a real model (Qwen2.5-0.5B Q8_0, CPU, `--ctx 2048 --max-seqs 2`, a document of 568 prompt tokens, then twelve other documents
of the same size; the cache then held 246 blocks and had dropped the first one; `scripts/kv-reload.py`): asked again, the server with a
disk tier answered in 0.44 s with 560 of 568 prompt tokens from the cache (35 blocks read back, 0 misses, same text); the control without a
tier needed 4.18 s with 0 cached (the cold request took 3.6-3.7 s). The disk was this host's SSD with the files in the page cache, so a
slower disk reads more slowly. Not measured: a busy server with many requests at once, and a GPU (the KV import then copies to the device).

## A memory tier in front of the disk

`FLLM_KV_RAM_BYTES` (Settings > Engine: *Keep the most recent ones in memory*) adds host memory in front of the disk, inside the same
`KvTier`, so the restart read, the reload while running, the statistics and the Studio buttons all see both. Every block that is written
lands in memory and, when a folder is set, on disk. The least recently used block leaves memory when the budget is full; reading a block
makes it the most recent one. A read looks in memory first, then on disk, and a block that came from the disk is put into memory again.
Either tier can be off: with a budget and no folder the tier works as a cache in memory only (nothing survives a restart), with a folder and
no budget it is the disk tier of before. A block with other tokens under the same hash is refused in memory as on disk, a block larger than
the whole budget is not kept, and `DELETE /admin/kv-tier` empties both. `GET /admin/kv-tier` adds `ram_max_bytes`, `ram_used_bytes`,
`ram_blocks` and `ram_hits`.

Why it matters on a server: host memory is read in microseconds where a disk takes milliseconds, and an 8-GPU server has hundreds of GB of it
next to a few hundred GB of GPU memory per card, so a prefix the GPU pool had to drop is a memory read away instead of a recompute.

Tests: memory alone serves and keeps the place in the chain, the budget drops the least recently used block (reading renews a block), memory
is read before disk and a disk block returns into memory, clearing empties both, a tier with neither is off; at the runtime level a dropped
prefix comes back from memory with no disk at all, a memory tier too small to hold the chain leaves the disk to do it, and salted requests
stay out of memory too. Six mutations (reading does not renew, a disk block does not return, the token check missing, the budget not
enforced, clear leaving memory, `has` blind to memory) each fail a test.

Measured on a real model (`scripts/kv-ram.py`, the setup of the disk run above, memory tier only, no folder): the dropped 568-token prompt came
back in 0.66 s with 560 tokens cached (35 blocks read back, same text); the control without a tier needed 3.90 s with 0 cached. The disk run
took 0.44 s because its files were in the page cache of this host; the difference between the two figures is noise of this machine, not an
advantage of the disk. Not measured: the copy between GPU memory and host memory (this host has no GPU; the import goes through the same
`kv_import_block` the disk read uses) and a busy server.
