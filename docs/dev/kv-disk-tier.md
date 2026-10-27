# KV disk tiering: writes and reads both work

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

Nothing in this document needs a GPU.
