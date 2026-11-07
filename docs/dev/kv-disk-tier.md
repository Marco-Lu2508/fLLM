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

## Looking ahead: the prefetch thread

With a memory tier and a folder, each scheduling round looks at the requests that wait: the blocks of their prompts that the cache dropped
and the disk has are queued, and a thread of its own (`fllm-kv-prefetch`) moves them from the disk into memory while the worker computes.
When the request is admitted, its reload finds the blocks in memory and does not wait for the disk. The prefetch does not count as a hit
(`prefetched_blocks` and `prefetch_pending` in `GET /admin/kv-tier` count it), skips blocks that are in memory already, never touches
requests with a `cache_salt` or with images, and does nothing without both tiers. It is the part of SGLang's HiCache (prefetch from the
lower tier while the request waits) that does not need a GPU. What is not built: loading layer by layer in parallel with the computation
(the copy from host memory to the GPU overlaps with layer N while layer N+1 loads), which needs a GPU to be written or measured.

## Which model a block belongs to

A block on disk is named after the hash of its tokens, and that hash knows nothing about the weights: the KV values of the same tokens are
different for a Q8_0 and a Q4_K_M file of one model, for a model with an adapter, and for another KV cache type. The tier therefore keeps
one subfolder per identity below `FLLM_KV_DISK_DIR` (the name is a hash of the model file digest, the adapters with their scales, and the KV
geometry: architecture, layers, heads, head and value size, block size, key and value type). Servers with the same identity share their blocks
(a restart finds them); others cannot read, list or delete each other's files, and the byte budget counts the subfolder of the server. The
memory tier is per process and needs no identity.

Measured with `scripts/kv-quants.py` on Qwen2.5-0.5B (Q8_0 and Q4_K_M, one folder, a 568-token prompt): before, the Q4_K_M server read 36 Q8_0
blocks (560 of 568 tokens cached, 0.91 s against 3.43 s for the control with an empty folder: a wrong gain); after, it reads none (0 cached,
2.90 s) and the Q8_0 blocks sit in their own subfolder. Blocks written by versions before this change lie in the folder itself and are
never read again.

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

## A changed word

The cache finds a prompt by the chain of its blocks (16 tokens each), so a change leaves everything before the block it falls in and loses
everything after it. Measured on Qwen2.5-0.5B with a 568-token prompt (`scripts/kv-quants.py`, the same on Q8_0 and Q4_K_M): the word changed
in the first section, 0 of 568 tokens cached; in the middle, 256; near the end, 496; the original again, 560 (a block is only reused whole,
and the last token is always computed). This is why a shared prefix has to come first and byte-identical, and what changes per user or per
task after it. Where in a prompt a changed word sits decides how much of it is computed again; nothing is lost permanently: the original
prompt is still cached afterwards.

## Cache types and quantizations

Every KV cache type (`f16, bf16, fp8_e4m3, fp8_e5m2, int8, q8_0, q5_0, q5_1, q4_0, q4_1, iq4_nl, f32`) was run through disk tier and back with
the tiny model on the CPU: a restarted runtime reads the blocks and generates the tokens of a full prefill. `nvfp4` could not be run on the
CPU with that model, so it is not covered. The weights' quantization does not matter for the cache logic: the blocks are KV values, whatever
the weights are, as long as the identity above keeps different weights apart.

## A codebase in the cache

Developers who work on the same code send the same long start of a prompt: a system prompt, the tool list, the files of the project. The
prefix cache serves exactly that, and only that: a block is found again when everything before it is byte-identical, so the shared part has
to come first and in the same order for everybody, and what differs per user or per task comes after it. A date or an id in front of the
shared part makes all of it a miss.

What the cache does for such a prefix, and what was changed so that it holds for a very long one (a model with a 1M-token context, 62 500
blocks of 16 tokens):

* **It stays.** `POST /admin/prefix-cache/pin` with the same `messages`, `prompt` or `tokens` as the request keeps the cached blocks of that
  prompt out of eviction (`"pin": false` lets them go; `GET /admin/prefix-cache` shows `pinned`). The prompt is rendered like a request's, so
  the blocks that are pinned are the ones the next request looks for. A pin first reads back from the memory and disk tiers what the cache
  dropped, so the same call after a restart brings the prefix back. The answer says how many blocks were pinned and whether that is the whole
  prompt (`complete`). Pinned blocks are not free memory: a pool that is pinned full makes new requests wait. At most 50 % of the pool can
  be pinned (`FLLM_PIN_MAX_PERCENT`, 0 switches pinning off); a prompt that does not fit is pinned from its start as far as the share
  allows and the answer says `"limited": true` with `limit_blocks`. Requests with a `cache_salt` have blocks of their own that this route
  does not name.
* **Pins survive a restart.** With a disk folder the pinned chains are saved (`pins.txt` in the folder of the model: the hash of the last
  block of every pinned chain, written on every pin, unpin and clear). The next start reads the blocks back and pins those chains again
  (`pinned N prefix blocks again` in the log); a chain whose blocks are gone is skipped, and a limit that is smaller now cuts it the same way.
* **It comes back whole, and all of it.** The restart read used to follow one chain from one root; it now reads every chain on the tier (the
  cache is a tree, a leaf is a prompt, the shared start is read once). The reload reads at most 256 blocks per round while sequences run (a long read would hold up their next token)
  and up to 16 384 (262 144 tokens) per round when none runs; a longer prefix continues in the next round. The restart read is bounded by the
  free pool (minus an eighth for the first requests) and no longer by 128 blocks.
* **Making room is cheap.** The eviction used to walk the whole use order to find the oldest block without children: 6.3 s for 500
  evictions at 64 000 cached blocks. It keeps the leaves in use order now: 1.2 ms for the same.
* **Memory first.** `FLLM_KV_RAM_BYTES` holds the blocks in host memory in front of the disk. For a 1M-token prefix the size is the KV
  cache of the model for 1M tokens (it depends on the model; a 70B Llama with f16 keys and values is about 330 GB, half of it with fp8), which
  fits the host memory of a server and not the memory of one GPU.

Measured on a real model (Qwen2.5-0.5B Q8_0, `--ctx 2048 --max-seqs 2`, a 568-token shared prompt, then twelve other documents, `scripts/kv-pin.py`):

| | shared prompt afterwards | from the cache |
| --- | --- | --- |
| no pin, no tier (control) | 3.87 s | 0 of 568 |
| pinned (34 blocks), no tier | 0.48 s | 544 of 568 |
| after a restart, the pin as the first call (34 blocks read back from the disk by the pin), then the other documents | 0.59 s | 560 of 568 |
| after a restart, no pin, disk tier | 0.55 s | 560 of 568 (the reload read 35 blocks) |

The last two rows say what the pin is for and what it is not: with a disk or memory tier the reload brings a dropped prefix back on the
next request anyway, so the pin buys only that it never has to be read back (no read, no wait). Without a tier it is the only thing that keeps
the prefix. The first row is the case without either, where the cache has lost the prefix after twelve documents.

What this does not do: a model that keeps a recurrent state besides its KV cache (Qwen3.5, Mamba and others) has no prefix cache here (the
state of the middle of a prompt cannot be rebuilt from blocks), so none of the above applies to it. The prefix has to fit the model's
context and the pool; `--ctx` defaults to at most 32 768 tokens on a GPU and has to be set for more. Tests use a tiny model; nothing of this
was run with a 1M-token prompt, and the copy of 62 000 blocks between host and GPU memory is not measured.
