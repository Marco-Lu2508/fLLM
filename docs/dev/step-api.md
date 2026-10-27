# Step-wise generation: what blocks beam search, multi-step and jump-forward

Written after reading the engine, not after guessing. Three features are missing for the same reason, and the reason is a
single missing seam.

## Where generation actually happens

- `Scheduler::schedule()` returns a `ScheduledStep { prefill, decode, verify }` — **only tests call it**. It is not the
  production path.
- Production runs through `RustDecoder::forward_with_hidden(ids, lengths, tables, key, value, logit_rows, …)`: rank 0 builds
  a batch, every rank computes the same rows, and the rows named in `logit_rows` come back as logits.
- With `--parallel` (tensor parallel), `parallel::group::worker_loop` is a thread per rank that receives a `StepDesc` and
  calls exactly that function. Rank 0 decides what a step is.

So the seam already exists at the *batch* level. What does not exist is a way to **stop between two tokens**, look at the
logits, change which sequences continue, and resume — which is precisely what beam search, multi-step scheduling and
jump-forward need.

## What the seam has to look like

```rust
/// One forward pass over the current batch, with the logits of every active sequence.
pub struct StepOutput {
    pub seq_ids: Vec<i64>,
    /// logits of the last row of each sequence (already gathered by `logit_rows`)
    pub logits: Vec<Vec<f32>>,
}

pub trait Stepwise {
    /// Advance every active sequence by exactly one token and hand back what the sampler needs.
    fn step_once(&mut self) -> Result<StepOutput, String>;
    /// Drop sequences that lost the beam, keep the KV blocks of the winners.
    fn keep_only(&mut self, keep: &[i64]) -> Result<(), String>;
    /// Append a token to a sequence (the sampler's decision, or a forced token for jump-forward).
    fn push_token(&mut self, seq: i64, token: i64) -> Result<(), String>;
}
```

Three properties matter:

1. `step_once` must be **exactly** the batch that a normal decode step builds, otherwise the numbers differ from
   non-beam runs and nobody can compare them.
2. `keep_only` must free the KV blocks of dropped sequences (`BlockManager::free_sequence`) and leave the rest untouched —
   a beam that leaks blocks starves the batch after a few dozen tokens.
3. `push_token` has to bypass the sampler but not the grammar: jump-forward inserts tokens that the grammar has already
   accepted.

## What each feature then needs

| Feature | On top of the seam |
|---|---|
| Beam search | K sequences per request; per step, take `top_b` logprobs of each, expand to K·B, score with the length-normalised sum, `keep_only(K)` |
| Multi-step scheduling | call `step_once` n times while the batch does not change, amortising the scheduler's bookkeeping (the forward passes stay n) |
| Jump-forward | ask the grammar for its forced prefix, `push_token` those tokens, verify in one `step_once` and roll back the KV of the rejected tail |

## Order

1. `step_once` + `keep_only` behind a feature flag, with a test that a single-sequence run through the seam produces the
   same tokens as the normal path on the tiny model (`tiny_models::decode` is the reference).
2. Beam search on top — the only one whose interface is fully determined by the OpenAI field `use_beam_search`.
3. Multi-step and jump-forward afterwards; both are optimisations and both need a benchmark before/after to be worth it.

## Why this was not written blind

An earlier session tried to build beam search on top of the one-shot API and stopped, because the result would have been
`best_of` with a different name. The server now says exactly that in its error message instead of pretending.

Nothing here needs a GPU. The CUDA items (TP/PP/EP beyond the existing parallel group, CUDA graph buckets, P/D, FA-3, FP8)
are a separate problem: they need hardware that this network does not have.
