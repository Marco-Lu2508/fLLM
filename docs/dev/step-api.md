# Step-wise generation: beam search, multi-step scheduling and jump-forward

The first version of this note (written before any of it existed) argued that three features need one missing seam: a way to
stop between two tokens, look at the logits, change which sequences continue and resume. This version says what was built
and how it differs from that sketch.

## Status

| Feature | State |
|---|---|
| Beam search (`use_beam_search`) | built, CPU backend, non-streaming. `beam.rs`, `InferenceRuntime::submit_beam`, `docs/api-compat.md` |
| Multi-step scheduling (`--multi-step N`) | built, CPU backend only; ignored on accelerators until someone verifies it there |
| Jump-forward | not built |

## Where generation happens

- `Scheduler::schedule()` returns a `ScheduledStep { prefill, decode, verify }`. `InferenceRuntime::step_round` calls it,
  builds one ragged batch, runs `RustDecoder::forward_with_hidden` (rank 0 decides the batch; with `--parallel` the other
  ranks run the same call from `parallel::group::worker_loop`), and samples the rows named in `logit_rows`.
- The sampler runs inside `step_round`, per decode row. There is no `Stepwise` trait. The sketch in the first version
  (`step_once` / `keep_only` / `push_token`) turned out to be unnecessary for beam search: a beam is an ordinary scheduled
  sequence whose sampling step is replaced by "keep the logits and wait".

## Beam search as built

- A beam search is a `BeamRun` (`runtime.rs`) around a `beam::BeamSearch` (`beam.rs`, pure scoring, unit-tested without a
  model). Every live beam is a normal `Sequence` of the request, so the scheduler, the batch builder, preemption and the
  block manager treat it like any other.
- In `step_round`, a decode row of a beam sequence is not sampled. Its logit row is stored in `pending_logits` (the same map
  that holds the prompt's last logits for a sequence that has just finished prefill) and the position is committed with
  `commit_chunk`. The sequence is now in the "prompt finished, token pending" state, which the scheduler already knows.
- `drive_beams` runs after the decode loop. A search advances when all of its beams are *ready* (running, KV complete,
  logits pending, not swapped out). A beam that was preempted for recompute becomes ready again by itself, because the
  re-prefill ends in the same pending state; the group simply waits.
- `BeamSearch::advance` returns, for the next step, which parent each surviving beam extends and with which token. The
  runtime then (1) releases beams that nobody extends, so their blocks are free for the forks, (2) forks a parent for each
  further child **before** the parent moves on, (3) appends the chosen token to every survivor and commits it.
- `fork_sequence` shares completely computed blocks by reference count and copies only the block that holds the computed
  tail. Blocks are append-only, so a shared full block is never written; this is the copy-on-write of the block manager. (The
  first version of the primitive copied every block and took one reference too many for each of them, so a forked
  sequence's blocks were never freed. Fixed together with this work.)
- The result leaves the runtime as `TokenEvent`s, one choice per returned hypothesis, produced when the search ends and
  handed out by `take_beam_events` as the client channel has room (the channel is bounded, a burst of 8 x 4096 events would
  overflow it). `active_requests()` counts a request with undelivered events.
- Budget: `submit_beam` refuses (`beam::REFUSED`, HTTP 400) a width above `MAX_BEAMS` (8) or above `--max-seqs`, and a request
  whose worst case (prompt blocks plus, per beam, `max_tokens` worth of blocks and one tail) exceeds the pool. It answers
  `beam::BUSY` (HTTP 503) when other searches already hold the sequence slots; without that rule two searches could each
  hold half of the slots and wait for the other forever. If the pool runs out later anyway (other requests took the blocks),
  the search fails alone with an error; the other requests continue.

What is proven: see `rust/engine/src/beam_tests.rs` (runtime) and `beam::tests` (scoring), and the HTTP test
`beam_search_over_http`. What is not: GPU backends (`fork_ok` decides), models with recurrent state (refused for a width
above 1), grammar-constrained and streaming beam search (refused), `logprobs` of beam output (refused).

## Multi-step scheduling as built

`InferenceRuntime::step()` runs one scheduler round and then up to `multi_step - 1` further rounds on the same batch.
`multi_step_batch` hands out that batch only when the previous round was nothing but plain decode rows and nothing changed:
the running set equals the decode set, nobody waits, every sequence still has a token to decode and a block for it (taken
up front, so the extra round cannot fail half way), no request timed out, no speculation engine is attached, and no
sequence has a grammar, media or a beam search. Otherwise the round is an ordinary single step.

Each extra round goes through the same `step_round` body as a normal one (sampling, stop checks, `max_tokens`, EOS, events
in order). It only skips `Scheduler::schedule()` and the draft proposal code. Because a sequence that finishes leaves the
running set, the next round is not replayed, so nothing is generated past a stop token or `max_tokens`.

What it saves is the scheduler's bookkeeping between rounds (`scheduler_ns`), not forward passes: the number of forward
passes is identical, which the test asserts. The only number measured is `scheduler_ns` on the tiny test model, in one
noisy run: about 17 us single-step against 8 to 11 us with `--multi-step` for 40 forward passes. That is not a speedup
claim for a real model, where the forward pass dominates and nothing was measured. A stop *string* is
noticed by the server after the tokens exist, so up to `N - 1` tokens of a stopping request are generated and discarded.

## Jump-forward (not built)

It would ask the grammar for its forced prefix, append those tokens, verify them in one forward pass and roll back the
rejected tail. The pieces now exist (`fork_sequence`, `commit_decode`, the pending-logits state), but nothing uses them,
and it needs a benchmark before and after to be worth merging.
