//! Beam search: the scoring and bookkeeping, without any model.
//!
//! The runtime (`InferenceRuntime::submit_beam`) owns the sequences and their KV blocks; this module decides, from the logits
//! of every live beam, which continuations survive. Keeping the two apart makes the search testable with plain numbers.
//!
//! Semantics (the same as vLLM's `use_beam_search`, spelled out because the details differ between engines):
//!
//! * The score of a hypothesis is the sum of the log-probabilities of its generated tokens (log-softmax of the raw logits after
//!   `logit_bias`, `min_tokens`, `allowed_token_ids` and bad-word controls), divided by `len ^ length_penalty`. `len` is the
//!   number of generated tokens, the final stop token included. Temperature, `top_p`, `top_k` and `seed` play no role.
//! * All live beams have the same length. Per step every beam proposes its best `2 * width + (number of stop tokens)` tokens;
//!   the candidates are ordered by total log-probability (ties: lower beam index, then lower token id), and the best `width`
//!   non-finishing ones become the next beams. A candidate that finishes (stop token, or a stop string through `stop_check`)
//!   is kept as a finished hypothesis only when it ranks among the first `width` candidates, otherwise it is dropped.
//! * With `width == 1` this is greedy decoding: ties go to the lower token id, as in the greedy sampler.
//! * The search ends when no beam is left, when the length limit is reached (the live beams are then finished with
//!   `length`), or by the `early_stopping` rule once `width` hypotheses are finished.

use std::sync::Arc;

/// Largest beam width accepted (`best_of` with `use_beam_search`). Every beam is a sequence with its own KV tail.
pub const MAX_BEAMS: usize = 8;
/// Prefix of the error messages for requests that can never run (HTTP 400).
pub const REFUSED: &str = "beam search refused: ";
/// Prefix of the error messages for requests that cannot run right now (HTTP 503).
pub const BUSY: &str = "beam search busy: ";

/// `early_stopping` of vLLM and Hugging Face.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EarlyStopping {
    /// `false`: stop when the best live beam can no longer beat the worst finished hypothesis, judged at its current length.
    #[default]
    Heuristic,
    /// `true`: stop as soon as `width` hypotheses are finished.
    Always,
    /// `"never"`: like the heuristic, but with the best length the live beam could still reach (`max_tokens` when
    /// `length_penalty > 0`), so the search only stops when no live beam can win.
    Never,
}

/// Tells whether the generated tokens of a candidate end in a stop string.
pub type StopCheck = Arc<dyn Fn(&[i64]) -> bool + Send + Sync>;

#[derive(Clone)]
pub struct BeamParams {
    /// Beams kept alive (vLLM: `best_of`).
    pub width: usize,
    /// Hypotheses returned (`n`), at most `width`.
    pub returned: usize,
    pub length_penalty: f64,
    pub early_stopping: EarlyStopping,
    /// Stop strings of the request, as a function of the generated tokens; `None` when the request has none.
    pub stop_check: Option<StopCheck>,
    /// Keep the finished hypotheses for `InferenceRuntime::take_beam_result` (used by tests and library callers).
    pub keep_result: bool,
}

impl BeamParams {
    pub fn new(width: usize, returned: usize) -> Self {
        Self {
            width,
            returned,
            length_penalty: 1.0,
            early_stopping: EarlyStopping::Heuristic,
            stop_check: None,
            keep_result: false,
        }
    }
}

impl std::fmt::Debug for BeamParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BeamParams")
            .field("width", &self.width)
            .field("returned", &self.returned)
            .field("length_penalty", &self.length_penalty)
            .field("early_stopping", &self.early_stopping)
            .field("stop_check", &self.stop_check.is_some())
            .finish()
    }
}

/// A finished hypothesis.
#[derive(Debug, Clone, PartialEq)]
pub struct Hypothesis {
    /// Generated tokens, the stop token included.
    pub tokens: Vec<i64>,
    /// Log-probability of each generated token.
    pub token_logprobs: Vec<f64>,
    /// Sum of `token_logprobs`.
    pub logprob: f64,
    /// `logprob / len ^ length_penalty`, the number the hypotheses are ranked by.
    pub score: f64,
    /// `stop` (stop token or stop string) or `length`.
    pub finish_reason: &'static str,
}

/// A live beam.
#[derive(Debug, Clone, PartialEq)]
pub struct Beam {
    /// The runtime sequence that holds this beam's KV.
    pub seq: i64,
    pub tokens: Vec<i64>,
    pub token_logprobs: Vec<f64>,
    pub logprob: f64,
}

/// One surviving continuation: extend beam `parent` (an index into the beams before the step) by `token`.
#[derive(Debug, Clone, PartialEq)]
pub struct Selection {
    pub parent: usize,
    pub token: i64,
    pub logprob: f64,
}

pub struct BeamSearch {
    params: BeamParams,
    max_tokens: usize,
    min_tokens: usize,
    finishing: Vec<i64>,
    pub alive: Vec<Beam>,
    finished: Vec<Hypothesis>,
    done: bool,
}

fn log_sum_exp(row: &[f32]) -> f64 {
    let max = row.iter().copied().filter(|v| v.is_finite()).fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return f64::NEG_INFINITY;
    }
    let sum: f64 = row.iter().filter(|v| v.is_finite()).map(|&v| f64::from(v - max).exp()).sum();
    f64::from(max) + sum.ln()
}

impl BeamSearch {
    /// `finishing`: tokens that end a hypothesis (EOS and the request's stop token ids). `leader` is the sequence that holds the
    /// prompt.
    pub fn new(params: BeamParams, leader: i64, max_tokens: usize, min_tokens: usize, finishing: Vec<i64>) -> Self {
        Self {
            params,
            max_tokens,
            min_tokens,
            finishing,
            alive: vec![Beam { seq: leader, tokens: Vec::new(), token_logprobs: Vec::new(), logprob: 0.0 }],
            finished: Vec::new(),
            done: false,
        }
    }

    pub fn params(&self) -> &BeamParams {
        &self.params
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Finished hypotheses, best first.
    pub fn finished(&self) -> &[Hypothesis] {
        &self.finished
    }

    /// The `returned` best hypotheses, best first.
    pub fn best(&self) -> Vec<Hypothesis> {
        self.finished.iter().take(self.params.returned).cloned().collect()
    }

    fn normalise(&self, logprob: f64, len: usize) -> f64 {
        logprob / (len.max(1) as f64).powf(self.params.length_penalty)
    }

    fn finish(&mut self, tokens: Vec<i64>, token_logprobs: Vec<f64>, logprob: f64, reason: &'static str) {
        let score = self.normalise(logprob, tokens.len());
        self.finished.push(Hypothesis { tokens, token_logprobs, logprob, score, finish_reason: reason });
        // stable: of two equal scores the one that finished first stays ahead
        self.finished.sort_by(|a, b| b.score.total_cmp(&a.score));
        self.finished.truncate(self.params.width);
    }

    /// One step. `logits[i]` are the (already controlled) logits that follow beam `alive[i]`. Returns the surviving
    /// continuations in their new order; `alive` is replaced accordingly (a child keeps its parent's `seq` until the runtime
    /// assigns the forks). After `is_done()` the returned list is empty.
    pub fn advance(&mut self, logits: &[Vec<f32>]) -> Vec<Selection> {
        assert_eq!(logits.len(), self.alive.len(), "one logit row per live beam");
        let width = self.params.width;
        let per_beam = 2 * width + self.finishing.len();
        struct Candidate {
            beam: usize,
            token: i64,
            logprob: f64,
            total: f64,
        }
        let mut candidates: Vec<Candidate> = Vec::new();
        for (beam, row) in logits.iter().enumerate() {
            let lse = log_sum_exp(row);
            if !lse.is_finite() {
                continue;
            }
            let mut order: Vec<usize> = (0..row.len()).filter(|&i| row[i].is_finite()).collect();
            let by_logit = |a: &usize, b: &usize| row[*b].total_cmp(&row[*a]).then(a.cmp(b));
            if order.len() > per_beam {
                order.select_nth_unstable_by(per_beam, by_logit);
                order.truncate(per_beam);
            }
            order.sort_by(by_logit);
            for token in order {
                let logprob = f64::from(row[token]) - lse;
                candidates.push(Candidate { beam, token: token as i64, logprob, total: self.alive[beam].logprob + logprob });
            }
        }
        candidates.sort_by(|a, b| b.total.total_cmp(&a.total).then(a.beam.cmp(&b.beam)).then(a.token.cmp(&b.token)));

        let length = self.alive.first().map_or(0, |b| b.tokens.len()) + 1;
        let mut next: Vec<(Selection, Beam)> = Vec::new();
        for (rank, candidate) in candidates.iter().enumerate() {
            if next.len() >= width {
                break;
            }
            let parent = &self.alive[candidate.beam];
            let mut tokens = parent.tokens.clone();
            tokens.push(candidate.token);
            let ends = self.finishing.contains(&candidate.token)
                || (tokens.len() >= self.min_tokens && self.params.stop_check.as_ref().is_some_and(|check| check(&tokens)));
            let mut token_logprobs = parent.token_logprobs.clone();
            token_logprobs.push(candidate.logprob);
            if ends {
                if rank < width {
                    self.finish(tokens, token_logprobs, candidate.total, "stop");
                }
                continue;
            }
            next.push((
                Selection { parent: candidate.beam, token: candidate.token, logprob: candidate.logprob },
                Beam { seq: parent.seq, tokens, token_logprobs, logprob: candidate.total },
            ));
        }

        if next.is_empty() {
            self.done = true;
            self.alive.clear();
            return Vec::new();
        }
        if length >= self.max_tokens {
            // the length limit: whatever is still alive ends here
            for (_, beam) in next {
                self.finish(beam.tokens, beam.token_logprobs, beam.logprob, "length");
            }
            self.done = true;
            self.alive.clear();
            return Vec::new();
        }
        if self.finished.len() >= width {
            let done = match self.params.early_stopping {
                EarlyStopping::Always => true,
                rule => {
                    let best = next.iter().map(|(_, beam)| beam.logprob).fold(f64::NEG_INFINITY, f64::max);
                    let reachable = if rule == EarlyStopping::Never && self.params.length_penalty > 0.0 { self.max_tokens } else { length };
                    let attainable = self.normalise(best, reachable);
                    self.finished.last().is_some_and(|worst| worst.score >= attainable)
                }
            };
            if done {
                self.done = true;
                self.alive.clear();
                return Vec::new();
            }
        }
        let (selections, beams): (Vec<_>, Vec<_>) = next.into_iter().unzip();
        self.alive = beams;
        selections
    }

    /// Records the sequence of beam `index` after the runtime has forked the children.
    pub fn set_seq(&mut self, index: usize, seq: i64) {
        self.alive[index].seq = seq;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(values: &[f32]) -> Vec<f32> {
        values.to_vec()
    }

    fn search(width: usize, max_tokens: usize, finishing: Vec<i64>) -> BeamSearch {
        BeamSearch::new(BeamParams::new(width, width), 1, max_tokens, 0, finishing)
    }

    #[test]
    fn width_one_takes_the_largest_logit_and_the_lower_id_on_ties() {
        let mut s = search(1, 10, vec![]);
        let picked = s.advance(&[row(&[1.0, 3.0, 3.0, 0.0])]);
        assert_eq!(picked.len(), 1);
        assert_eq!(picked[0].token, 1, "a tie goes to the lower token id, like the greedy sampler");
        assert!(picked[0].logprob < 0.0);
    }

    #[test]
    fn candidates_are_ranked_by_total_logprob_across_beams() {
        let mut s = search(2, 10, vec![]);
        // step 1 from the single leader: tokens 0 and 1 survive
        let first = s.advance(&[row(&[2.0, 1.0, -5.0, -5.0])]);
        assert_eq!(first.iter().map(|p| p.token).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(s.alive.len(), 2);
        // step 2: beam 0 is flat, beam 1 has one very likely token; the sum decides
        let second = s.advance(&[row(&[0.0, 0.0, 0.0, 0.0]), row(&[10.0, -10.0, -10.0, -10.0])]);
        let sums: Vec<f64> = s.alive.iter().map(|b| b.logprob).collect();
        assert!(sums[0] >= sums[1], "beams are ordered best first");
        assert_eq!(second.len(), 2);
        assert_eq!(second[0].parent, 1, "the unlikely parent with the near-certain continuation wins");
    }

    #[test]
    fn a_stop_token_finishes_a_hypothesis_only_within_the_first_width_candidates() {
        // token 3 is the stop token
        let mut s = search(2, 10, vec![3]);
        // stop token ranks first: finished, and two other beams continue
        let picked = s.advance(&[row(&[1.0, 0.5, 0.0, 9.0])]);
        assert_eq!(s.finished().len(), 1);
        assert_eq!(s.finished()[0].tokens, vec![3]);
        assert_eq!(s.finished()[0].finish_reason, "stop");
        assert_eq!(picked.len(), 2);
        assert!(!s.is_done());
        // a stop token far down the list is ignored, not finished
        let mut s = search(1, 10, vec![3]);
        let picked = s.advance(&[row(&[9.0, 8.0, 7.0, 6.0])]);
        assert_eq!(picked[0].token, 0);
        assert!(s.finished().is_empty());
    }

    #[test]
    fn the_length_limit_finishes_live_beams_with_length() {
        let mut s = search(2, 2, vec![]);
        s.advance(&[row(&[2.0, 1.0, 0.0])]);
        assert!(!s.is_done());
        let picked = s.advance(&[row(&[2.0, 1.0, 0.0]), row(&[0.0, 3.0, 0.0])]);
        assert!(picked.is_empty());
        assert!(s.is_done());
        assert_eq!(s.finished().len(), 2);
        assert!(s.finished().iter().all(|h| h.finish_reason == "length" && h.tokens.len() == 2));
        assert!(s.finished()[0].score >= s.finished()[1].score);
    }

    #[test]
    fn length_penalty_changes_the_ranking() {
        let mut short = BeamSearch::new(BeamParams::new(1, 1), 1, 10, 0, vec![]);
        short.finish(vec![1], vec![-1.0], -1.0, "stop");
        let mut long = BeamSearch::new(BeamParams { length_penalty: 2.0, ..BeamParams::new(2, 2) }, 1, 10, 0, vec![]);
        long.finish(vec![1], vec![-1.0], -1.0, "stop");
        long.finish(vec![1, 2, 3, 4], vec![-1.0; 4], -4.0, "stop");
        // -4 / 4^2 = -0.25 beats -1 / 1^2 = -1 only with the larger penalty
        assert_eq!(long.finished()[0].tokens.len(), 4);
        assert!((short.finished()[0].score + 1.0).abs() < 1e-12);
    }

    #[test]
    fn early_stopping_always_ends_once_width_hypotheses_are_finished() {
        let params = BeamParams { early_stopping: EarlyStopping::Always, ..BeamParams::new(1, 1) };
        let mut s = BeamSearch::new(params, 1, 10, 0, vec![2]);
        // the stop token is the best candidate, so one hypothesis is finished and the search ends
        let picked = s.advance(&[row(&[0.0, 0.0, 5.0])]);
        assert!(picked.is_empty());
        assert!(s.is_done());
        assert_eq!(s.best()[0].tokens, vec![2]);
    }

    #[test]
    fn a_stop_string_check_finishes_a_hypothesis() {
        let check: StopCheck = Arc::new(|tokens: &[i64]| tokens.last() == Some(&1));
        let params = BeamParams { stop_check: Some(check), ..BeamParams::new(2, 2) };
        let mut s = BeamSearch::new(params, 1, 10, 0, vec![]);
        s.advance(&[row(&[1.0, 3.0, 0.0, -1.0])]);
        assert_eq!(s.finished().len(), 1);
        assert_eq!(s.finished()[0].tokens, vec![1]);
        assert_eq!(s.finished()[0].finish_reason, "stop");
    }

    #[test]
    fn min_tokens_delays_the_stop_string_check() {
        let check: StopCheck = Arc::new(|_: &[i64]| true);
        let params = BeamParams { stop_check: Some(check), ..BeamParams::new(1, 1) };
        let mut s = BeamSearch::new(params, 1, 10, 3, vec![]);
        let picked = s.advance(&[row(&[1.0, 0.0])]);
        assert_eq!(picked.len(), 1, "fewer than min_tokens: the check is not consulted");
        assert!(s.finished().is_empty());
    }
}
