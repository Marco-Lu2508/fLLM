/// High performance sampling module for fllm-rs.
/// Supports deterministic greedy argmax, temperature scaling, top-k filtering,
/// top-p (nucleus) sampling, and repetition penalty with zero/minimal allocations.

#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub min_p: f64,
    pub typical_p: f64,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repeat_last_n: i64,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: usize,
    pub dry_penalty_last_n: i64,
    pub dry_breakers: Vec<Vec<i64>>,
    pub mirostat: u8,
    pub mirostat_tau: f32,
    pub mirostat_eta: f32,
    pub seed: Option<u64>,
    pub logit_bias: Vec<(usize, f32)>,
    /// Raw model probabilities, before any sampling transforms. None keeps the fast path.
    pub logprobs: Option<usize>,
    pub choices: usize,
    pub min_tokens: usize,
    pub ignore_eos: bool,
    pub stop_token_ids: Vec<i64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            min_p: 0.0,
            typical_p: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repeat_last_n: -1,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_penalty_last_n: -1,
            dry_breakers: Vec::new(),
            mirostat: 0,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            seed: None,
            logit_bias: Vec::new(),
            logprobs: None,
            choices: 1,
            min_tokens: 0,
            ignore_eos: false,
            stop_token_ids: Vec::new(),
        }
    }
}

impl SamplingParams {
    /// Explicit stop IDs retain their meaning even if natural EOS is ignored.
    pub fn stops_on_token(&self, token: i64, eos: &[i64]) -> bool {
        self.stop_token_ids.contains(&token) || !self.ignore_eos && eos.contains(&token)
    }

    #[inline]
    /// The next token is the largest logit and nothing changes the logits before it is taken (temperature 0, no penalties, bias or
    /// DRY): top-k, top-p, min-p and the like cannot matter. Servers often fill in a default top-k, so `is_greedy` alone misses this.
    pub fn argmax_only(&self) -> bool {
        self.temperature <= 1e-5
            && (self.repetition_penalty - 1.0).abs() < 1e-5
            && self.presence_penalty == 0.0
            && self.frequency_penalty == 0.0
            && self.dry_multiplier <= 0.0
            && self.logit_bias.is_empty()
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature <= 1e-5
            && (self.repetition_penalty - 1.0).abs() < 1e-5
            && (self.top_p >= 1.0 || self.top_p <= 0.0)
            && (self.top_k == 0 || self.top_k == usize::MAX)
            && self.min_p == 0.0
            && self.typical_p == 1.0
            && self.presence_penalty == 0.0
            && self.frequency_penalty == 0.0
            && self.dry_multiplier == 0.0
            && self.logit_bias.is_empty()
            && self.mirostat == 0
            && self.choices == 1
            && self.logprobs.is_none()
            && self.min_tokens == 0
            && !self.ignore_eos
            && self.stop_token_ids.is_empty()
    }
}

/// State belongs to a sequence, never to the shared inference worker.
#[derive(Debug, Clone)]
pub struct SamplingState {
    pub rng: u64,
    pub mirostat_mu: f32,
}
impl SamplingState {
    pub fn new(params: &SamplingParams, sequence_id: i64) -> Self {
        Self {
            rng: params.seed.unwrap_or(
                0x8a5cd789635d2dff ^ (sequence_id as u64).wrapping_mul(0x9e3779b97f4a7c15),
            ),
            mirostat_mu: 2.0 * params.mirostat_tau,
        }
    }
}

fn history_window(tokens: &[i64], n: i64) -> &[i64] {
    if n < 0 {
        tokens
    } else {
        &tokens[tokens.len().saturating_sub(n as usize)..]
    }
}

pub fn apply_penalties(logits: &mut [f32], history: &[i64], params: &SamplingParams) {
    let full_history = history;
    let history = history_window(history, params.repeat_last_n);
    apply_repetition_penalty(logits, history, params.repetition_penalty);
    let mut counts = std::collections::HashMap::<usize, usize>::new();
    for &token in history {
        if token >= 0 && (token as usize) < logits.len() {
            *counts.entry(token as usize).or_default() += 1;
        }
    }
    for (id, count) in counts {
        logits[id] -= params.presence_penalty + params.frequency_penalty * count as f32;
    }
    for &(id, bias) in &params.logit_bias {
        if let Some(logit) = logits.get_mut(id) {
            *logit += bias;
        }
    }
    apply_dry(logits, full_history, params);
}

/// Penalize continuations of repeated suffixes, stopping at configured sequence breakers.
fn apply_dry(logits: &mut [f32], history: &[i64], params: &SamplingParams) {
    if params.dry_multiplier <= 0.0 {
        return;
    }
    let history = history_window(history, params.dry_penalty_last_n);
    if history.len() < 2 {
        return;
    }
    let mut suffix_start = 0;
    for breaker in &params.dry_breakers {
        if breaker.is_empty() {
            continue;
        }
        if let Some(position) = history.windows(breaker.len()).rposition(|w| w == breaker) {
            suffix_start = suffix_start.max(position + breaker.len());
        }
    }
    let suffix = &history[suffix_start..];
    let mut repetitions = std::collections::HashMap::<usize, usize>::new();
    for end in 0..history.len().saturating_sub(1) {
        let next = history[end + 1];
        if next < 0 || next as usize >= logits.len() {
            continue;
        }
        let mut length = 0;
        while length < suffix.len()
            && length <= end
            && suffix[suffix.len() - 1 - length] == history[end - length]
        {
            length += 1;
        }
        if length >= params.dry_allowed_length {
            repetitions
                .entry(next as usize)
                .and_modify(|n| *n = (*n).max(length))
                .or_insert(length);
        }
    }
    for (token, length) in repetitions {
        let exponent = (length - params.dry_allowed_length).min(64) as i32;
        logits[token] -= params.dry_multiplier * params.dry_base.powi(exponent);
    }
}

pub fn sample_with_state(
    logits: &[f32],
    history: &[i64],
    params: &SamplingParams,
    state: &mut SamplingState,
) -> i64 {
    if params.is_greedy() || params.argmax_only() {
        return argmax(logits) as i64;
    }
    let mut modified = logits.to_vec();
    apply_penalties(&mut modified, history, params);
    if params.temperature <= 1e-5 {
        return argmax(&modified) as i64;
    }
    let max = modified.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let inv_temp = (1.0 / params.temperature) as f32;
    let mut candidates: Vec<(usize, f32)> = modified
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .map(|(id, value)| (id, ((*value - max) * inv_temp).exp()))
        .collect();
    candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    if candidates.is_empty() {
        return argmax(logits) as i64;
    }
    let normalize = |c: &mut Vec<(usize, f32)>| {
        let sum: f32 = c.iter().map(|x| x.1).sum();
        for x in c {
            x.1 /= sum;
        }
    };
    normalize(&mut candidates);
    if params.mirostat > 0 {
        if params.mirostat == 1 {
            let m = candidates.len().min(100);
            let mut numerator = 0.0f32;
            let mut denominator = 0.0f32;
            for i in 0..m.saturating_sub(1) {
                let t = ((i + 2) as f32 / (i + 1) as f32).ln();
                let b = (candidates[i].1.max(1e-30) / candidates[i + 1].1.max(1e-30)).ln();
                numerator += t * b;
                denominator += t * t;
            }
            let s = numerator / denominator.max(1e-9);
            let epsilon = s - 1.0;
            // The fitted Zipf exponent can be below one. exp_m1 avoids
            // cancellation near s=1; a flat distribution retains its support.
            let ratio = if epsilon.abs() < 1e-5 {
                1.0 / (logits.len() as f32).ln()
            } else {
                epsilon / -(-epsilon * (logits.len() as f32).ln()).exp_m1()
            };
            let k = if s <= 1e-6 {
                candidates.len() as f32
            } else {
                (ratio * state.mirostat_mu.exp2()).powf(1.0 / s)
            };
            candidates.truncate((k as usize).clamp(1, candidates.len()));
        } else {
            let count = candidates
                .iter()
                .take_while(|x| -x.1.max(1e-30).log2() <= state.mirostat_mu)
                .count()
                .max(1);
            candidates.truncate(count);
        }
    } else {
        if params.top_k > 0 {
            candidates.truncate(params.top_k.min(candidates.len()));
        }
        if params.min_p > 0.0 {
            let threshold = candidates[0].1 * params.min_p as f32;
            candidates.retain(|x| x.1 >= threshold);
        }
        normalize(&mut candidates);
        if params.typical_p < 1.0 {
            let entropy: f32 = candidates.iter().map(|x| -x.1 * x.1.max(1e-30).ln()).sum();
            candidates.sort_unstable_by(|a, b| {
                (-a.1.max(1e-30).ln() - entropy)
                    .abs()
                    .total_cmp(&(-b.1.max(1e-30).ln() - entropy).abs())
                    .then(a.0.cmp(&b.0))
            });
            let mut sum = 0.0;
            let count = candidates
                .iter()
                .position(|x| {
                    sum += x.1;
                    sum >= params.typical_p as f32
                })
                .map(|i| i + 1)
                .unwrap_or(candidates.len());
            candidates.truncate(count);
            candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            normalize(&mut candidates);
        }
        if params.top_p > 0.0 && params.top_p < 1.0 {
            let mut sum = 0.0;
            let count = candidates
                .iter()
                .position(|x| {
                    sum += x.1;
                    sum >= params.top_p as f32
                })
                .map(|i| i + 1)
                .unwrap_or(candidates.len());
            candidates.truncate(count);
        }
    }
    normalize(&mut candidates);
    let mut threshold = rand_f32(&mut state.rng);
    let mut selected = *candidates.last().unwrap();
    for &candidate in &candidates {
        threshold -= candidate.1;
        if threshold <= 0.0 {
            selected = candidate;
            break;
        }
    }
    if params.mirostat > 0 {
        state.mirostat_mu -=
            params.mirostat_eta * (-selected.1.max(1e-30).log2() - params.mirostat_tau);
    }
    selected.0 as i64
}

pub fn argmax(logits: &[f32]) -> usize {
    if logits.is_empty() {
        return 0;
    }
    // Two passes that the compiler turns into vector code: the largest value (eight running maxima), then its first position.
    // A plain loop that tracks the index cannot be vectorized and costs 0.15 ms per 150 000 logits; with 32 sequences per step
    // that was a third of the engine thread.
    let mut lanes = [f32::NEG_INFINITY; 8];
    let (chunks, remainder) = logits.as_chunks::<8>();
    for chunk in chunks {
        for (lane, &v) in lanes.iter_mut().zip(chunk) {
            *lane = if v > *lane { v } else { *lane };
        }
    }
    let mut max = lanes.iter().fold(f32::NEG_INFINITY, |m, &v| if v > m { v } else { m });
    for &v in remainder {
        if v > max {
            max = v;
        }
    }
    logits.iter().position(|&v| v == max).unwrap_or(0)
}

/// Fast 64-bit XorShift pseudo-random number generator.
#[inline]
pub fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    if x == 0 {
        x = 0x8a5cd789635d2dff;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Generate uniform random float in [0.0, 1.0)
#[inline]
pub fn rand_f32(state: &mut u64) -> f32 {
    let bits = (xorshift64(state) >> 40) as u32;
    bits as f32 / 16777216.0
}

/// Applies repetition penalty to logits in-place for tokens that appeared in context history.
/// Standard formula:
/// If logit > 0: logit /= penalty
/// If logit <= 0: logit *= penalty
pub fn apply_repetition_penalty(logits: &mut [f32], tokens: &[i64], penalty: f32) {
    if (penalty - 1.0).abs() < 1e-5 || tokens.is_empty() {
        return;
    }
    for (idx, &tok) in tokens.iter().enumerate() {
        if tok < 0 || tok as usize >= logits.len() {
            continue;
        }
        // Avoid penalizing the same token ID multiple times
        if tokens[..idx].contains(&tok) {
            continue;
        }
        let val = &mut logits[tok as usize];
        if *val > 0.0 {
            *val /= penalty;
        } else {
            *val *= penalty;
        }
    }
}

/// Sample a token from logits given sampling parameters and history.
pub fn sample_token(
    logits: &[f32],
    history: &[i64],
    params: &SamplingParams,
    rng_state: &mut u64,
) -> i64 {
    let mut state = SamplingState {
        rng: *rng_state,
        mirostat_mu: 2.0 * params.mirostat_tau,
    };
    let result = sample_with_state(logits, history, params, &mut state);
    *rng_state = state.rng;
    result
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[1.0, 5.0, 3.0]), 1);
        assert_eq!(argmax(&[10.0, 5.0, 3.0]), 0);
    }

    #[test]
    fn test_greedy_sample() {
        let mut rng = 42;
        let params = SamplingParams::default();
        assert_eq!(sample_token(&[1.0, 10.0, 2.0], &[], &params, &mut rng), 1);
    }

    #[test]
    fn test_repetition_penalty() {
        let mut rng = 42;
        let params = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 2.0,
            ..Default::default()
        };
        // Token 1 has logit 6.0, Token 0 has logit 4.0
        // When token 1 is in history, its logit becomes 6.0 / 2.0 = 3.0
        // Token 0 (4.0) should now win.
        let logits = [4.0, 6.0, 1.0];
        let history = [1i64];
        let picked = sample_token(&logits, &history, &params, &mut rng);
        assert_eq!(picked, 0);
    }

    #[test]
    fn test_top_k() {
        let mut rng = 999;
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 2,
            ..Default::default()
        };
        // Tokens 0 and 1 are top 2. Tokens 2, 3 have lower logits.
        let logits = [10.0, 9.0, 1.0, 0.5];
        for _ in 0..100 {
            let tok = sample_token(&logits, &[], &params, &mut rng);
            assert!(tok == 0 || tok == 1, "Expected token 0 or 1, got {}", tok);
        }
    }

    #[test]
    fn test_top_p() {
        let mut rng = 888;
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 0.9,
            ..Default::default()
        };
        // Token 0 dominates (>99% probability)
        let logits = [20.0, 2.0, 1.0];
        for _ in 0..100 {
            let tok = sample_token(&logits, &[], &params, &mut rng);
            assert_eq!(tok, 0);
        }
    }

    #[test]
    fn test_temperature_sample() {
        let mut rng = 12345;
        let mut counts = [0usize; 3];
        let logits = [1.0, 3.0, 5.0];
        let params = SamplingParams {
            temperature: 0.8,
            ..Default::default()
        };
        for _ in 0..1000 {
            let tok = sample_token(&logits, &[], &params, &mut rng);
            counts[tok as usize] += 1;
        }
        assert!(counts[2] > counts[1]);
        assert!(counts[1] > counts[0]);
    }

    #[test]
    fn min_p_removes_low_probability_candidates() {
        let params = SamplingParams {
            temperature: 1.0,
            min_p: 0.5,
            ..Default::default()
        };
        let mut state = SamplingState::new(&params, 1);
        for _ in 0..100 {
            assert_eq!(
                sample_with_state(&[5.0, 3.0, 0.0], &[], &params, &mut state),
                0
            );
        }
    }
    #[test]
    fn frequency_and_presence_penalties_use_counts_and_window() {
        let params = SamplingParams {
            frequency_penalty: 1.0,
            presence_penalty: 0.5,
            repeat_last_n: 2,
            ..Default::default()
        };
        let mut logits = [5.0, 5.0];
        apply_penalties(&mut logits, &[0, 1, 1], &params);
        assert_eq!(logits, [5.0, 2.5]);
    }
    #[test]
    fn dry_penalizes_repeated_continuation_but_respects_breakers() {
        let mut params = SamplingParams {
            dry_multiplier: 2.0,
            dry_allowed_length: 2,
            ..Default::default()
        };
        let mut logits = [0.0; 6];
        apply_penalties(&mut logits, &[1, 2, 3, 4, 1, 2], &params);
        assert_eq!(logits[3], -2.0);
        params.dry_breakers = vec![vec![2]];
        let mut logits = [0.0; 6];
        apply_penalties(&mut logits, &[1, 2, 3, 4, 1, 2], &params);
        assert_eq!(logits[3], 0.0);
    }
    #[test]
    fn seeded_sequences_are_independent_of_interleaved_requests() {
        let params = SamplingParams {
            temperature: 1.0,
            seed: Some(77),
            ..Default::default()
        };
        let mut a = SamplingState::new(&params, 1);
        let mut b = SamplingState::new(&params, 2);
        let mut other = SamplingState::new(&SamplingParams::default(), 3);
        for _ in 0..100 {
            let x = sample_with_state(&[1.0, 2.0, 3.0], &[], &params, &mut a);
            sample_with_state(&[1.0, 2.0, 3.0], &[], &params, &mut other);
            assert_eq!(x, sample_with_state(&[1.0, 2.0, 3.0], &[], &params, &mut b));
        }
    }
    #[test]
    fn mirostat_retains_and_updates_entropy_state() {
        for mode in [1, 2] {
            let params = SamplingParams {
                temperature: 1.0,
                mirostat: mode,
                mirostat_tau: 1.5,
                ..Default::default()
            };
            let mut state = SamplingState::new(&params, 1);
            let before = state.mirostat_mu;
            for _ in 0..20 {
                let id = sample_with_state(&[1.0, 2.0, 3.0], &[], &params, &mut state);
                assert!((0..3).contains(&id));
            }
            assert!(state.mirostat_mu.is_finite());
            assert_ne!(before, state.mirostat_mu);
        }
    }
}

/// One sampled token and the requested alternatives in the unmodified LM distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogprobs {
    pub logprob: f64,
    pub top: Vec<(u32, f64)>,
}

pub fn raw_logprobs(logits: &[f32], token: usize, top_n: usize) -> Result<TokenLogprobs, String> {
    if token >= logits.len() || logits.iter().any(|v| v.is_nan() || *v == f32::INFINITY) {
        return Err("invalid model logits for logprobs".into());
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    if !max.is_finite() {
        return Err("model logits contain no finite probability".into());
    }
    let normalizer = logits
        .iter()
        .map(|v| (*v as f64 - max).exp())
        .sum::<f64>()
        .ln();
    // At most 20 alternatives, without sorting or cloning the vocabulary.
    let mut top: Vec<(u32, f64)> = Vec::with_capacity(top_n.min(20) + 1);
    for (id, &v) in logits.iter().enumerate() {
        let lp = (v as f64 - max) - normalizer;
        let at =
            top.partition_point(|&(other, prob)| prob > lp || (prob == lp && other < id as u32));
        if at < top_n.min(20) {
            top.insert(at, (id as u32, lp));
            top.truncate(top_n.min(20));
        }
    }
    Ok(TokenLogprobs {
        logprob: (logits[token] as f64 - max) - normalizer,
        top,
    })
}

#[cfg(test)]
mod probability_tests {
    use super::*;
    #[test]
    fn raw_probabilities_are_stable_and_ties_keep_token_order() {
        let p = raw_logprobs(&[1000.0, 1000.0, 999.0], 2, 2).unwrap();
        assert_eq!(p.top.iter().map(|v| v.0).collect::<Vec<_>>(), [0, 1]);
        assert!((p.logprob - (-1.0 - (2.0 + (-1.0_f64).exp()).ln())).abs() < 1e-12);
        let shifted = raw_logprobs(&[f32::MAX, f32::MAX], 0, 0).unwrap();
        assert!((shifted.logprob + 2.0_f64.ln()).abs() < 1e-12);
        assert!(raw_logprobs(&[f32::NAN], 0, 1).is_err());
        assert!(raw_logprobs(&[f32::NEG_INFINITY], 0, 1).is_err());
    }
    #[test]
    fn ignored_eos_can_be_sampled_but_explicit_stop_ids_still_end_generation() {
        let mut p = SamplingParams {
            ignore_eos: true,
            stop_token_ids: vec![3],
            ..Default::default()
        };
        assert!(!p.stops_on_token(2, &[2]));
        assert!(p.stops_on_token(3, &[2]));
        p.stop_token_ids.push(2);
        assert!(p.stops_on_token(2, &[2]));
    }
    #[test]
    fn seeded_sequences_do_not_share_rng_state() {
        let p = SamplingParams {
            temperature: 1.0,
            seed: Some(19),
            ..Default::default()
        };
        let mut a = SamplingState::new(&p, 1);
        let mut b = SamplingState::new(&p, 999);
        let mut other = SamplingState::new(&p, 2);
        for _ in 0..32 {
            let expected = sample_with_state(&[0.0, 0.2, 0.4], &[], &p, &mut a);
            sample_with_state(&[1.0, 2.0, 3.0], &[], &p, &mut other);
            assert_eq!(
                expected,
                sample_with_state(&[0.0, 0.2, 0.4], &[], &p, &mut b)
            );
        }
    }
}
