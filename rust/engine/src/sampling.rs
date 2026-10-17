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
    /// Request-level switch for speculative decoding (`"speculative": false` in the request body opts out).
    pub speculative: bool,
    /// When non-empty only these tokens may be generated.
    pub allowed_token_ids: Vec<usize>,
    /// Token sequences that must not be generated (the last token of a sequence is blocked once its prefix was generated).
    pub bad_word_ids: Vec<Vec<i64>>,
    /// `logprobs_mode`: report the log-probabilities of the distribution the token was drawn from (after penalties, temperature and
    /// filters) instead of the raw model distribution.
    pub logprobs_processed: bool,
    /// Scheduling priority: smaller values are served first under `--scheduling-policy priority`.
    pub priority: i32,
    /// Seconds after which the request is aborted (queue time included).
    pub timeout_secs: Option<f64>,
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
            speculative: true,
            allowed_token_ids: Vec::new(),
            bad_word_ids: Vec::new(),
            logprobs_processed: false,
            priority: 0,
            timeout_secs: None,
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
            && self.allowed_token_ids.is_empty()
            && self.bad_word_ids.is_empty()
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
            && self.allowed_token_ids.is_empty()
            && self.bad_word_ids.is_empty()
    }
}

/// Source of the uniform random numbers of a sequence. `Philox` is a counter-based generator keyed by the request seed: draw `n` of
/// a sequence is a pure function of `(seed, n)`, so the result does not depend on batching, on other requests or on the thread that
/// runs the step, and the CUDA sampler (which uses the same generator) can reproduce it. `Xorshift` is the previous stateful
/// generator. `FLLM_SAMPLER_RNG=xorshift` selects it; the default is `philox`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RngKind {
    Philox,
    Xorshift,
}

pub fn default_rng_kind() -> RngKind {
    static KIND: std::sync::OnceLock<RngKind> = std::sync::OnceLock::new();
    *KIND.get_or_init(|| match std::env::var("FLLM_SAMPLER_RNG").ok().as_deref() {
        Some("xorshift") => RngKind::Xorshift,
        _ => RngKind::Philox,
    })
}

const PHILOX_M0: u32 = 0xD251_1F53;
const PHILOX_M1: u32 = 0xCD9E_8D57;
const PHILOX_W0: u32 = 0x9E37_79B9;
const PHILOX_W1: u32 = 0xBB67_AE85;

/// Philox-4x32-10 (Salmon et al., "Parallel random numbers: as easy as 1, 2, 3", SC11; the Random123 reference constants).
pub fn philox4x32_10(mut ctr: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    for _ in 0..10 {
        let p0 = u64::from(PHILOX_M0) * u64::from(ctr[0]);
        let p1 = u64::from(PHILOX_M1) * u64::from(ctr[2]);
        ctr = [
            ((p1 >> 32) as u32) ^ ctr[1] ^ key[0],
            p1 as u32,
            ((p0 >> 32) as u32) ^ ctr[3] ^ key[1],
            p0 as u32,
        ];
        key = [key[0].wrapping_add(PHILOX_W0), key[1].wrapping_add(PHILOX_W1)];
    }
    ctr
}

/// Uniform number in the open interval (0, 1) for draw `draw` and element `index` of the stream of `seed`. The CUDA sampler calls the
/// same function with `index` = vocabulary id for its per-token Gumbel noise; sequential draws of one sequence use `index` 0.
#[inline]
pub fn philox_uniform(seed: u64, draw: u64, index: u32) -> f32 {
    let w = philox4x32_10(
        [index, 0, draw as u32, (draw >> 32) as u32],
        [seed as u32, (seed >> 32) as u32],
    )[0];
    ((w >> 8) as f32 + 0.5) / 16_777_216.0
}

/// State belongs to a sequence, never to the shared inference worker.
#[derive(Debug, Clone)]
pub struct SamplingState {
    /// The request seed (Philox key) or the xorshift state.
    pub rng: u64,
    pub mirostat_mu: f32,
    /// Number of uniform draws taken so far (the Philox counter).
    pub draws: u64,
    pub kind: RngKind,
}
impl SamplingState {
    pub fn new(params: &SamplingParams, sequence_id: i64) -> Self {
        Self::with_kind(params, sequence_id, default_rng_kind())
    }
    pub fn with_kind(params: &SamplingParams, sequence_id: i64, kind: RngKind) -> Self {
        Self {
            rng: params.seed.unwrap_or(
                0x8a5cd789635d2dff ^ (sequence_id as u64).wrapping_mul(0x9e3779b97f4a7c15),
            ),
            mirostat_mu: 2.0 * params.mirostat_tau,
            draws: 0,
            kind,
        }
    }
    /// The next uniform number of this sequence, in [0, 1) for xorshift and (0, 1) for Philox.
    pub fn next_uniform(&mut self) -> f32 {
        match self.kind {
            RngKind::Philox => {
                let u = philox_uniform(self.rng, self.draws, 0);
                self.draws += 1;
                u
            }
            RngKind::Xorshift => rand_f32(&mut self.rng),
        }
    }
}

/// Hard constraints on the next token that are applied before the penalties: `min_tokens` (no end of sequence yet), the allow list,
/// and bad words. `generated` is the number of tokens generated so far and `history` the full token history.
pub fn apply_generation_controls(logits: &mut [f32], params: &SamplingParams, eos: &[i64], generated: usize, history: &[i64]) {
    if generated < params.min_tokens {
        for &id in eos.iter().chain(&params.stop_token_ids) {
            if let Some(value) = logits.get_mut(id as usize) {
                *value = f32::NEG_INFINITY;
            }
        }
    }
    if !params.allowed_token_ids.is_empty() {
        let mut allowed = vec![false; logits.len()];
        for &id in &params.allowed_token_ids {
            if let Some(slot) = allowed.get_mut(id) {
                *slot = true;
            }
        }
        for (logit, ok) in logits.iter_mut().zip(allowed) {
            if !ok {
                *logit = f32::NEG_INFINITY;
            }
        }
    }
    for word in &params.bad_word_ids {
        let Some((&last, prefix)) = word.split_last() else { continue };
        if history.len() >= prefix.len() && history[history.len() - prefix.len()..] == *prefix {
            if let Some(value) = logits.get_mut(last as usize) {
                *value = f32::NEG_INFINITY;
            }
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
    let candidates = processed_distribution(logits, history, params, state.mirostat_mu);
    if candidates.len() == 1 {
        // a point mass (greedy after the penalties, or a single survivor) needs no random number
        if params.mirostat > 0 {
            state.mirostat_mu -=
                params.mirostat_eta * (-candidates[0].1.max(1e-30).log2() - params.mirostat_tau);
        }
        return candidates[0].0 as i64;
    }
    let selected = select_from(&candidates, state.next_uniform());
    if params.mirostat > 0 {
        state.mirostat_mu -=
            params.mirostat_eta * (-selected.1.max(1e-30).log2() - params.mirostat_tau);
    }
    selected.0 as i64
}

/// The entry of `candidates` (probabilities that sum to one) selected by the uniform number `u`.
pub fn select_from(candidates: &[(usize, f32)], u: f32) -> (usize, f32) {
    let mut threshold = u;
    let mut selected = *candidates.last().unwrap();
    for &candidate in candidates {
        threshold -= candidate.1;
        if threshold <= 0.0 {
            selected = candidate;
            break;
        }
    }
    selected
}

/// The distribution a request samples from after the penalties, logit bias, temperature, top-k, min-p, typical-p and top-p (or the
/// mirostat truncation): `(token, probability)` pairs, sorted by probability (ties: lower id first), normalized. Greedy requests give a
/// single pair with probability one. Speculative decoding uses this for the target and the draft so that both are filtered the same way.
pub fn processed_distribution(
    logits: &[f32],
    history: &[i64],
    params: &SamplingParams,
    mirostat_mu: f32,
) -> Vec<(usize, f32)> {
    if params.is_greedy() || params.argmax_only() {
        return vec![(argmax(logits), 1.0)];
    }
    let mut modified = logits.to_vec();
    apply_penalties(&mut modified, history, params);
    if params.temperature <= 1e-5 {
        return vec![(argmax(&modified), 1.0)];
    }
    let max = modified.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let inv_temp = (1.0 / params.temperature) as f32;
    let mut candidates: Vec<(usize, f32)> = modified
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .map(|(id, value)| (id, ((*value - max) * inv_temp).exp()))
        .collect();
    let by_probability = |a: &(usize, f32), b: &(usize, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
    if params.mirostat == 0 && params.top_k > 0 && params.top_k < candidates.len() {
        // top-k keeps the same set whether or not the tail is sorted first; selecting it avoids sorting the whole vocabulary
        candidates.select_nth_unstable_by(params.top_k - 1, by_probability);
        candidates.truncate(params.top_k);
    }
    candidates.sort_unstable_by(by_probability);
    if candidates.is_empty() {
        return vec![(argmax(logits), 1.0)];
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
                (ratio * mirostat_mu.exp2()).powf(1.0 / s)
            };
            candidates.truncate((k as usize).clamp(1, candidates.len()));
        } else {
            let count = candidates
                .iter()
                .take_while(|x| -x.1.max(1e-30).log2() <= mirostat_mu)
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
    candidates
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
        draws: 0,
        kind: RngKind::Xorshift,
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

/// Log-probabilities from a processed distribution (`processed_distribution`): the chosen token and the best alternatives.
pub fn processed_logprobs(dist: &[(usize, f32)], token: usize, top_n: usize) -> TokenLogprobs {
    let log = |p: f32| if p > 0.0 { f64::from(p).ln() } else { f64::NEG_INFINITY };
    let chosen = dist.iter().find(|e| e.0 == token).map_or(0.0, |e| e.1);
    TokenLogprobs {
        logprob: log(chosen),
        top: dist.iter().take(top_n.min(20)).map(|&(id, p)| (id as u32, log(p))).collect(),
    }
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

#[cfg(test)]
mod philox_tests {
    use super::*;

    #[test]
    fn philox_matches_the_random123_known_answers() {
        // kat_vectors of Random123 for philox4x32 with 10 rounds
        assert_eq!(philox4x32_10([0, 0, 0, 0], [0, 0]), [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]);
        assert_eq!(
            philox4x32_10([u32::MAX; 4], [u32::MAX; 2]),
            [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]
        );
        assert_eq!(
            philox4x32_10([0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344], [0xa409_3822, 0x299f_31d0]),
            [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1]
        );
    }

    #[test]
    fn philox_uniforms_are_open_unit_interval_and_flat() {
        let mut buckets = [0u32; 16];
        for i in 0..160_000u64 {
            let u = philox_uniform(42, i, 0);
            assert!(u > 0.0 && u < 1.0);
            buckets[(u * 16.0) as usize] += 1;
        }
        for b in buckets {
            assert!((9_300..=10_700).contains(&b), "bucket {b}");
        }
        assert_ne!(philox_uniform(1, 0, 0), philox_uniform(2, 0, 0));
        assert_ne!(philox_uniform(1, 0, 0), philox_uniform(1, 0, 1));
        assert_eq!(philox_uniform(7, 3, 9), philox_uniform(7, 3, 9));
    }

    #[test]
    fn philox_state_is_a_pure_function_of_the_draw_count() {
        let params = SamplingParams { temperature: 1.0, seed: Some(5), ..Default::default() };
        let logits = [0.1f32, 0.5, 0.9, 1.3, 0.2];
        let mut a = SamplingState::with_kind(&params, 1, RngKind::Philox);
        let first: Vec<i64> = (0..20).map(|_| sample_with_state(&logits, &[], &params, &mut a)).collect();
        // restarting at draw 10 reproduces the tail: no hidden state besides (seed, draws)
        let mut b = SamplingState::with_kind(&params, 99, RngKind::Philox);
        b.draws = 10;
        let tail: Vec<i64> = (0..10).map(|_| sample_with_state(&logits, &[], &params, &mut b)).collect();
        assert_eq!(&first[10..], &tail[..]);
    }

    #[test]
    fn processed_distribution_applies_top_k_top_p_and_min_p_in_order() {
        let logits = [3.0f32, 2.0, 1.0, 0.0, -1.0];
        let params = SamplingParams { temperature: 1.0, top_k: 3, ..Default::default() };
        let d = processed_distribution(&logits, &[], &params, 0.0);
        assert_eq!(d.iter().map(|e| e.0).collect::<Vec<_>>(), vec![0, 1, 2]);
        assert!((d.iter().map(|e| e.1).sum::<f32>() - 1.0).abs() < 1e-5);
        let params = SamplingParams { temperature: 1.0, top_p: 0.7, ..Default::default() };
        assert_eq!(processed_distribution(&logits, &[], &params, 0.0).len(), 2);
        let params = SamplingParams { temperature: 1.0, min_p: 0.3, ..Default::default() };
        assert_eq!(processed_distribution(&logits, &[], &params, 0.0).len(), 2);
        assert_eq!(processed_distribution(&logits, &[], &SamplingParams::default(), 0.0), vec![(0, 1.0)]);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Device sampler reference
// ---------------------------------------------------------------------------------------------------------------------------------

/// Result of sampling one row with the device sampler's algorithm.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceSample {
    /// -1 when no token is left.
    pub id: i32,
    pub logprob: f32,
    /// Best alternatives first: `(id, logprob)`.
    pub top: Vec<(i32, f32)>,
}

/// Order-preserving map of floats to unsigned integers (larger float, larger key); the same as `f2key` in cuda/sampling.cu.
#[inline]
pub fn float_key(f: f32) -> u32 {
    let u = f.to_bits();
    if u & 0x8000_0000 != 0 {
        !u
    } else {
        u | 0x8000_0000
    }
}

/// The largest key `t` such that the elements with key >= t hold at least `target` of the mass (0 when even all of them fall short):
/// the exact threshold that the kernel finds by bisection, here by sorting. `items` are `(key, mass)`.
fn mass_threshold(items: &mut [(u32, f64)], target: f64) -> u32 {
    items.sort_unstable_by_key(|item| std::cmp::Reverse(item.0));
    let mut total = 0.0f64;
    let mut i = 0;
    while i < items.len() {
        let key = items[i].0;
        while i < items.len() && items[i].0 == key {
            total += items[i].1;
            i += 1;
        }
        if total >= target {
            return key;
        }
    }
    0
}

/// What the CUDA sampler computes for one row (cuda/sampling.cu), on the host: the same steps and formulas, with thresholds found by
/// sorting. `pens` are the distinct tokens of the penalty window with their counts, `mask` a bit per token (set = allowed).
/// `raw_logits` stay untouched for the raw log-probabilities. Used by the CPU backend's `sample_rows` and by selftest-gpu as the
/// reference of the device.
pub fn device_sample_reference(
    raw_logits: &[f32],
    p: &crate::backend::SampleRowParams,
    pens: &[(i32, i32)],
    bias: &[(i32, f32)],
    mask: Option<&[u32]>,
    top_stride: usize,
) -> DeviceSample {
    let vocab = raw_logits.len();
    let sanitize = |v: f32| if v.is_finite() { v } else { f32::NEG_INFINITY };
    let mut w: Vec<f32> = raw_logits.iter().map(|&v| sanitize(v)).collect();
    let raw = w.clone();
    let raw_max = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let raw_lse = if p.logprobs_mode == 1 {
        raw.iter().map(|&v| f64::from((v - raw_max).exp())).sum::<f64>().ln()
    } else {
        0.0
    };
    for &(id, count) in pens {
        if id < 0 || id as usize >= vocab {
            continue;
        }
        let mut v = w[id as usize];
        if (p.repetition_penalty - 1.0).abs() >= 1e-5 {
            v = if v > 0.0 { v / p.repetition_penalty } else { v * p.repetition_penalty };
        }
        v -= p.presence_penalty + p.frequency_penalty * count as f32;
        w[id as usize] = v;
    }
    for &(id, value) in bias {
        if id >= 0 && (id as usize) < vocab {
            w[id as usize] += value;
        }
    }
    if let Some(bits) = mask {
        for (i, v) in w.iter_mut().enumerate() {
            if bits.get(i >> 5).is_none_or(|word| (word >> (i & 31)) & 1 == 0) {
                *v = f32::NEG_INFINITY;
            }
        }
    }
    let finite_max = |w: &[f32]| w.iter().copied().filter(|v| v.is_finite()).fold(f32::NEG_INFINITY, f32::max);
    let m = finite_max(&w);
    let top_n = if p.logprobs_mode == 0 { 0 } else { (p.top_n.max(0) as usize).min(20).min(top_stride) };
    let greedy = p.temperature <= 1e-5;
    let inv_t = if greedy { 1.0 } else { (1.0 / f64::from(p.temperature)) as f32 };
    if m == f32::NEG_INFINITY {
        return DeviceSample { id: -1, logprob: 0.0, top: Vec::new() };
    }
    let mut lnz = 0.0f64;
    let chosen = if greedy {
        let mut best = f32::NEG_INFINITY;
        let mut besti = 0usize;
        for (i, &v) in w.iter().enumerate() {
            if v > best {
                best = v;
                besti = i;
            }
        }
        besti
    } else {
        if p.top_k > 0 && (p.top_k as usize) < vocab {
            let mut keys: Vec<u32> = w.iter().map(|&v| float_key(v)).collect();
            keys.sort_unstable_by(|a, b| b.cmp(a));
            let t = keys[p.top_k as usize - 1];
            for v in w.iter_mut() {
                if float_key(*v) < t {
                    *v = f32::NEG_INFINITY;
                }
            }
        }
        let e_of = |v: f32| if v.is_finite() { ((v - m) * inv_t).exp() } else { 0.0 };
        let sum_e = |w: &[f32]| w.iter().map(|&v| f64::from(e_of(v))).sum::<f64>();
        let mut z = sum_e(&w);
        if p.min_p > 0.0 {
            for v in w.iter_mut() {
                if e_of(*v) < p.min_p {
                    *v = f32::NEG_INFINITY;
                }
            }
            z = sum_e(&w);
        }
        if p.typical_p < 1.0 {
            let entropy = w
                .iter()
                .map(|&v| e_of(v))
                .filter(|&e| e > 0.0)
                .map(|e| {
                    let pr = e / z as f32;
                    f64::from(-pr * pr.max(1e-30).ln())
                })
                .sum::<f64>() as f32;
            let key_of = |e: f32| -> u32 {
                if e <= 0.0 || e.is_nan() {
                    return 0;
                }
                let pr = e / z as f32;
                float_key(-(-pr.max(1e-30).ln() - entropy).abs())
            };
            let mut items: Vec<(u32, f64)> = w.iter().map(|&v| e_of(v)).filter(|&e| e > 0.0).map(|e| (key_of(e), f64::from(e))).collect();
            let t = mass_threshold(&mut items, f64::from(p.typical_p) * z);
            for v in w.iter_mut() {
                let e = e_of(*v);
                if e > 0.0 && key_of(e) < t {
                    *v = f32::NEG_INFINITY;
                }
            }
            z = sum_e(&w);
        }
        if p.top_p > 0.0 && p.top_p < 1.0 {
            let mut items: Vec<(u32, f64)> = w.iter().filter(|v| e_of(**v) > 0.0).map(|&v| (float_key(v), f64::from(e_of(v)))).collect();
            let t = mass_threshold(&mut items, f64::from(p.top_p) * z);
            for v in w.iter_mut() {
                if e_of(*v) > 0.0 && float_key(*v) < t {
                    *v = f32::NEG_INFINITY;
                }
            }
            z = sum_e(&w);
        }
        lnz = z.ln();
        let mut best = f32::NEG_INFINITY;
        let mut besti = usize::MAX;
        for (i, &v) in w.iter().enumerate() {
            if !v.is_finite() {
                continue;
            }
            let u = philox_uniform(p.seed, p.draw, i as u32);
            let g = -(-u.ln()).ln();
            let score = (v - m) * inv_t + g;
            if score > best {
                best = score;
                besti = i;
            }
        }
        besti
    };
    let logprob = match p.logprobs_mode {
        1 => raw[chosen] - raw_max - raw_lse as f32,
        2 if greedy => 0.0,
        2 => (w[chosen] - m) * inv_t - lnz as f32,
        _ => 0.0,
    };
    // alternatives: value descending, id ascending
    let mut top = Vec::new();
    if top_n > 0 {
        let base: &[f32] = if p.logprobs_mode == 1 { &raw } else { &w };
        let mut order: Vec<usize> = (0..vocab).filter(|&i| base[i].is_finite()).collect();
        order.sort_by(|&a, &b| base[b].total_cmp(&base[a]).then(a.cmp(&b)));
        for &i in order.iter().take(top_n) {
            let lp = match p.logprobs_mode {
                1 => base[i] - raw_max - raw_lse as f32,
                _ if greedy => {
                    if top.is_empty() {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                }
                _ => (base[i] - m) * inv_t - lnz as f32,
            };
            top.push((i as i32, lp));
        }
    }
    DeviceSample { id: chosen as i32, logprob, top }
}

/// The `(token, count)` pairs of the penalty window of `history` (the distinct tokens and how often each occurs), as the device
/// sampler wants them.
pub fn penalty_pairs(history: &[i64], repeat_last_n: i64, vocab: usize) -> Vec<(i32, i32)> {
    let window = history_window(history, repeat_last_n);
    let mut counts = std::collections::BTreeMap::<i32, i32>::new();
    for &t in window {
        if t >= 0 && (t as usize) < vocab {
            *counts.entry(t as i32).or_default() += 1;
        }
    }
    counts.into_iter().collect()
}

impl SamplingParams {
    /// Whether the device sampler (`Stream::sample_rows`) implements this request's sampling exactly. Mirostat and DRY depend on the
    /// sequence in ways that stay on the host, and the host sampler's own random draw differs from the device's Gumbel draw only
    /// in which stream of numbers it consumes, so every other request qualifies.
    pub fn device_samplable(&self) -> bool {
        self.mirostat == 0 && self.dry_multiplier <= 0.0
    }

    /// The device sampler row for a request about to draw number `draw` with the given penalty/bias list offsets.
    pub fn device_row(&self, seed: u64, draw: u64) -> crate::backend::SampleRowParams {
        crate::backend::SampleRowParams {
            temperature: self.temperature as f32,
            top_p: self.top_p as f32,
            min_p: self.min_p as f32,
            typical_p: self.typical_p as f32,
            top_k: if self.top_k == usize::MAX { 0 } else { self.top_k.min(i32::MAX as usize) as i32 },
            repetition_penalty: self.repetition_penalty,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            seed,
            draw,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod device_sampler_tests {
    use super::*;
    use crate::backend::SampleRowParams;

    fn logits(seed: u64, n: usize) -> Vec<f32> {
        let mut state = seed | 1;
        (0..n)
            .map(|_| {
                let u = (xorshift64(&mut state) >> 40) as f32 / 16_777_216.0;
                (u - 0.5) * 8.0
            })
            .collect()
    }

    fn row(seed: u64, draw: u64) -> SampleRowParams {
        SampleRowParams { temperature: 1.0, seed, draw, ..Default::default() }
    }

    #[test]
    fn float_keys_are_monotone_across_signs_and_infinities() {
        let values = [f32::NEG_INFINITY, -1e30, -2.0, -0.0, 0.0, 1e-30, 3.0, f32::INFINITY];
        for pair in values.windows(2) {
            assert!(float_key(pair[0]) <= float_key(pair[1]), "{pair:?}");
        }
        assert!(float_key(-1.0) < float_key(1.0));
    }

    #[test]
    fn greedy_rows_apply_penalties_bias_and_masks_before_the_argmax() {
        let l = vec![1.0, 5.0, 4.0, 3.0];
        let greedy = SampleRowParams::default();
        assert_eq!(device_sample_reference(&l, &greedy, &[], &[], None, 0).id, 1);
        // a presence penalty on token 1 hands the win to token 2
        let p = SampleRowParams { presence_penalty: 2.0, ..greedy };
        assert_eq!(device_sample_reference(&l, &p, &[(1, 1)], &[], None, 0).id, 2);
        // a bias on token 3
        assert_eq!(device_sample_reference(&l, &greedy, &[], &[(3, 10.0)], None, 0).id, 3);
        // a grammar bitmask allowing tokens 0 and 3 only
        assert_eq!(device_sample_reference(&l, &greedy, &[], &[], Some(&[0b1001]), 0).id, 3);
        // nothing allowed
        assert_eq!(device_sample_reference(&l, &greedy, &[], &[], Some(&[0]), 0).id, -1);
    }

    #[test]
    fn the_draw_depends_only_on_seed_draw_and_logits() {
        let l = logits(3, 500);
        let a = device_sample_reference(&l, &row(7, 3), &[], &[], None, 0);
        let b = device_sample_reference(&l, &row(7, 3), &[], &[], None, 0);
        assert_eq!(a, b);
        let ids: std::collections::HashSet<i32> = (0..40).map(|d| device_sample_reference(&l, &row(7, d), &[], &[], None, 0).id).collect();
        assert!(ids.len() > 5, "different draws must give different tokens");
    }

    #[test]
    fn samples_follow_the_processed_distribution_of_the_host_sampler() {
        // the device algorithm (Gumbel-max over thresholded logits) against `processed_distribution` (sorting and cumulative sums)
        let l = logits(11, 24);
        let configs = [
            (SamplingParams { temperature: 0.8, ..Default::default() }, row(1, 0)),
            (SamplingParams { temperature: 1.0, top_k: 6, ..Default::default() }, SampleRowParams { top_k: 6, ..row(1, 0) }),
            (SamplingParams { temperature: 1.0, top_p: 0.7, ..Default::default() }, SampleRowParams { top_p: 0.7, ..row(1, 0) }),
            (SamplingParams { temperature: 1.3, min_p: 0.1, ..Default::default() }, SampleRowParams { temperature: 1.3, min_p: 0.1, ..row(1, 0) }),
            (
                SamplingParams { temperature: 0.9, top_k: 12, top_p: 0.9, min_p: 0.02, ..Default::default() },
                SampleRowParams { temperature: 0.9, top_k: 12, top_p: 0.9, min_p: 0.02, ..row(1, 0) },
            ),
            (SamplingParams { temperature: 1.0, typical_p: 0.6, ..Default::default() }, SampleRowParams { typical_p: 0.6, ..row(1, 0) }),
        ];
        for (host, mut device) in configs {
            device.temperature = host.temperature as f32;
            let expected = processed_distribution(&l, &[], &host, 0.0);
            let mut counts = vec![0u32; l.len()];
            let trials = 40_000u64;
            for draw in 0..trials {
                let r = SampleRowParams { draw, ..device };
                let id = device_sample_reference(&l, &r, &[], &[], None, 0).id;
                counts[id as usize] += 1;
            }
            // every token outside the host sampler's support must never be drawn, and the support's frequencies must match
            let mut chi = 0.0;
            let mut cells = 0;
            for (id, &count) in counts.iter().enumerate() {
                let p = expected.iter().find(|e| e.0 == id).map_or(0.0, |e| f64::from(e.1));
                if p == 0.0 {
                    assert_eq!(count, 0, "token {id} is outside the support of {host:?}");
                } else {
                    let e = p * trials as f64;
                    if e > 5.0 {
                        chi += (f64::from(count) - e).powi(2) / e;
                        cells += 1;
                    }
                }
            }
            assert!(chi < 14.0 + 3.0 * f64::from(cells), "{host:?}: chi-square {chi} over {cells} cells");
        }
    }

    #[test]
    fn top_alternatives_and_log_probabilities_follow_the_requested_distribution() {
        let l = logits(5, 40);
        let raw = SampleRowParams { logprobs_mode: 1, top_n: 3, ..row(2, 0) };
        let r = device_sample_reference(&l, &raw, &[], &[], None, 4);
        assert_eq!(r.top.len(), 3);
        let host = raw_logprobs(&l, r.id as usize, 3).unwrap();
        assert!((f64::from(r.logprob) - host.logprob).abs() < 1e-4, "{} vs {}", r.logprob, host.logprob);
        for ((id, lp), (hid, hlp)) in r.top.iter().zip(&host.top) {
            assert_eq!(*id as u32, *hid);
            assert!((f64::from(*lp) - hlp).abs() < 1e-4);
        }
        // processed mode: the survivors of top-k 4 sum to probability one
        let processed = SampleRowParams { logprobs_mode: 2, top_n: 8, top_k: 4, ..row(2, 0) };
        let r = device_sample_reference(&l, &processed, &[], &[], None, 8);
        let finite: Vec<f32> = r.top.iter().map(|t| t.1).filter(|v| v.is_finite()).collect();
        assert_eq!(finite.len(), 4);
        assert!((finite.iter().map(|v| f64::from(*v).exp()).sum::<f64>() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn penalty_pairs_count_the_window() {
        assert_eq!(penalty_pairs(&[3, 1, 3, 3, 9], -1, 8), vec![(1, 1), (3, 3)]);
        assert_eq!(penalty_pairs(&[3, 1, 3, 3, 9], 2, 8), vec![(3, 1)]);
    }

    /// Writes cases for the CPU emulation of cuda/sampling.cu (a development check without a GPU, see docs/speculative-decoding.md):
    /// `FLLM_SAMPLER_CASES=/tmp/cases.txt cargo test --release --lib dump_device_sampler_cases -- --ignored`.
    #[test]
    #[ignore]
    fn dump_device_sampler_cases() {
        use std::fmt::Write as _;
        let path = std::env::var("FLLM_SAMPLER_CASES").expect("set FLLM_SAMPLER_CASES");
        let mut out = String::new();
        let mut cases = 0;
        let mut body = String::new();
        for (n, vocab) in [50usize, 1000, 4099].into_iter().enumerate() {
            for k in 0..120u64 {
                let l = logits(100 + k + 1000 * n as u64, vocab);
                let mut p = SampleRowParams { temperature: 0.5 + 0.1 * (k % 9) as f32, seed: 77 + k * 31, draw: k * 3, ..Default::default() };
                match k % 8 {
                    0 => p.temperature = 0.0,
                    1 => p.top_k = 1 + (k % 17) as i32,
                    2 => p.top_p = 0.3 + 0.1 * (k % 6) as f32,
                    3 => p.min_p = 0.02 * (1 + k % 7) as f32,
                    4 => p.typical_p = 0.5 + 0.05 * (k % 9) as f32,
                    5 => {
                        p.top_k = 30;
                        p.top_p = 0.9;
                        p.min_p = 0.01;
                    }
                    6 => {
                        p.repetition_penalty = 1.3;
                        p.presence_penalty = 0.4;
                        p.frequency_penalty = 0.2;
                    }
                    _ => {}
                }
                p.logprobs_mode = (k % 3) as i32;
                p.top_n = if p.logprobs_mode == 0 { 0 } else { 6 };
                let pens: Vec<(i32, i32)> = if k % 8 == 6 { (0..7).map(|i| ((i * 13 + k as usize) as i32 % vocab as i32, 1 + (i % 3) as i32)).collect::<std::collections::BTreeMap<_, _>>().into_iter().collect() } else { Vec::new() };
                let bias: Vec<(i32, f32)> = if k % 5 == 0 { vec![((k as i32 * 7) % vocab as i32, 2.5)] } else { Vec::new() };
                let words = vocab.div_ceil(32);
                let mask: Vec<u32> = if k % 7 == 3 { (0..words).map(|w| 0xA5A5_5A5Au32.rotate_left(w as u32 + k as u32)).collect() } else { Vec::new() };
                p.pen_len = pens.len() as i32;
                p.bias_len = bias.len() as i32;
                p.mask_row = if mask.is_empty() { -1 } else { 0 };
                let top_stride = 6;
                let want = device_sample_reference(&l, &p, &pens, &bias, (!mask.is_empty()).then_some(&mask[..]), top_stride);
                writeln!(body, "{vocab} {top_stride} {words}").unwrap();
                writeln!(
                    body,
                    "{:x} {:x} {:x} {:x} {} {:x} {:x} {:x} {} {} {} {} {}",
                    p.temperature.to_bits(), p.top_p.to_bits(), p.min_p.to_bits(), p.typical_p.to_bits(), p.top_k,
                    p.repetition_penalty.to_bits(), p.presence_penalty.to_bits(), p.frequency_penalty.to_bits(),
                    p.seed, p.draw, p.mask_row, p.logprobs_mode, p.top_n
                )
                .unwrap();
                writeln!(body, "{}", l.iter().map(|v| format!("{:x}", v.to_bits())).collect::<Vec<_>>().join(" ")).unwrap();
                writeln!(body, "{} {}", pens.len(), pens.iter().map(|(a, b)| format!("{a} {b}")).collect::<Vec<_>>().join(" ")).unwrap();
                writeln!(body, "{} {}", bias.len(), bias.iter().map(|(a, b)| format!("{a} {:x}", b.to_bits())).collect::<Vec<_>>().join(" ")).unwrap();
                writeln!(body, "{} {}", mask.len(), mask.iter().map(|w| format!("{w:x}")).collect::<Vec<_>>().join(" ")).unwrap();
                writeln!(
                    body,
                    "{} {:x} {} {}",
                    want.id,
                    want.logprob.to_bits(),
                    want.top.len(),
                    want.top.iter().map(|(a, b)| format!("{a} {:x}", b.to_bits())).collect::<Vec<_>>().join(" ")
                )
                .unwrap();
                cases += 1;
            }
        }
        writeln!(out, "{cases}").unwrap();
        out.push_str(&body);
        std::fs::write(path, out).unwrap();
    }
}

#[cfg(test)]
mod processed_logprob_tests {
    use super::*;

    #[test]
    fn processed_log_probabilities_describe_the_filtered_distribution() {
        let logits = [3.0f32, 2.0, 1.0, 0.0, -1.0];
        let params = SamplingParams { temperature: 1.0, top_k: 3, logprobs: Some(5), logprobs_processed: true, ..Default::default() };
        let dist = processed_distribution(&logits, &[], &params, 0.0);
        let lp = processed_logprobs(&dist, 1, 5);
        assert_eq!(lp.top.len(), 3, "only the survivors of top-k are listed");
        assert!((lp.top.iter().map(|t| t.1.exp()).sum::<f64>() - 1.0).abs() < 1e-6);
        assert!((lp.logprob - lp.top[1].1).abs() < 1e-12);
        // a token outside the support has probability zero
        assert_eq!(processed_logprobs(&dist, 4, 5).logprob, f64::NEG_INFINITY);
        // greedy: a point mass
        let greedy = processed_distribution(&logits, &[], &SamplingParams::default(), 0.0);
        assert_eq!(processed_logprobs(&greedy, 0, 3).top, vec![(0, 0.0)]);
    }
}
