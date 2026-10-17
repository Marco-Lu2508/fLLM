//! Speculative decoding building blocks that do not depend on a model: the verification rules, the draft-length controller, the
//! statistics, and the suffix-matching draft source.
//!
//! Verification follows the speculative sampling scheme of Leviathan et al. ("Fast Inference from Transformers via Speculative
//! Decoding", ICML 2023) and Chen et al. ("Accelerating Large Language Model Decoding with Speculative Sampling", 2023). A draft
//! token `x` proposed with probability `q(x)` is accepted with probability `min(1, p(x) / q(x))` where `p` is the target
//! distribution after the same filters (temperature, top-k, top-p, min-p, penalties) the request uses without speculation. On
//! rejection the next token is drawn from `norm(max(0, p - q))`; when every draft token is accepted the bonus token is drawn from `p`
//! at the last position. The emitted sequence then has exactly the distribution of plain sampling from the target. Draft sources
//! without a probability (n-gram, suffix match, a greedy MTP head) have a point mass `q`, which reduces the rule to "accept with
//! probability `p(x)`, otherwise draw from `p` with `x` removed". A greedy request has a point-mass `p`, so the same code reduces to
//! "accept iff `x` is the argmax".

use crate::sampling::select_from;
use std::collections::{HashMap, VecDeque};

/// `(token, probability)` pairs that sum to one (see `sampling::processed_distribution`).
pub type Dist = Vec<(usize, f32)>;

fn prob_of(dist: &[(usize, f32)], token: usize) -> f32 {
    dist.iter().find(|e| e.0 == token).map_or(0.0, |e| e.1)
}

/// A token drawn from `dist`; a point mass consumes no random number.
fn draw(dist: &[(usize, f32)], uniform: &mut impl FnMut() -> f32) -> usize {
    if dist.len() == 1 {
        dist[0].0
    } else {
        select_from(dist, uniform()).0
    }
}

/// `norm(max(0, p - q))`; `q = None` is a point mass at `x`. When `p` and `q` coincide the residual is empty and `p` itself is used.
fn residual(p: &[(usize, f32)], q: Option<&[(usize, f32)]>, x: usize) -> Dist {
    let mut out = Dist::with_capacity(p.len());
    let mut sum = 0.0f64;
    match q {
        None => {
            for &(t, pr) in p {
                if t != x {
                    out.push((t, pr));
                    sum += f64::from(pr);
                }
            }
        }
        Some(q) => {
            let q: HashMap<usize, f32> = q.iter().copied().collect();
            for &(t, pr) in p {
                let r = pr - q.get(&t).copied().unwrap_or(0.0);
                if r > 0.0 {
                    out.push((t, r));
                    sum += f64::from(r);
                }
            }
        }
    }
    if sum <= 1e-12 {
        return p.to_vec();
    }
    for e in &mut out {
        e.1 = (f64::from(e.1) / sum) as f32;
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChainOutcome {
    /// How many draft tokens were accepted.
    pub accepted: usize,
    /// The token that follows them: the resampled token after a rejection, or the bonus token.
    pub next: i64,
}

/// Verify a chain of draft tokens. `target(i)` is the processed target distribution at draft position `i` (given the accepted
/// draft tokens before it; `i == draft.len()` is the bonus position) and is evaluated lazily, only for positions that are reached.
/// `q[i]` is the distribution `draft[i]` was drawn from, `None` for a point mass. `uniform` yields numbers in (0, 1).
pub fn verify_chain(
    draft: &[i64],
    q: &[Option<Dist>],
    mut target: impl FnMut(usize) -> Dist,
    mut uniform: impl FnMut() -> f32,
) -> ChainOutcome {
    for (i, &token) in draft.iter().enumerate() {
        let p = target(i);
        let x = token as usize;
        let qi = q.get(i).and_then(|q| q.as_deref());
        let px = prob_of(&p, x);
        let qx = qi.map_or(1.0, |q| prob_of(q, x));
        // qx == 0 cannot happen for a token drawn from q; treat such input as a point mass
        let (qi, qx) = if qx > 0.0 { (qi, qx) } else { (None, 1.0) };
        let accept = px >= qx || (px > 0.0 && uniform() * qx < px);
        if !accept {
            let rest = residual(&p, qi, x);
            return ChainOutcome { accepted: i, next: draw(&rest, &mut uniform) as i64 };
        }
    }
    let p = target(draft.len());
    ChainOutcome { accepted: draft.len(), next: draw(&p, &mut uniform) as i64 }
}

/// A node of a draft token tree. Nodes are listed so that a parent comes before its children; `parent == None` hangs the node below
/// the current token (the root, which is not part of the list).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TreeNode {
    pub parent: Option<usize>,
    pub token: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeOutcome {
    /// Accepted nodes from the root downwards.
    pub path: Vec<usize>,
    pub next: i64,
}

/// Verify a draft token tree whose candidates carry no probability (every sibling is a point mass, as for top-k heads of Medusa or
/// EAGLE, or for several suffix-match continuations). `target(node)` is the processed target distribution after the path that ends
/// at `node` (`None` = the root). Siblings are tried in list order, each against the target distribution with the rejected siblings
/// removed and renormalized (the rule of SpecInfer, Miao et al. 2024), which keeps the output distribution equal to plain sampling.
pub fn verify_tree(
    nodes: &[TreeNode],
    mut target: impl FnMut(Option<usize>) -> Dist,
    mut uniform: impl FnMut() -> f32,
) -> TreeOutcome {
    let mut current: Option<usize> = None;
    let mut path = Vec::new();
    loop {
        let mut dist = target(current);
        let mut descended = false;
        for (index, node) in nodes.iter().enumerate() {
            if node.parent != current {
                continue;
            }
            let x = node.token as usize;
            let px = prob_of(&dist, x);
            if px >= 1.0 || (px > 0.0 && uniform() < px) {
                path.push(index);
                current = Some(index);
                descended = true;
                break;
            }
            dist = residual(&dist, None, x);
        }
        if !descended {
            return TreeOutcome { path, next: draw(&dist, &mut uniform) as i64 };
        }
    }
}

/// Returns an M x M boolean visibility matrix where `mask[i * M + j]` is true iff node `j`
/// is an ancestor of node `i` in `nodes` (or `i == j`).
pub fn build_tree_mask(nodes: &[TreeNode]) -> Vec<bool> {
    let m = nodes.len();
    let mut mask = vec![false; m * m];
    for i in 0..m {
        mask[i * m + i] = true;
        let mut curr = nodes[i].parent;
        while let Some(p) = curr {
            if p < m {
                mask[i * m + p] = true;
                curr = nodes[p].parent;
            } else {
                break;
            }
        }
    }
    mask
}

/// Builds the attention bias for draft tree nodes with prefix context:
/// For each row i in [0..M], produces a bias row of length `prefix_len + M`.
/// Keys at position p < prefix_len are prefix tokens (bias 0.0f32).
/// Keys at position prefix_len + j are draft tree tokens (bias 0.0f32 if j is ancestor or self, else -INFINITY).
pub fn build_tree_attention_bias(nodes: &[TreeNode], prefix_len: usize) -> Vec<f32> {
    let m = nodes.len();
    let total_len = prefix_len + m;
    let mask = build_tree_mask(nodes);
    let mut bias = vec![0.0f32; m * total_len];
    for i in 0..m {
        let row_start = i * total_len;
        for j in 0..m {
            if !mask[i * m + j] {
                bias[row_start + prefix_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    bias
}

/// Given the accepted path from tree verification and the slots allocated for the tree nodes,
/// returns the slots along the surviving branch to retain in the KV cache, discarding rejected branches.
pub fn compact_tree_kv(outcome_path: &[usize], allocated_slots: &[usize]) -> Vec<usize> {
    outcome_path.iter().filter_map(|&idx| allocated_slots.get(idx).copied()).collect()
}

/// Generates a standard Medusa draft tree topology from candidate head top-k sizes.
/// Returns tree nodes and their (head_index, candidate_index) origins.
pub fn make_medusa_tree(head_topk: &[usize]) -> (Vec<TreeNode>, Vec<(usize, usize)>) {
    let mut nodes = Vec::new();
    let mut origins = Vec::new();
    let mut current_level = vec![None]; // None is root

    for (head_idx, &k) in head_topk.iter().enumerate() {
        let mut next_level = Vec::new();
        for &parent in &current_level {
            for cand_idx in 0..k {
                let node_idx = nodes.len();
                nodes.push(TreeNode {
                    parent,
                    token: 0,
                });
                origins.push((head_idx, cand_idx));
                next_level.push(Some(node_idx));
            }
        }
        current_level = next_level;
    }
    (nodes, origins)
}

/// Adaptive draft length: grows while the draft tokens are accepted, shrinks when they are not. The smoothed per-token acceptance
/// rate `a` gives the expected tokens per target pass `(1 - a^(k+1)) / (1 - a)`; the controller moves `k` by one towards the
/// best trade-off between that gain and the `k` extra rows the verification pass evaluates.
#[derive(Debug, Clone)]
pub struct AdaptiveDraft {
    pub min: usize,
    pub max: usize,
    current: usize,
    rate: f64,
    seen: u64,
}

impl AdaptiveDraft {
    pub fn new(min: usize, max: usize) -> Self {
        let max = max.max(1);
        let min = min.clamp(1, max);
        Self { min, max, current: 4.clamp(min, max), rate: 0.7, seen: 0 }
    }
    pub fn fixed(k: usize) -> Self {
        Self { min: k, max: k, current: k, rate: 0.7, seen: 0 }
    }
    pub fn length(&self) -> usize {
        self.current
    }
    pub fn rate(&self) -> f64 {
        self.rate
    }
    /// Record one verification of `proposed` draft tokens of which `accepted` survived.
    pub fn update(&mut self, proposed: usize, accepted: usize) {
        if proposed == 0 || self.min == self.max {
            return;
        }
        // a rejected token ends the chain, so the per-token rate is accepted / (accepted + rejected positions that were tested)
        let tested = (accepted + usize::from(accepted < proposed)) as f64;
        let observed = accepted as f64 / tested.max(1.0);
        let weight = if self.seen < 8 { 0.25 } else { 0.1 };
        self.rate += weight * (observed - self.rate);
        self.seen += 1;
        let gain = |k: usize| -> f64 {
            let a = self.rate.clamp(0.0, 0.999);
            (1.0 - a.powi(k as i32 + 1)) / (1.0 - a) / (1.0 + 0.08 * k as f64)
        };
        let k = self.current;
        if k < self.max && gain(k + 1) > gain(k) {
            self.current = k + 1;
        } else if k > self.min && gain(k - 1) >= gain(k) {
            self.current = k - 1;
        }
    }
}

/// Counters of one speculative method, for `/metrics` and the startup/shutdown log.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SpecStats {
    /// Verification steps (target passes that carried draft tokens).
    pub steps: u64,
    pub proposed: u64,
    pub accepted: u64,
    /// Tokens produced by those steps (accepted drafts + the correction/bonus token).
    pub emitted: u64,
    pub draft_ns: u64,
    pub verify_ns: u64,
}

impl SpecStats {
    pub fn add(&mut self, other: &SpecStats) {
        self.steps += other.steps;
        self.proposed += other.proposed;
        self.accepted += other.accepted;
        self.emitted += other.emitted;
        self.draft_ns += other.draft_ns;
        self.verify_ns += other.verify_ns;
    }
    pub fn record(&mut self, proposed: usize, accepted: usize) {
        self.steps += 1;
        self.proposed += proposed as u64;
        self.accepted += accepted as u64;
        self.emitted += accepted as u64 + 1;
    }
    pub fn acceptance_rate(&self) -> f64 {
        if self.proposed == 0 { 0.0 } else { self.accepted as f64 / self.proposed as f64 }
    }
    /// Tokens produced per verification step (1.0 = no gain).
    pub fn mean_emitted(&self) -> f64 {
        if self.steps == 0 { 0.0 } else { self.emitted as f64 / self.steps as f64 }
    }
    /// Mean accepted draft tokens per step.
    pub fn mean_accepted_length(&self) -> f64 {
        if self.steps == 0 { 0.0 } else { self.accepted as f64 / self.steps as f64 }
    }
    /// Estimated decode speedup over plain decoding: tokens per pass divided by the relative cost of a pass. The cost of the draft
    /// work is measured; the extra rows of the verification pass are charged at 3 % of a pass per draft token (decode is memory
    /// bound, so the rows are nearly free), a rough figure that `fllm-rs bench` replaces with a measurement.
    pub fn speedup_estimate(&self) -> f64 {
        if self.steps == 0 {
            return 1.0;
        }
        let rows = self.proposed as f64 / self.steps as f64;
        let draft_cost = if self.verify_ns > 0 { self.draft_ns as f64 / self.verify_ns as f64 } else { 0.0 };
        self.mean_emitted() / (1.0 + 0.03 * rows + draft_cost)
    }
}

/// The speculation methods selectable with `--speculative`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpeculativeMethod {
    Off,
    Ngram,
    Draft,
    Mtp,
    Eagle,
    Eagle3,
    Medusa,
    Suffix,
}

impl SpeculativeMethod {
    pub fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "off" | "none" => Self::Off,
            "ngram" | "lookup" | "prompt-lookup" => Self::Ngram,
            "draft" | "draft-model" => Self::Draft,
            "mtp" | "nextn" => Self::Mtp,
            "eagle" => Self::Eagle,
            "eagle3" | "eagle-3" => Self::Eagle3,
            "medusa" => Self::Medusa,
            "suffix" => Self::Suffix,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Ngram => "ngram",
            Self::Draft => "draft",
            Self::Mtp => "mtp",
            Self::Eagle => "eagle",
            Self::Eagle3 => "eagle3",
            Self::Medusa => "medusa",
            Self::Suffix => "suffix",
        }
    }
    /// Methods whose proposals this engine can run today. EAGLE, EAGLE-3 and Medusa need a draft head that consumes the target's hidden
    /// states and a tree-attention pass; they are rejected with an explanation instead of silently falling back.
    pub fn unsupported_reason(self) -> Option<&'static str> {
        None
    }
}

// ---------------------------------------------------------------------------------------------------------------------------
// Suffix matching across the conversation and the recent generations (SuffixDecoding, Oliaro et al. 2024, with hashed n-gram
// contexts in place of a suffix tree: the same longest-match-then-most-frequent-continuation rule, simpler memory behaviour).
// ---------------------------------------------------------------------------------------------------------------------------

const CONTEXT_ORDERS: [usize; 9] = [2, 3, 4, 6, 8, 12, 16, 24, 32];
const MAX_OCCURRENCES: usize = 8;

#[derive(Debug, Clone, Copy)]
pub struct SuffixConfig {
    /// Shortest context that may start a draft.
    pub min_match: usize,
    /// Tokens kept in the cache in total (live and archived documents).
    pub token_budget: usize,
}

impl Default for SuffixConfig {
    fn default() -> Self {
        Self { min_match: 3, token_budget: 1 << 20 }
    }
}

struct Doc {
    tokens: Vec<i64>,
    indexed: usize,
    live: bool,
}

/// Index of every document (running sequence or archived generation) by the hash of each context of a few lengths. A lookup takes the
/// longest suffix of the current context that occurs earlier, collects the continuations after the most recent occurrences, and follows
/// the continuation that most occurrences agree on token by token.
pub struct SuffixCache {
    cfg: SuffixConfig,
    docs: HashMap<u64, Doc>,
    archive: VecDeque<u64>,
    index: HashMap<(u8, u64), VecDeque<(u64, u32)>>,
    total: usize,
    next_doc: u64,
    by_owner: HashMap<i64, u64>,
}

/// `tokens` extends `old` (checked on the length and the last few tokens: the same sequence only ever grows, or restarts).
fn continues(old: &[i64], tokens: &[i64]) -> bool {
    let n = old.len();
    tokens.len() >= n && tokens[n.saturating_sub(8)..n] == old[n.saturating_sub(8)..]
}

fn context_hash(tokens: &[i64]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &t in tokens {
        h ^= t as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
        h ^= h >> 29;
    }
    h
}

impl SuffixCache {
    pub fn new(cfg: SuffixConfig) -> Self {
        Self { cfg, docs: HashMap::new(), archive: VecDeque::new(), index: HashMap::new(), total: 0, next_doc: 1, by_owner: HashMap::new() }
    }

    pub fn tokens_cached(&self) -> usize {
        self.total
    }

    /// Make the document of `owner` (a sequence id) equal to `tokens`, indexing what is new. A sequence that was rewound (a
    /// preemption or a different continuation) starts a new document.
    pub fn observe(&mut self, owner: i64, tokens: &[i64]) {
        let id = match self.by_owner.get(&owner) {
            Some(&id) if self.docs.get(&id).is_some_and(|d| continues(&d.tokens, tokens)) => id,
            _ => {
                if let Some(old) = self.by_owner.remove(&owner) {
                    self.archive_doc(old);
                }
                let id = self.next_doc;
                self.next_doc += 1;
                self.docs.insert(id, Doc { tokens: Vec::new(), indexed: 0, live: true });
                self.by_owner.insert(owner, id);
                id
            }
        };
        let doc = self.docs.get_mut(&id).unwrap();
        let from = doc.tokens.len();
        doc.tokens.extend_from_slice(&tokens[from..]);
        self.total += tokens.len() - from;
        // contexts ending at position `end` need a continuation token, so the last token is indexed by the next call
        let start = doc.indexed;
        let target = doc.tokens.len().saturating_sub(1).max(start);
        doc.indexed = target;
        let mut additions = Vec::new();
        for end in start..target {
            for (slot, &order) in CONTEXT_ORDERS.iter().enumerate() {
                if end + 1 >= order {
                    additions.push(((slot as u8, context_hash(&doc.tokens[end + 1 - order..=end])), end as u32));
                }
            }
        }
        for (key, pos) in additions {
            let list = self.index.entry(key).or_default();
            list.push_back((id, pos));
            if list.len() > MAX_OCCURRENCES * 4 {
                list.pop_front();
            }
        }
        self.evict();
    }

    /// Archive the documents of every owner that is not in `live` (sequences that finished or were cancelled).
    pub fn retain_live(&mut self, live: &std::collections::HashSet<i64>) {
        let gone: Vec<i64> = self.by_owner.keys().filter(|o| !live.contains(o)).copied().collect();
        for owner in gone {
            self.finish(owner);
        }
    }

    /// The sequence `owner` has finished: keep its document for later requests.
    pub fn finish(&mut self, owner: i64) {
        if let Some(id) = self.by_owner.remove(&owner) {
            self.archive_doc(id);
        }
    }

    fn archive_doc(&mut self, id: u64) {
        if let Some(doc) = self.docs.get_mut(&id) {
            doc.live = false;
            self.archive.push_back(id);
        }
    }

    fn evict(&mut self) {
        while self.total > self.cfg.token_budget {
            let Some(id) = self.archive.pop_front() else { break };
            if let Some(doc) = self.docs.remove(&id) {
                self.total -= doc.tokens.len();
            }
        }
        if self.index.len() > 8 * self.cfg.token_budget.max(1024) {
            // stale keys of evicted documents are filtered on lookup; rebuild when the table dwarfs the live data
            let docs = &self.docs;
            self.index.retain(|_, list| {
                list.retain(|(d, _)| docs.contains_key(d));
                !list.is_empty()
            });
        }
    }

    /// Up to `max_tokens` draft tokens that continue `context`, or an empty vector when no earlier occurrence of at least
    /// `min_match` tokens is known.
    pub fn propose(&self, owner: i64, context: &[i64], max_tokens: usize) -> Vec<i64> {
        if max_tokens == 0 || context.len() < self.cfg.min_match {
            return Vec::new();
        }
        let own = self.by_owner.get(&owner).copied();
        for (slot, &order) in CONTEXT_ORDERS.iter().enumerate().rev() {
            if order < self.cfg.min_match || context.len() < order {
                continue;
            }
            let key = (slot as u8, context_hash(&context[context.len() - order..]));
            let Some(list) = self.index.get(&key) else { continue };
            // most recent occurrences first
            let mut continuations: Vec<&[i64]> = Vec::new();
            for &(doc_id, pos) in list.iter().rev() {
                let Some(doc) = self.docs.get(&doc_id) else { continue };
                let pos = pos as usize;
                if pos + 1 > doc.tokens.len() || pos + 1 < order {
                    continue;
                }
                // verify the context (the hash may collide) and skip the current position of the sequence itself
                if doc.tokens[pos + 1 - order..=pos] != context[context.len() - order..] {
                    continue;
                }
                if Some(doc_id) == own && pos + 1 >= context.len() {
                    continue;
                }
                let rest = &doc.tokens[pos + 1..];
                if !rest.is_empty() {
                    continuations.push(&rest[..rest.len().min(max_tokens)]);
                }
                if continuations.len() >= MAX_OCCURRENCES {
                    break;
                }
            }
            if continuations.is_empty() {
                continue;
            }
            return follow_majority(&continuations, max_tokens);
        }
        Vec::new()
    }
}

/// Walk the continuations token by token, taking at each depth the token most of the still-matching continuations agree on (the most
/// recent one wins ties) and stopping when fewer than half of the original occurrences support it past the first token.
fn follow_majority(continuations: &[&[i64]], max_tokens: usize) -> Vec<i64> {
    let mut alive: Vec<usize> = (0..continuations.len()).collect();
    let mut out = Vec::new();
    for depth in 0..max_tokens {
        let mut counts: Vec<(i64, usize)> = Vec::new();
        for &i in &alive {
            if let Some(&t) = continuations[i].get(depth) {
                match counts.iter_mut().find(|c| c.0 == t) {
                    Some(c) => c.1 += 1,
                    None => counts.push((t, 1)),
                }
            }
        }
        let Some(&(best, support)) = counts.iter().rev().max_by_key(|c| c.1) else { break };
        if depth > 0 && support * 2 < continuations.len() {
            break;
        }
        out.push(best);
        alive.retain(|&i| continuations[i].get(depth) == Some(&best));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::{philox_uniform, processed_distribution, SamplingParams};

    fn uniform_stream(seed: u64) -> impl FnMut() -> f32 {
        let mut draws = 0u64;
        move || {
            let u = philox_uniform(seed, draws, 0);
            draws += 1;
            u
        }
    }

    /// A first-order toy language model over 5 tokens: `table[prev]` are logits.
    fn toy_logits(seed: u64, prev: usize) -> Vec<f32> {
        (0..5u64)
            .map(|t| {
                let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ ((prev as u64) << 20) ^ t;
                x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                x ^= x >> 31;
                (x % 1000) as f32 / 250.0
            })
            .collect()
    }

    fn dist(seed: u64, prev: usize, params: &SamplingParams) -> Dist {
        processed_distribution(&toy_logits(seed, prev), &[], params, 0.0)
    }

    fn chi_square(observed: &[u64], expected: &[f64]) -> f64 {
        let n: u64 = observed.iter().sum();
        observed
            .iter()
            .zip(expected)
            .filter(|(_, &e)| e * n as f64 > 5.0)
            .map(|(&o, &e)| {
                let e = e * n as f64;
                (o as f64 - e).powi(2) / e
            })
            .sum()
    }

    fn dense(d: &Dist) -> Vec<f64> {
        let mut v = vec![0.0; 5];
        for &(t, p) in d {
            v[t] = f64::from(p);
        }
        v
    }

    /// One speculation round starting from context token `last`: the draft model (`draft_seed`) proposes `k` tokens, the target
    /// (`target_seed`) verifies. Returns the emitted tokens.
    fn round(params: &SamplingParams, target_seed: u64, draft_seed: u64, last: usize, k: usize, rng: &mut impl FnMut() -> f32) -> Vec<usize> {
        let mut draft = Vec::new();
        let mut q = Vec::new();
        let mut prev = last;
        for _ in 0..k {
            let qd = dist(draft_seed, prev, params);
            let x = if qd.len() == 1 { qd[0].0 } else { select_from(&qd, rng()).0 };
            draft.push(x as i64);
            q.push(Some(qd));
            prev = x;
        }
        let drafted = draft.clone();
        let outcome = verify_chain(
            &draft,
            &q,
            |i| dist(target_seed, if i == 0 { last } else { drafted[i - 1] as usize }, params),
            &mut *rng,
        );
        let mut out: Vec<usize> = drafted[..outcome.accepted].iter().map(|&t| t as usize).collect();
        out.push(outcome.next as usize);
        out
    }

    fn check_distribution(params: &SamplingParams, draft_seed: u64, point_mass_draft: bool) {
        let target_seed = 11;
        let last = 2usize;
        let trials = 100_000;
        let mut first = vec![0u64; 5];
        let mut pairs = vec![0u64; 25];
        let mut rng = uniform_stream(0x5eed);
        for _ in 0..trials {
            let out = if point_mass_draft {
                // a deterministic draft (the draft model's argmax chain), as the n-gram and MTP sources do
                let mut draft = Vec::new();
                let mut prev = last;
                for _ in 0..3 {
                    let qd = processed_distribution(&toy_logits(draft_seed, prev), &[], &SamplingParams::default(), 0.0);
                    draft.push(qd[0].0 as i64);
                    prev = qd[0].0;
                }
                let d = draft.clone();
                let o = verify_chain(&draft, &[], |i| dist(target_seed, if i == 0 { last } else { d[i - 1] as usize }, params), &mut rng);
                let mut v: Vec<usize> = d[..o.accepted].iter().map(|&t| t as usize).collect();
                v.push(o.next as usize);
                v
            } else {
                round(params, target_seed, draft_seed, last, 3, &mut rng)
            };
            first[out[0]] += 1;
            if out.len() > 1 {
                pairs[out[0] * 5 + out[1]] += 1;
            } else {
                // the round ended after one token: draw the second token from the target so the joint can be compared
                let p = dist(target_seed, out[0], params);
                let x = if p.len() == 1 { p[0].0 } else { select_from(&p, rng()).0 };
                pairs[out[0] * 5 + x] += 1;
            }
        }
        let p0 = dense(&dist(target_seed, last, params));
        let stat = chi_square(&first, &p0);
        assert!(stat < 22.0, "first token: chi-square {stat} over the critical 18.5 (df 4, p = 0.001): {first:?} vs {p0:?}");
        let mut expected = vec![0.0; 25];
        for a in 0..5 {
            let pa = dense(&dist(target_seed, a, params));
            for b in 0..5 {
                expected[a * 5 + b] = p0[a] * pa[b];
            }
        }
        let stat = chi_square(&pairs, &expected);
        assert!(stat < 60.0, "token pairs: chi-square {stat} over the critical 51.2 (df 24, p = 0.001)");
    }

    #[test]
    fn sampled_output_matches_the_target_distribution_with_a_distribution_draft() {
        let params = SamplingParams { temperature: 0.9, ..Default::default() };
        check_distribution(&params, 99, false);
    }

    #[test]
    fn sampled_output_matches_the_target_under_top_k_top_p_and_min_p() {
        for params in [
            SamplingParams { temperature: 1.0, top_k: 3, ..Default::default() },
            SamplingParams { temperature: 0.8, top_p: 0.8, ..Default::default() },
            SamplingParams { temperature: 1.2, min_p: 0.15, top_k: 4, ..Default::default() },
        ] {
            check_distribution(&params, 5, false);
        }
    }

    #[test]
    fn the_chi_square_check_detects_a_verifier_that_accepts_everything() {
        // control: emitting the draft model's own samples is far from the target distribution
        let params = SamplingParams { temperature: 1.0, ..Default::default() };
        let mut rng = uniform_stream(3);
        let mut first = vec![0u64; 5];
        for _ in 0..20_000 {
            let qd = dist(99, 2, &params);
            first[select_from(&qd, rng()).0] += 1;
        }
        let stat = chi_square(&first, &dense(&dist(11, 2, &params)));
        assert!(stat > 100.0, "the target and draft distributions are too close for the test to mean anything ({stat})");
    }

    #[test]
    fn point_mass_drafts_keep_the_target_distribution() {
        let params = SamplingParams { temperature: 1.0, ..Default::default() };
        check_distribution(&params, 3, true);
    }

    #[test]
    fn greedy_verification_is_exact_and_consumes_no_random_numbers() {
        let params = SamplingParams::default();
        let target_seed = 4;
        let mut calls = 0;
        // draft = the target's own greedy continuation: everything is accepted
        let mut draft = Vec::new();
        let mut prev = 1usize;
        for _ in 0..4 {
            let t = dist(target_seed, prev, &params)[0].0;
            draft.push(t as i64);
            prev = t;
        }
        let d = draft.clone();
        let out = verify_chain(&draft, &[], |i| dist(target_seed, if i == 0 { 1 } else { d[i - 1] as usize }, &params), || {
            calls += 1;
            0.5
        });
        assert_eq!(out.accepted, 4);
        assert_eq!(out.next as usize, dist(target_seed, prev, &params)[0].0);
        // a wrong second token is rejected and replaced by the argmax
        let mut bad = d.clone();
        bad[1] = (bad[1] + 1) % 5;
        let b = bad.clone();
        let out = verify_chain(&bad, &[], |i| dist(target_seed, if i == 0 { 1 } else { b[i - 1] as usize }, &params), || {
            calls += 1;
            0.5
        });
        assert_eq!(out.accepted, 1);
        assert_eq!(out.next, d[1]);
        assert_eq!(calls, 0);
    }

    #[test]
    fn tree_verification_matches_the_target_distribution() {
        let params = SamplingParams { temperature: 1.0, ..Default::default() };
        let seed = 21;
        // a tree with two candidates below the root, and one grandchild below the first
        let nodes = [
            TreeNode { parent: None, token: 1 },
            TreeNode { parent: None, token: 3 },
            TreeNode { parent: Some(0), token: 0 },
        ];
        let token_after = |node: Option<usize>| match node {
            None => 2usize,
            Some(i) => nodes[i].token as usize,
        };
        let mut rng = uniform_stream(77);
        let mut first = vec![0u64; 5];
        let mut pairs = vec![0u64; 25];
        for _ in 0..100_000 {
            let o = verify_tree(&nodes, |n| dist(seed, token_after(n), &params), &mut rng);
            let mut seq: Vec<usize> = o.path.iter().map(|&i| nodes[i].token as usize).collect();
            seq.push(o.next as usize);
            first[seq[0]] += 1;
            let second = if seq.len() > 1 {
                seq[1]
            } else {
                let p = dist(seed, seq[0], &params);
                select_from(&p, rng()).0
            };
            pairs[seq[0] * 5 + second] += 1;
        }
        let p0 = dense(&dist(seed, 2, &params));
        assert!(chi_square(&first, &p0) < 22.0, "tree first token: {first:?} vs {p0:?}");
        let mut expected = vec![0.0; 25];
        for a in 0..5 {
            let pa = dense(&dist(seed, a, &params));
            for b in 0..5 {
                expected[a * 5 + b] = p0[a] * pa[b];
            }
        }
        let stat = chi_square(&pairs, &expected);
        assert!(stat < 60.0, "tree token pairs: chi-square {stat}");
    }

    #[test]
    fn residual_of_equal_distributions_falls_back_to_the_target() {
        let p = vec![(0, 0.5), (1, 0.5)];
        assert_eq!(residual(&p, Some(&p), 0), p);
        let r = residual(&p, None, 0);
        assert_eq!(r, vec![(1, 1.0)]);
    }

    #[test]
    fn adaptive_length_grows_on_acceptance_and_shrinks_on_rejection() {
        let mut a = AdaptiveDraft::new(1, 8);
        for _ in 0..40 {
            let k = a.length();
            a.update(k, k);
        }
        assert_eq!(a.length(), 8);
        for _ in 0..60 {
            let k = a.length();
            a.update(k, 0);
        }
        assert_eq!(a.length(), 1);
        let mut f = AdaptiveDraft::fixed(5);
        f.update(5, 0);
        assert_eq!(f.length(), 5);
    }

    #[test]
    fn stats_report_acceptance_and_mean_length() {
        let mut s = SpecStats::default();
        s.record(4, 3);
        s.record(4, 1);
        assert_eq!(s.acceptance_rate(), 0.5);
        assert_eq!(s.mean_accepted_length(), 2.0);
        assert_eq!(s.mean_emitted(), 3.0);
        assert!(s.speedup_estimate() > 1.0 && s.speedup_estimate() < 3.0);
    }

    #[test]
    fn methods_parse_and_unsupported_ones_explain_themselves() {
        for (text, method) in [("off", SpeculativeMethod::Off), ("ngram", SpeculativeMethod::Ngram), ("draft", SpeculativeMethod::Draft), ("mtp", SpeculativeMethod::Mtp), ("eagle3", SpeculativeMethod::Eagle3), ("suffix", SpeculativeMethod::Suffix)] {
            assert_eq!(SpeculativeMethod::parse(text), Some(method));
            assert_eq!(SpeculativeMethod::parse(method.name()), Some(method));
        }
        assert!(SpeculativeMethod::parse("bogus").is_none());
        assert!(SpeculativeMethod::Medusa.unsupported_reason().is_none());
        assert!(SpeculativeMethod::Eagle.unsupported_reason().is_none());
        assert!(SpeculativeMethod::Eagle3.unsupported_reason().is_none());
        assert!(SpeculativeMethod::Suffix.unsupported_reason().is_none());
    }

    #[test]
    fn suffix_cache_finds_the_longest_earlier_match_across_documents() {
        let mut cache = SuffixCache::new(SuffixConfig::default());
        // an earlier generation, archived
        let old: Vec<i64> = vec![10, 11, 12, 13, 14, 15, 16, 17, 18, 19];
        cache.observe(1, &old);
        cache.finish(1);
        // the live conversation repeats part of it
        let live: Vec<i64> = vec![5, 6, 12, 13, 14];
        cache.observe(2, &live);
        assert_eq!(cache.propose(2, &live, 4), vec![15, 16, 17, 18]);
        // too short a context: nothing
        assert!(cache.propose(2, &[13, 14], 4).is_empty());
        // no match
        assert!(cache.propose(2, &[40, 41, 42, 43], 4).is_empty());
    }

    #[test]
    fn suffix_cache_follows_the_majority_continuation_and_ignores_its_own_tail() {
        let mut cache = SuffixCache::new(SuffixConfig::default());
        for (owner, tail) in [(1, [9, 9]), (2, [9, 9]), (3, [7, 7])] {
            let mut doc = vec![1, 2, 3];
            doc.extend_from_slice(&tail);
            cache.observe(owner, &doc);
            cache.finish(owner);
        }
        let live = vec![0, 1, 2, 3];
        cache.observe(4, &live);
        assert_eq!(cache.propose(4, &live, 3), vec![9, 9]);
        // the tail of the live sequence itself never matches itself
        let mut solo = SuffixCache::new(SuffixConfig::default());
        let doc = vec![1, 2, 3, 4, 5, 1, 2, 3];
        solo.observe(1, &doc);
        assert_eq!(solo.propose(1, &doc, 3), vec![4, 5, 1]);
    }

    #[test]
    fn suffix_cache_indexes_incrementally_and_respects_the_budget() {
        let mut cache = SuffixCache::new(SuffixConfig { min_match: 3, token_budget: 40 });
        let mut doc: Vec<i64> = Vec::new();
        for chunk in 0..6 {
            doc.extend((0..6).map(|i| (chunk * 100 + i) as i64));
            cache.observe(1, &doc);
        }
        assert_eq!(cache.tokens_cached(), 36);
        cache.finish(1);
        let other: Vec<i64> = (500..530).collect();
        cache.observe(2, &other);
        assert_eq!(cache.tokens_cached(), other.len());
        // the archived document was evicted to make room
        assert!(cache.propose(2, &[100, 101, 102], 3).is_empty());
        // incremental observation of a growing live document keeps matching
        let mut live = vec![1i64, 2, 3, 4, 5, 6];
        let mut c2 = SuffixCache::new(SuffixConfig::default());
        c2.observe(1, &live);
        live.extend([1, 2, 3]);
        c2.observe(1, &live);
        assert_eq!(c2.propose(1, &live, 3), vec![4, 5, 6]);
    }

    #[test]
    fn tree_attention_masks_and_compaction() {
        let nodes = vec![
            TreeNode { parent: None, token: 10 },
            TreeNode { parent: Some(0), token: 20 },
            TreeNode { parent: None, token: 30 },
        ];
        let mask = build_tree_mask(&nodes);
        assert_eq!(mask.len(), 9);
        assert!(mask[0]);
        assert!(!mask[1]);
        assert!(!mask[2]);
        assert!(mask[3]);
        assert!(mask[4]);
        assert!(!mask[5]);
        assert!(!mask[6]);
        assert!(!mask[7]);
        assert!(mask[8]);

        let bias = build_tree_attention_bias(&nodes, 4);
        assert_eq!(bias.len(), 3 * 7);
        for r in 0..3 {
            for p in 0..4 {
                assert_eq!(bias[r * 7 + p], 0.0);
            }
        }
        assert_eq!(bias[11], 0.0);
        assert_eq!(bias[13], f32::NEG_INFINITY);

        let outcome_path = vec![0, 1];
        let allocated_slots = vec![100, 101, 102];
        let compacted = compact_tree_kv(&outcome_path, &allocated_slots);
        assert_eq!(compacted, vec![100, 101]);

        let (tree_nodes, origins) = make_medusa_tree(&[2, 2]);
        assert_eq!(tree_nodes.len(), 6);
        assert_eq!(origins.len(), 6);
    }
}