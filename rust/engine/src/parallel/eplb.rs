//! Expert-parallel load balancing (EPLB) arithmetic: statistics of how often the router picks each expert, and a placement of experts on
//! ranks that spreads the load, replicating the hottest experts into spare slots (what vLLM and SGLang call redundant experts, and
//! TensorRT-LLM's wide-EP expert slots).
//!
//! This module is the planning half only: it turns counts into a placement and reports the imbalance before and after. The runtime half
//! (moving expert weights between ranks and routing a token to one replica) is not built; today the experts of a rank are the contiguous
//! range of `ParCtx::expert_range`. `contiguous_placement` reproduces that layout so the gain of a placement can be quoted against it.
use std::sync::atomic::{AtomicU64, Ordering};

/// Router decisions per layer and expert, counted lock-free from the ranks' threads.
pub struct ExpertLoad {
    layers: usize,
    experts: usize,
    counts: Vec<AtomicU64>,
}

impl ExpertLoad {
    pub fn new(layers: usize, experts: usize) -> Self {
        Self { layers, experts, counts: (0..layers * experts).map(|_| AtomicU64::new(0)).collect() }
    }

    /// One token routed to `chosen` experts in `layer`.
    pub fn record(&self, layer: usize, chosen: &[usize]) {
        for &e in chosen {
            if layer < self.layers && e < self.experts {
                self.counts[layer * self.experts + e].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    pub fn layer_counts(&self, layer: usize) -> Vec<u64> {
        self.counts[layer * self.experts..(layer + 1) * self.experts].iter().map(|c| c.load(Ordering::Relaxed)).collect()
    }

    /// Halve all counts: older traffic counts less (call once per statistics window).
    pub fn decay(&self) {
        for c in &self.counts {
            c.store(c.load(Ordering::Relaxed) / 2, Ordering::Relaxed);
        }
    }
}

/// Which experts every rank holds (an expert may appear on several ranks).
#[derive(Clone, Debug, PartialEq)]
pub struct Placement {
    pub ranks: Vec<Vec<usize>>,
}

impl Placement {
    pub fn replicas_of(&self, expert: usize) -> Vec<usize> {
        (0..self.ranks.len()).filter(|r| self.ranks[*r].contains(&expert)).collect()
    }

    /// Expected tokens per rank when each expert's traffic is split evenly over its replicas.
    pub fn rank_loads(&self, counts: &[u64]) -> Vec<f64> {
        let mut loads = vec![0.0; self.ranks.len()];
        for (e, &c) in counts.iter().enumerate() {
            let replicas = self.replicas_of(e);
            for &r in &replicas {
                loads[r] += c as f64 / replicas.len() as f64;
            }
        }
        loads
    }

    /// The replica of `expert` that serves a token (`token_hash` picks among the replicas; the same token always gets the same one).
    pub fn route(&self, expert: usize, token_hash: u64) -> Option<usize> {
        let replicas = self.replicas_of(expert);
        (!replicas.is_empty()).then(|| replicas[(token_hash % replicas.len() as u64) as usize])
    }
}

/// Busiest rank over the average rank (1.0 = perfectly balanced).
pub fn imbalance(loads: &[f64]) -> f64 {
    let mean = loads.iter().sum::<f64>() / loads.len().max(1) as f64;
    if mean <= 0.0 {
        return 1.0;
    }
    loads.iter().fold(0.0f64, |m, v| m.max(*v)) / mean
}

/// The layout the runtime uses today: expert `e` on the rank whose contiguous range contains it (first ranks take the remainder).
pub fn contiguous_placement(experts: usize, ranks: usize) -> Placement {
    Placement { ranks: (0..ranks).map(|r| crate::parallel::ParCtx::expert_range(experts, ranks, r).collect()).collect() }
}

/// Spread `counts` (router decisions per expert) over `ranks` ranks with `slots_per_rank` expert slots each. Spare slots
/// (`ranks * slots_per_rank - experts`) hold replicas of the experts with the highest load per replica; the replicas are then placed
/// longest-processing-time first on the least loaded rank that has a free slot and does not hold the expert yet.
pub fn plan_placement(counts: &[u64], ranks: usize, slots_per_rank: usize) -> Result<Placement, String> {
    let experts = counts.len();
    let slots = ranks * slots_per_rank;
    if ranks == 0 || experts == 0 {
        return Err("a placement needs at least one rank and one expert".into());
    }
    if slots < experts {
        return Err(format!("{ranks} ranks x {slots_per_rank} slots hold {slots} experts, the model has {experts}"));
    }
    let mut replicas = vec![1usize; experts];
    for _ in 0..slots - experts {
        // one more replica for the expert whose per-replica load is highest (an expert has at most one replica per rank)
        let next = (0..experts).filter(|e| replicas[*e] < ranks).max_by(|a, b| {
            let (la, lb) = (counts[*a] as f64 / replicas[*a] as f64, counts[*b] as f64 / replicas[*b] as f64);
            la.total_cmp(&lb).then(b.cmp(a))
        });
        match next {
            Some(e) => replicas[e] += 1,
            None => break,
        }
    }
    let mut items: Vec<(f64, usize)> = Vec::with_capacity(slots);
    for (e, &copies) in replicas.iter().enumerate() {
        items.extend(std::iter::repeat_n((counts[e] as f64 / copies as f64, e), copies));
    }
    items.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut placement = vec![Vec::<usize>::new(); ranks];
    let mut load = vec![0.0f64; ranks];
    for (weight, e) in items {
        let pick = (0..ranks)
            .filter(|r| placement[*r].len() < slots_per_rank && !placement[*r].contains(&e))
            .min_by(|a, b| load[*a].total_cmp(&load[*b]).then(a.cmp(b)))
            .or_else(|| (0..ranks).filter(|r| placement[*r].len() < slots_per_rank).min_by(|a, b| load[*a].total_cmp(&load[*b]).then(a.cmp(b))));
        let Some(r) = pick else { return Err("no free slot left (internal error)".into()) };
        placement[r].push(e);
        load[r] += weight;
    }
    for list in &mut placement {
        list.sort_unstable();
    }
    Ok(Placement { ranks: placement })
}

/// Summary line for a layer: imbalance of the contiguous layout and of the planned placement.
pub fn describe(counts: &[u64], ranks: usize, slots_per_rank: usize) -> Result<String, String> {
    let before = imbalance(&contiguous_placement(counts.len(), ranks).rank_loads(counts));
    let plan = plan_placement(counts, ranks, slots_per_rank)?;
    let after = imbalance(&plan.rank_loads(counts));
    Ok(format!("busiest rank carries {before:.2}x the average with the contiguous layout, {after:.2}x with {} spare slot(s) per rank", slots_per_rank - counts.len().div_ceil(ranks)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Zipf-like skew: expert e gets 1 / (e + 1) of the traffic, hot experts are neighbours (the worst case for contiguous ranges).
    fn skewed(experts: usize, total: f64) -> Vec<u64> {
        let norm: f64 = (0..experts).map(|e| 1.0 / (e as f64 + 1.0)).sum();
        (0..experts).map(|e| (total / norm / (e as f64 + 1.0)) as u64).collect()
    }

    #[test]
    fn counting_is_per_layer_and_decays() {
        let load = ExpertLoad::new(2, 4);
        load.record(0, &[1, 1, 3]);
        load.record(1, &[0]);
        load.record(5, &[0]); // unknown layer: ignored
        load.record(0, &[9]); // unknown expert: ignored
        assert_eq!(load.layer_counts(0), [0, 2, 0, 1]);
        assert_eq!(load.layer_counts(1), [1, 0, 0, 0]);
        load.decay();
        assert_eq!(load.layer_counts(0), [0, 1, 0, 0]);
    }

    #[test]
    fn every_expert_is_placed_and_slots_are_respected() {
        for (experts, ranks, slots) in [(256, 8, 32), (256, 8, 34), (128, 8, 20), (384, 8, 56), (8, 8, 1), (9, 4, 3)] {
            let counts = skewed(experts, 1e6);
            let p = plan_placement(&counts, ranks, slots).unwrap();
            assert_eq!(p.ranks.len(), ranks);
            for (r, list) in p.ranks.iter().enumerate() {
                assert!(list.len() <= slots, "rank {r}");
                let mut seen = list.clone();
                seen.dedup();
                assert_eq!(seen.len(), list.len(), "rank {r} holds an expert twice");
            }
            for e in 0..experts {
                assert!(!p.replicas_of(e).is_empty(), "expert {e} of {experts} has no rank");
            }
            let total: usize = p.ranks.iter().map(Vec::len).sum();
            assert!(total >= experts && total <= ranks * slots, "{experts}/{ranks}/{slots}: {total} slots used");
        }
        assert!(plan_placement(&[1; 10], 2, 4).is_err());
        assert!(plan_placement(&[], 2, 4).is_err());
    }

    #[test]
    fn a_skewed_load_is_much_better_balanced_than_contiguous_ranges() {
        let counts = skewed(256, 1e6);
        let contiguous = imbalance(&contiguous_placement(256, 8).rank_loads(&counts));
        let no_spare = imbalance(&plan_placement(&counts, 8, 32).unwrap().rank_loads(&counts));
        let spare = imbalance(&plan_placement(&counts, 8, 34).unwrap().rank_loads(&counts));
        assert!(contiguous > 2.0, "the contiguous layout puts the hot experts on rank 0: {contiguous}");
        assert!(no_spare < contiguous / 2.0, "{no_spare} vs {contiguous}");
        assert!(spare <= no_spare + 1e-9 && spare < 1.15, "replicas of the hottest experts: {spare}");
        // the hottest expert alone carries 1/H(256) = 16% of the traffic, more than one rank's fair share (12.5%): only replicas can fix that
        assert!(plan_placement(&counts, 8, 34).unwrap().replicas_of(0).len() > 1);
    }

    #[test]
    fn a_uniform_load_stays_balanced() {
        let counts = vec![1000u64; 128];
        let p = plan_placement(&counts, 8, 16).unwrap();
        assert!((imbalance(&p.rank_loads(&counts)) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn routing_picks_a_replica_deterministically() {
        let counts = skewed(16, 1e5);
        let p = plan_placement(&counts, 4, 6).unwrap();
        let replicas = p.replicas_of(0);
        assert!(replicas.len() > 1);
        for h in 0..50u64 {
            let r = p.route(0, h).unwrap();
            assert!(replicas.contains(&r));
            assert_eq!(p.route(0, h), Some(r));
        }
        assert_eq!(p.route(99, 1), None);
    }

    #[test]
    fn the_description_quotes_both_numbers() {
        let text = describe(&skewed(256, 1e6), 8, 34).unwrap();
        assert!(text.contains("contiguous") && text.contains("spare"), "{text}");
    }
}
