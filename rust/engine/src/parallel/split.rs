//! The layer-split planner: which layers go to which device when one model is spread over several devices (pipeline stages).
//!
//! The planner is pure arithmetic over byte counts, so that it can be tested with many device-count / size combinations without any
//! hardware or model file. `backend::memory::plan_pipeline` fills `ModelSizes` from a GGUF index and `serve` fills `SplitDevice`
//! from the device inventory.
//!
//! * Every stage gets a contiguous range of layers (at least one). Stage 0 also embeds (the table stays in host memory), the last
//!   stage also holds the output head.
//! * The target share of the weights of a stage is proportional to its `speed` (memory bandwidth, or the user's `--tensor-split`
//!   ratio) and limited by what its memory budget can hold next to the KV cache of one full-context sequence. Devices of equal speed
//!   therefore receive layers in proportion to their memory.
//! * A local search then moves layer boundaries to the best of (a) all stages fit, (b) the common KV pool is as large as the context
//!   wants, (c) the slowest stage (weights / speed) is as fast as possible.
//! * The KV pool is one block table shared by all stages: its block count is the smallest any stage can hold.
use std::ops::Range;

/// One device that takes part in the split.
#[derive(Clone, Debug, PartialEq)]
pub struct SplitDevice {
    /// For messages: `cuda:0 (NVIDIA RTX 2000 Ada)`.
    pub label: String,
    /// Memory this device may use (free memory times the utilisation; for the CPU the free host memory).
    pub budget_bytes: u64,
    /// Relative speed of reading weights (memory bandwidth in GB/s, or a ratio); only the proportions matter.
    pub speed: f64,
}

/// What the planner needs to know about the model, in device-layout bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelSizes {
    /// Weights of each layer.
    pub layer_bytes: Vec<u64>,
    /// KV cache bytes of each layer for one block (`block_size` tokens); 0 for layers without a cache.
    pub kv_block_bytes: Vec<u64>,
    /// Output head and final norm, held by the last stage.
    pub head_bytes: u64,
    /// Runtime overhead, scratch and driver context of every stage.
    pub fixed_bytes: u64,
    /// Blocks every stage must be able to hold (one sequence of the full context).
    pub min_kv_blocks: u64,
    /// Blocks the planner tries to reach (the pool the context asks for); more adds nothing to the score.
    pub wanted_kv_blocks: u64,
}

/// The result for one device.
#[derive(Clone, Debug, PartialEq)]
pub struct StageShare {
    pub device: usize,
    pub layers: Range<usize>,
    pub weight_bytes: u64,
    /// Output head included for the last stage.
    pub head_bytes: u64,
    pub kv_block_bytes: u64,
    pub fixed_bytes: u64,
    pub budget_bytes: u64,
    /// KV blocks this stage alone could hold next to its weights.
    pub max_kv_blocks: u64,
}

impl StageShare {
    /// Bytes of the stage at a pool of `blocks` KV blocks.
    pub fn bytes_at(&self, blocks: u64) -> u64 {
        self.weight_bytes + self.head_bytes + self.fixed_bytes + self.kv_block_bytes * blocks
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SplitPlan {
    pub stages: Vec<StageShare>,
    /// Blocks of the common KV pool: the smallest `max_kv_blocks` of any stage, at most `wanted_kv_blocks`.
    pub kv_blocks: u64,
}

impl SplitPlan {
    pub fn ranges(&self) -> Vec<Range<usize>> {
        self.stages.iter().map(|s| s.layers.clone()).collect()
    }
}

/// How well a set of ranges does; compared lexicographically, smaller is better.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Score {
    /// Bytes by which stages exceed their budget at the minimum pool (0 = everything fits).
    deficit: u64,
    /// Blocks short of what the context wants.
    block_shortfall: u64,
    /// Time of the slowest stage (bytes read per token over speed).
    makespan: f64,
}

impl Score {
    fn better_than(&self, other: &Score) -> bool {
        (self.deficit, self.block_shortfall) < (other.deficit, other.block_shortfall)
            || ((self.deficit, self.block_shortfall) == (other.deficit, other.block_shortfall) && self.makespan < other.makespan * (1.0 - 1e-9))
    }
}

fn stage_share(devices: &[SplitDevice], sizes: &ModelSizes, stage: usize, layers: Range<usize>, last: bool) -> StageShare {
    let weight_bytes: u64 = sizes.layer_bytes[layers.clone()].iter().sum();
    let kv_block_bytes: u64 = sizes.kv_block_bytes[layers.clone()].iter().sum();
    let head_bytes = if last { sizes.head_bytes } else { 0 };
    let used = weight_bytes + head_bytes + sizes.fixed_bytes;
    let room = devices[stage].budget_bytes.saturating_sub(used);
    let max_kv_blocks = room.checked_div(kv_block_bytes).unwrap_or(u64::MAX);
    StageShare {
        device: stage,
        layers,
        weight_bytes,
        head_bytes,
        kv_block_bytes,
        fixed_bytes: sizes.fixed_bytes,
        budget_bytes: devices[stage].budget_bytes,
        max_kv_blocks,
    }
}

fn shares(devices: &[SplitDevice], sizes: &ModelSizes, bounds: &[usize]) -> Vec<StageShare> {
    let n = devices.len();
    (0..n).map(|s| stage_share(devices, sizes, s, bounds[s]..bounds[s + 1], s + 1 == n)).collect()
}

fn score(devices: &[SplitDevice], sizes: &ModelSizes, stages: &[StageShare]) -> Score {
    let mut deficit = 0u64;
    let mut blocks = u64::MAX;
    let mut makespan = 0.0f64;
    for (s, share) in stages.iter().enumerate() {
        let need = share.bytes_at(sizes.min_kv_blocks);
        deficit += need.saturating_sub(share.budget_bytes);
        blocks = blocks.min(share.max_kv_blocks);
        let speed = devices[s].speed.max(1e-9);
        makespan = makespan.max((share.weight_bytes + share.head_bytes) as f64 / speed);
    }
    Score { deficit, block_shortfall: sizes.wanted_kv_blocks.saturating_sub(blocks), makespan }
}

/// Initial boundaries: cumulative weight targets proportional to the speeds, each stage limited by what it can hold, every stage with
/// at least one layer.
fn initial_bounds(devices: &[SplitDevice], sizes: &ModelSizes) -> Vec<usize> {
    let n = devices.len();
    let layers = sizes.layer_bytes.len();
    let total: f64 = sizes.layer_bytes.iter().map(|&b| b as f64).sum::<f64>() + sizes.head_bytes as f64;
    // what a stage can hold next to its fixed share and the KV of the minimum pool, per layer-average
    let avg_kv: f64 = if layers == 0 { 0.0 } else { sizes.kv_block_bytes.iter().sum::<u64>() as f64 / layers as f64 };
    let caps: Vec<f64> = devices
        .iter()
        .enumerate()
        .map(|(s, d)| {
            let head = if s + 1 == n { sizes.head_bytes as f64 } else { 0.0 };
            // the KV cache scales with the layers the stage holds; the average layer is a good enough estimate for the first guess
            let avg_w = if layers == 0 { 1.0 } else { sizes.layer_bytes.iter().sum::<u64>() as f64 / layers as f64 };
            let per_layer = avg_w + avg_kv * sizes.min_kv_blocks as f64;
            let free = (d.budget_bytes as f64 - sizes.fixed_bytes as f64 - head).max(0.0);
            if per_layer <= 0.0 {
                free
            } else {
                free / per_layer * avg_w
            }
        })
        .collect();
    // water-filling: proportional to speed, a saturated device keeps its cap and the rest is shared by the others
    let mut target = vec![0.0f64; n];
    let mut open: Vec<usize> = (0..n).collect();
    let mut remaining = total;
    while !open.is_empty() && remaining > 0.0 {
        let speed_sum: f64 = open.iter().map(|&i| devices[i].speed.max(1e-9)).sum();
        let mut saturated = Vec::new();
        for &i in &open {
            let want = remaining * devices[i].speed.max(1e-9) / speed_sum;
            if want > caps[i] - target[i] {
                saturated.push(i);
            }
        }
        if saturated.is_empty() {
            for &i in &open {
                target[i] += remaining * devices[i].speed.max(1e-9) / speed_sum;
            }
            remaining = 0.0;
        } else {
            for &i in &saturated {
                remaining -= caps[i] - target[i];
                target[i] = caps[i];
            }
            open.retain(|i| !saturated.contains(i));
        }
    }
    if remaining > 0.0 {
        // does not fit anywhere: spread the rest by speed, the search and the error message take over
        let speed_sum: f64 = devices.iter().map(|d| d.speed.max(1e-9)).sum();
        for (i, d) in devices.iter().enumerate() {
            target[i] += remaining * d.speed.max(1e-9) / speed_sum;
        }
    }
    // cumulative layer boundaries closest to the cumulative targets (the head counts for the last stage)
    let mut bounds = vec![0usize];
    let mut cum_target = 0.0;
    let mut cum_layers = 0.0f64;
    let mut at = 0usize;
    for (s, t) in target.iter().enumerate().take(n - 1) {
        cum_target += t;
        // advance while the middle of the next layer still lies below the target (round to the nearest boundary)
        while at < layers {
            let w = sizes.layer_bytes[at] as f64;
            if cum_layers + w / 2.0 <= cum_target {
                cum_layers += w;
                at += 1;
            } else {
                break;
            }
        }
        // leave one layer to each remaining stage and take at least one for this one
        let min_at = bounds[s] + 1;
        let max_at = layers - (n - 1 - s);
        bounds.push(at.clamp(min_at, max_at));
        at = *bounds.last().unwrap_or(&0);
        cum_layers = sizes.layer_bytes[..at].iter().map(|&b| b as f64).sum();
    }
    bounds.push(layers);
    bounds
}

/// Splits the layers of a model over `devices` (in pipeline order). Fails when the model cannot fit even with the best split, or when
/// there are more devices than layers.
pub fn plan_split(devices: &[SplitDevice], sizes: &ModelSizes) -> Result<SplitPlan, String> {
    let n = devices.len();
    let layers = sizes.layer_bytes.len();
    if n == 0 {
        return Err("no device to split the model over".into());
    }
    if sizes.kv_block_bytes.len() != layers {
        return Err("internal error: the layer size tables differ in length".into());
    }
    if layers < n {
        return Err(format!("the model has {layers} layers, which {n} devices cannot share (every device needs at least one layer)"));
    }
    let mut bounds = initial_bounds(devices, sizes);
    let mut best = score(devices, sizes, &shares(devices, sizes, &bounds));
    // local search: move one boundary by up to a few layers while that improves the score
    let mut improved = true;
    let mut guard = 0;
    while improved && guard < 4 * layers.max(8) {
        improved = false;
        guard += 1;
        for b in 1..n {
            for step in [-3i64, -2, -1, 1, 2, 3] {
                let moved = bounds[b] as i64 + step;
                if moved <= bounds[b - 1] as i64 || moved >= bounds[b + 1] as i64 {
                    continue;
                }
                let mut trial = bounds.clone();
                trial[b] = moved as usize;
                let s = score(devices, sizes, &shares(devices, sizes, &trial));
                if s.better_than(&best) {
                    best = s;
                    bounds = trial;
                    improved = true;
                }
            }
        }
    }
    let stages = shares(devices, sizes, &bounds);
    if best.deficit > 0 {
        let mut lines = Vec::new();
        for (s, share) in stages.iter().enumerate() {
            let need = share.bytes_at(sizes.min_kv_blocks);
            if need > share.budget_bytes {
                lines.push(format!(
                    "{} needs {:.2} GB for layers {}..{} but may use {:.2} GB",
                    devices[s].label,
                    need as f64 / 1e9,
                    share.layers.start,
                    share.layers.end,
                    share.budget_bytes as f64 / 1e9
                ));
            }
        }
        let total_budget: f64 = devices.iter().map(|d| d.budget_bytes as f64).sum();
        let total_need = sizes.layer_bytes.iter().sum::<u64>() as f64
            + sizes.head_bytes as f64
            + (sizes.fixed_bytes * n as u64) as f64
            + sizes.kv_block_bytes.iter().sum::<u64>() as f64 * sizes.min_kv_blocks as f64;
        return Err(format!(
            "the model does not fit on the {n} devices: it needs about {:.2} GB (weights, one full-context KV sequence and {n} runtime reserves), they may use {:.2} GB together ({}); \
             use a smaller quantization, --ctx, --kv-cache-type q8_0, or add a device (the CPU can take the last stage)",
            total_need / 1e9,
            total_budget / 1e9,
            lines.join("; ")
        ));
    }
    let kv_blocks = stages.iter().map(|s| s.max_kv_blocks).min().unwrap_or(0).min(sizes.wanted_kv_blocks);
    Ok(SplitPlan { stages, kv_blocks })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;

    fn dev(label: &str, gb: f64, speed: f64) -> SplitDevice {
        SplitDevice { label: label.to_owned(), budget_bytes: (gb * GB as f64) as u64, speed }
    }

    fn uniform(layers: usize, layer_gb: f64, kv_mb_per_block_layer: u64, head_gb: f64) -> ModelSizes {
        ModelSizes {
            layer_bytes: vec![(layer_gb * GB as f64) as u64; layers],
            kv_block_bytes: vec![kv_mb_per_block_layer * 1_000_000 / 1000; layers],
            head_bytes: (head_gb * GB as f64) as u64,
            fixed_bytes: GB / 2,
            min_kv_blocks: 256,
            wanted_kv_blocks: 2048,
        }
    }

    fn check_covers(plan: &SplitPlan, layers: usize) {
        let mut next = 0;
        for stage in &plan.stages {
            assert_eq!(stage.layers.start, next, "ranges are contiguous");
            assert!(!stage.layers.is_empty(), "every stage has a layer");
            next = stage.layers.end;
        }
        assert_eq!(next, layers, "all layers are assigned");
    }

    #[test]
    fn equal_devices_share_equally_and_odd_counts_differ_by_one() {
        for (n, layers) in [(2, 32), (3, 32), (4, 32), (4, 33), (5, 80), (8, 61), (2, 3), (3, 3)] {
            let devices: Vec<_> = (0..n).map(|i| dev(&format!("d{i}"), 24.0, 900.0)).collect();
            let plan = plan_split(&devices, &uniform(layers, 0.2, 2, 0.0)).unwrap();
            check_covers(&plan, layers);
            let lens: Vec<usize> = plan.stages.iter().map(|s| s.layers.len()).collect();
            let (min, max) = (lens.iter().min().unwrap(), lens.iter().max().unwrap());
            assert!(max - min <= 1, "n={n} layers={layers}: {lens:?}");
        }
    }

    #[test]
    fn layers_follow_the_speed_when_memory_is_plentiful() {
        let devices = vec![dev("fast", 48.0, 900.0), dev("slow", 48.0, 300.0)];
        let plan = plan_split(&devices, &uniform(40, 0.25, 2, 0.3)).unwrap();
        check_covers(&plan, 40);
        let (a, b) = (plan.stages[0].layers.len(), plan.stages[1].layers.len());
        assert!((28..=32).contains(&a), "fast device holds about three quarters: {a} vs {b}");
    }

    #[test]
    fn a_small_device_gets_what_it_can_hold() {
        // 8 GB card next to a 24 GB card at the same speed: layers in proportion to memory, nothing over budget
        let devices = vec![dev("small", 8.0, 500.0), dev("big", 24.0, 500.0)];
        let sizes = uniform(48, 0.4, 4, 0.5);
        let plan = plan_split(&devices, &sizes).unwrap();
        check_covers(&plan, 48);
        for s in &plan.stages {
            assert!(s.bytes_at(sizes.min_kv_blocks) <= s.budget_bytes);
        }
        assert!(plan.stages[0].layers.len() < plan.stages[1].layers.len());
        assert!(plan.stages[1].layers.len() >= 30, "{:?}", plan.ranges());
    }

    #[test]
    fn a_memory_limited_fast_device_is_capped() {
        // the fast device has little memory: speed alone would give it 3/4 of the model, memory allows 12 layers
        let devices = vec![dev("fast-small", 6.0, 1000.0), dev("slow-big", 40.0, 250.0)];
        let sizes = uniform(40, 0.4, 2, 0.4);
        let plan = plan_split(&devices, &sizes).unwrap();
        check_covers(&plan, 40);
        assert!(plan.stages[0].bytes_at(sizes.min_kv_blocks) <= plan.stages[0].budget_bytes);
        assert!(plan.stages[0].layers.len() <= 13, "{:?}", plan.ranges());
    }

    #[test]
    fn the_last_stage_pays_for_the_output_head() {
        let devices = vec![dev("a", 10.0, 500.0), dev("b", 10.0, 500.0)];
        let with_head = plan_split(&devices, &uniform(30, 0.25, 2, 3.0)).unwrap();
        let without = plan_split(&devices, &uniform(30, 0.25, 2, 0.0)).unwrap();
        assert!(with_head.stages[1].layers.len() < without.stages[1].layers.len() || with_head.stages[1].layers.len() < with_head.stages[0].layers.len());
        assert_eq!(with_head.stages[1].head_bytes, 3 * GB);
        assert_eq!(with_head.stages[0].head_bytes, 0);
    }

    #[test]
    fn the_pool_is_the_smallest_any_stage_holds_and_is_capped_by_the_wish() {
        let devices = vec![dev("a", 4.0, 500.0), dev("b", 4.0, 500.0)];
        let mut sizes = uniform(20, 0.12, 40, 0.1);
        sizes.wanted_kv_blocks = 100_000;
        let plan = plan_split(&devices, &sizes).unwrap();
        let smallest = plan.stages.iter().map(|s| s.max_kv_blocks).min().unwrap();
        assert_eq!(plan.kv_blocks, smallest);
        sizes.wanted_kv_blocks = 300;
        let capped = plan_split(&devices, &sizes).unwrap();
        assert_eq!(capped.kv_blocks, 300);
    }

    #[test]
    fn a_model_that_does_not_fit_is_refused_with_the_numbers() {
        let devices = vec![dev("a", 2.0, 500.0), dev("b", 2.0, 500.0)];
        let err = plan_split(&devices, &uniform(40, 0.4, 2, 0.4)).unwrap_err();
        assert!(err.contains("does not fit on the 2 devices"), "{err}");
        assert!(err.contains("may use"), "{err}");
    }

    #[test]
    fn more_devices_than_layers_and_no_devices_are_errors() {
        let sizes = uniform(3, 0.1, 1, 0.1);
        assert!(plan_split(&[], &sizes).is_err());
        let devices: Vec<_> = (0..4).map(|i| dev(&format!("d{i}"), 8.0, 100.0)).collect();
        assert!(plan_split(&devices, &sizes).unwrap_err().contains("every device needs at least one layer"));
    }

    #[test]
    fn a_single_device_takes_everything() {
        let plan = plan_split(&[dev("only", 24.0, 500.0)], &uniform(32, 0.3, 2, 0.4)).unwrap();
        assert_eq!(plan.ranges(), vec![0..32]);
    }

    #[test]
    fn uneven_layers_are_split_by_bytes_not_by_count() {
        // the first four layers are four times as large (dense layers in front of small expert layers)
        let mut sizes = uniform(24, 0.1, 1, 0.1);
        for b in sizes.layer_bytes.iter_mut().take(4) {
            *b *= 4;
        }
        let devices = vec![dev("a", 16.0, 500.0), dev("b", 16.0, 500.0)];
        let plan = plan_split(&devices, &sizes).unwrap();
        let (wa, wb) = (plan.stages[0].weight_bytes, plan.stages[1].weight_bytes + plan.stages[1].head_bytes);
        let ratio = wa as f64 / wb as f64;
        assert!((0.8..1.25).contains(&ratio), "{ratio} {:?}", plan.ranges());
        assert!(plan.stages[0].layers.len() < 12);
    }

    #[test]
    fn many_combinations_cover_all_layers_and_respect_the_budgets() {
        let mut checked = 0;
        for n in 1..=8usize {
            for layers in [n, n + 1, 24, 61, 80] {
                if layers < n {
                    continue;
                }
                for mem_spread in [1.0f64, 2.0, 5.0] {
                    for speed_spread in [1.0f64, 3.0, 10.0] {
                        let devices: Vec<_> = (0..n)
                            .map(|i| dev(&format!("d{i}"), 30.0 * mem_spread.powf(i as f64 / n as f64), 100.0 * speed_spread.powf((n - i) as f64 / n as f64)))
                            .collect();
                        let sizes = uniform(layers, 0.15, 3, 0.3);
                        let plan = plan_split(&devices, &sizes).unwrap_or_else(|e| panic!("n={n} layers={layers}: {e}"));
                        check_covers(&plan, layers);
                        for s in &plan.stages {
                            assert!(s.bytes_at(sizes.min_kv_blocks) <= s.budget_bytes, "n={n} layers={layers} {:?}", plan.ranges());
                        }
                        assert!(plan.kv_blocks >= sizes.min_kv_blocks);
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 100);
    }

    #[test]
    fn a_cpu_stage_can_take_the_tail_when_the_gpu_is_full() {
        // 6 GB GPU and 64 GB of host memory at a tenth of the speed: the GPU keeps what fits, the host takes the rest
        let devices = vec![dev("gpu", 6.0, 400.0), dev("cpu", 64.0, 40.0)];
        let sizes = uniform(40, 0.4, 2, 0.5);
        let plan = plan_split(&devices, &sizes).unwrap();
        check_covers(&plan, 40);
        assert!(plan.stages[0].layers.len() >= 8, "{:?}", plan.ranges());
        assert!(plan.stages[1].layers.len() > plan.stages[0].layers.len() || plan.stages[0].layers.len() > 8);
        assert!(plan.stages[0].bytes_at(sizes.min_kv_blocks) <= plan.stages[0].budget_bytes);
    }
}
