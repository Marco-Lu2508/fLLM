//! Which projections are worth running on the NPU, and for how many rows.
//!
//! The NPU wins where a projection is compute bound (prefill: many prompt rows per weight read) and loses where it is memory bound
//! (decode: one row per weight read; the NPU shares DRAM with the CPU, and every call costs a fixed submission latency). The rules
//! here are therefore:
//!
//! * a projection is a candidate if its format converts ([`super::quant::target_of`]) and it has at least `min_elems` weights;
//! * the compiled models have a *static* number of input rows (`chunk`); `batch` rows are cut into full chunks plus a remainder that
//!   is padded to a chunk, and the remainder only goes to the NPU if it has at least `min_batch` rows (else the CPU does it);
//! * a chunk goes to the NPU if its estimated time (call latency + FLOPs / NPU rate) is below the CPU's estimate.
//!
//! The three rates are rounded from measurements on one Core Ultra 5 235U (defaults: CPU 200 GOPS, NPU 1300 GOPS, 550 us per call; OPENVINO.md section 0)
//! and can be replaced with `FLLM_NPU_CPU_GOPS`, `FLLM_NPU_GOPS` and `FLLM_NPU_OVERHEAD_US`.
//!
//! Settings from the environment: `FLLM_NPU=0` disables the offload, `FLLM_NPU_MIN_BATCH`, `FLLM_NPU_CHUNK`, `FLLM_NPU_MIN_ELEMS`,
//! `FLLM_NPU_BUDGET_MB` (weight memory held by compiled models, 0 = limited only by the device), `FLLM_NPU_GROUP` (group size of
//! re-quantized weights; setting it also turns the exact Q4_0 / Q8_0 conversion off; 0 = channel-wise).
//!
//! Verification level: L0 (pure arithmetic, unit tested; the cost model has no measured basis).

use super::quant::{target_of, QuantPolicy, Target};
use crate::backend::QuantType;
use std::sync::OnceLock;

/// Settings of the offload decision.
#[derive(Clone, Debug)]
pub struct OffloadPlan {
    pub enabled: bool,
    /// Fewest rows of a remainder chunk worth a call.
    pub min_batch: usize,
    /// Static number of input rows of the compiled models.
    pub chunk: usize,
    /// Fewest weight elements of a projection worth compiling.
    pub min_elems: usize,
    /// Most bytes of weights the compiled models may hold in total (0 = no limit of our own).
    pub budget_bytes: u64,
    pub quant: QuantPolicy,
    /// Assumed effective rates (see the module documentation).
    pub cpu_gops: f64,
    pub npu_gops: f64,
    pub call_overhead_us: f64,
}

impl Default for OffloadPlan {
    fn default() -> Self {
        Self {
            enabled: true,
            min_batch: 64,
            chunk: 256,
            min_elems: 1 << 20,
            budget_bytes: 0,
            quant: QuantPolicy::int8_channel(),
            cpu_gops: 200.0,
            npu_gops: 1300.0,
            call_overhead_us: 550.0,
        }
    }
}

fn env_number<T: std::str::FromStr>(name: &str) -> Option<T> {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
}

impl OffloadPlan {
    /// The defaults overridden by the `FLLM_NPU_*` variables.
    pub fn from_env() -> Self {
        let mut plan = Self::default();
        if let Ok(value) = std::env::var("FLLM_NPU") {
            plan.enabled = !matches!(value.trim(), "0" | "off" | "false" | "no");
        }
        if let Some(v) = env_number::<usize>("FLLM_NPU_MIN_BATCH") {
            plan.min_batch = v.max(1);
        }
        if let Some(v) = env_number::<usize>("FLLM_NPU_CHUNK") {
            plan.chunk = v.max(1);
        }
        if let Some(v) = env_number::<usize>("FLLM_NPU_MIN_ELEMS") {
            plan.min_elems = v;
        }
        if let Some(v) = env_number::<u64>("FLLM_NPU_BUDGET_MB") {
            plan.budget_bytes = v.saturating_mul(1 << 20);
        }
        // FLLM_NPU_WEIGHTS: int8 (default, per channel), int4 (INT4 in groups of 128) or exact (Q4_0/Q8_0 bit for bit, groups of 32)
        match std::env::var("FLLM_NPU_WEIGHTS").ok().as_deref().map(|v| v.trim().to_ascii_lowercase()).as_deref() {
            Some("int4") => plan.quant = QuantPolicy::int4_grouped(),
            Some("exact") => plan.quant = QuantPolicy::default(),
            _ => {}
        }
        if let Some(v) = env_number::<usize>("FLLM_NPU_GROUP") {
            plan.quant = QuantPolicy {
                group: v,
                lossless: false,
                eight: plan.quant.eight,
            };
        }
        if let Some(v) = env_number::<f64>("FLLM_NPU_CPU_GOPS") {
            plan.cpu_gops = v.max(0.001);
        }
        if let Some(v) = env_number::<f64>("FLLM_NPU_GOPS") {
            plan.npu_gops = v.max(0.001);
        }
        if let Some(v) = env_number::<f64>("FLLM_NPU_OVERHEAD_US") {
            plan.call_overhead_us = v.max(0.0);
        }
        plan.min_batch = plan.min_batch.min(plan.chunk);
        plan
    }

    /// Whether a projection of this shape and format is a candidate at all, whatever the batch.
    pub fn eligible(&self, rows: usize, cols: usize, quant: QuantType) -> bool {
        self.enabled
            && target_of(quant).is_some()
            && rows >= 16
            && cols >= 32
            && rows.checked_mul(cols).is_some_and(|n| n >= self.min_elems)
    }

    /// How many of `batch` rows go to the NPU: all full chunks, plus the remainder if it has `min_batch` rows or more.
    pub fn npu_rows(&self, batch: usize) -> usize {
        let full = batch / self.chunk * self.chunk;
        let rest = batch - full;
        full + if rest >= self.min_batch { rest } else { 0 }
    }

    /// Estimated microseconds of one chunk on the CPU and on the NPU.
    pub fn chunk_times_us(&self, rows: usize, cols: usize) -> (f64, f64) {
        let ops = 2.0 * self.chunk as f64 * rows as f64 * cols as f64;
        (
            ops / (self.cpu_gops * 1e3),
            self.call_overhead_us + ops / (self.npu_gops * 1e3),
        )
    }

    /// The decision for one call: does any row of this `batch` go to the NPU for this projection?
    pub fn should_offload(&self, batch: usize, rows: usize, cols: usize, quant: QuantType) -> bool {
        if !self.eligible(rows, cols, quant) || self.npu_rows(batch) == 0 {
            return false;
        }
        let (cpu, npu) = self.chunk_times_us(rows, cols);
        npu < cpu
    }

    /// Bytes the NPU holds for a projection (codes and scales) under this plan's conversion policy.
    pub fn resident_bytes(&self, rows: usize, cols: usize, quant: QuantType) -> u64 {
        let (rows, cols) = (rows as u64, cols as u64);
        let group = |g: usize| {
            if g == 0 || g as u64 >= cols {
                cols
            } else {
                g as u64
            }
        };
        match target_of(quant) {
            None => 0,
            Some(Target::Float) => rows * cols * 2,
            Some(target) => {
                let exact =
                    self.quant.lossless && matches!(quant, QuantType::Q4_0 | QuantType::Q8_0);
                let g = if exact { 32 } else { group(self.quant.group) };
                let scales = rows * (cols / g.max(1)) * 2;
                let target = if target == Target::Four && self.quant.eight && !exact { Target::Eight } else { target };
                if target == Target::Four {
                    rows * cols / 2 + scales
                } else {
                    rows * cols + scales
                }
            }
        }
    }

    /// One line for the log.
    pub fn describe(&self) -> String {
        format!(
            "NPU offload {}: chunk {} rows, remainder from {} rows, projections from {} weights, weights {}, assumed rates cpu {} / npu {} GOPS, {} us per call",
            if self.enabled { "on" } else { "off" },
            self.chunk,
            self.min_batch,
            self.min_elems,
            self.quant.describe(),
            self.cpu_gops,
            self.npu_gops,
            self.call_overhead_us
        )
    }
}

/// The plan configured by the environment (read once).
pub fn global_plan() -> &'static OffloadPlan {
    static PLAN: OnceLock<OffloadPlan> = OnceLock::new();
    PLAN.get_or_init(OffloadPlan::from_env)
}

/// `true` when at least part of a `batch`-row product with a `rows x cols` weight of format `quant` should run on the NPU, under the
/// plan from the environment. This is the hook a host decoder calls before it computes a projection on the CPU.
pub fn should_offload(batch: usize, rows: usize, cols: usize, quant: QuantType) -> bool {
    global_plan().should_offload(batch, rows, cols, quant)
}

/// A projection of a model, for [`plan`].
#[derive(Clone, Debug)]
pub struct WeightShape {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub quant: QuantType,
}

/// Decision for one projection.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanEntry {
    pub name: String,
    pub offload: bool,
    /// Bytes the NPU would hold for it.
    pub npu_bytes: u64,
    pub reason: &'static str,
}

/// Choose the projections to compile for a model whose prompts have about `prompt_rows` rows, within `budget_bytes` of NPU weight
/// memory (`0` = the plan's own budget, and no limit if that is 0 too). Projections are taken by estimated time saved per byte.
pub fn plan(
    shapes: &[WeightShape],
    prompt_rows: usize,
    settings: &OffloadPlan,
    budget_bytes: u64,
) -> Vec<PlanEntry> {
    let budget = if budget_bytes > 0 {
        budget_bytes
    } else {
        settings.budget_bytes
    };
    let mut entries: Vec<PlanEntry> = shapes
        .iter()
        .map(|s| PlanEntry {
            name: s.name.clone(),
            offload: false,
            npu_bytes: settings.resident_bytes(s.rows, s.cols, s.quant),
            reason: "",
        })
        .collect();
    let mut candidates: Vec<(usize, f64)> = Vec::new();
    for (index, s) in shapes.iter().enumerate() {
        entries[index].reason = if !settings.enabled {
            "NPU offload disabled"
        } else if target_of(s.quant).is_none() {
            "format has no NPU conversion"
        } else if !settings.eligible(s.rows, s.cols, s.quant) {
            "projection too small"
        } else if settings.npu_rows(prompt_rows) == 0 {
            "prompt too short for the NPU"
        } else {
            let (cpu, npu) = settings.chunk_times_us(s.rows, s.cols);
            if npu >= cpu {
                "estimated slower than the CPU"
            } else {
                let chunks = (settings.npu_rows(prompt_rows) as f64 / settings.chunk as f64).ceil();
                candidates.push((
                    index,
                    (cpu - npu) * chunks / entries[index].npu_bytes.max(1) as f64,
                ));
                "candidate"
            }
        };
    }
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut used = 0u64;
    for (index, _) in candidates {
        let bytes = entries[index].npu_bytes;
        if budget > 0 && used + bytes > budget {
            entries[index].reason = "over the NPU weight budget";
            continue;
        }
        used += bytes;
        entries[index].offload = true;
        entries[index].reason = "offloaded";
    }
    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_is_split_into_chunks_and_a_remainder() {
        let plan = OffloadPlan {
            chunk: 128,
            min_batch: 64,
            ..OffloadPlan::default()
        };
        assert_eq!(plan.npu_rows(1), 0);
        assert_eq!(plan.npu_rows(63), 0);
        assert_eq!(plan.npu_rows(64), 64);
        assert_eq!(plan.npu_rows(100), 100);
        assert_eq!(plan.npu_rows(128), 128);
        assert_eq!(
            plan.npu_rows(150),
            128,
            "a remainder of 22 rows stays on the CPU"
        );
        assert_eq!(plan.npu_rows(200), 200);
        assert_eq!(plan.npu_rows(1024), 1024);
    }

    #[test]
    fn decode_never_offloads_and_large_prefill_does() {
        let plan = OffloadPlan::default();
        assert!(
            !plan.should_offload(1, 4096, 4096, QuantType::Q4_0),
            "decode"
        );
        assert!(!plan.should_offload(8, 4096, 4096, QuantType::Q4_0));
        assert!(
            plan.should_offload(256, 4096, 4096, QuantType::Q4_0),
            "prefill"
        );
        assert!(plan.should_offload(256, 4096, 4096, QuantType::Q4_K));
        assert!(plan.should_offload(256, 4096, 4096, QuantType::F16));
        assert!(
            !plan.should_offload(256, 8, 4096, QuantType::Q4_0),
            "too few rows"
        );
        assert!(
            !plan.should_offload(256, 256, 256, QuantType::Q4_0),
            "too small"
        );
        assert!(
            !plan.should_offload(256, 4096, 4096, QuantType::Nvfp4Marlin),
            "device-only layout"
        );
        let off = OffloadPlan {
            enabled: false,
            ..OffloadPlan::default()
        };
        assert!(!off.should_offload(256, 4096, 4096, QuantType::Q4_0));
    }

    #[test]
    fn call_overhead_can_make_the_cpu_win() {
        let slow_npu = OffloadPlan {
            npu_gops: 100.0,
            ..OffloadPlan::default()
        };
        assert!(
            !slow_npu.should_offload(256, 4096, 4096, QuantType::Q4_0),
            "npu slower than cpu"
        );
        let big_latency = OffloadPlan {
            call_overhead_us: 1.0e9,
            ..OffloadPlan::default()
        };
        assert!(!big_latency.should_offload(256, 4096, 4096, QuantType::Q4_0));
    }

    #[test]
    fn resident_bytes_follow_the_conversion_policy() {
        let exact = OffloadPlan { quant: QuantPolicy::default(), ..OffloadPlan::default() };
        // the default plan holds every integer format as INT8 with one scale per row
        let fast = OffloadPlan::default();
        assert_eq!(fast.resident_bytes(4096, 4096, QuantType::Q4_0), 4096 * 4096 + 4096 * 2);
        assert_eq!(fast.resident_bytes(4096, 4096, QuantType::Q4_K), 4096 * 4096 + 4096 * 2);
        // Q4_0: 4096x4096 -> 8 MiB of codes + 4096 * 128 scales * 2 bytes
        assert_eq!(
            exact.resident_bytes(4096, 4096, QuantType::Q4_0),
            4096 * 4096 / 2 + 4096 * 128 * 2
        );
        assert_eq!(
            exact.resident_bytes(4096, 4096, QuantType::Q8_0),
            4096 * 4096 + 4096 * 128 * 2
        );
        assert_eq!(
            exact.resident_bytes(4096, 4096, QuantType::Q4_K),
            4096 * 4096 / 2 + 4096 * 32 * 2
        );
        assert_eq!(
            exact.resident_bytes(4096, 4096, QuantType::F32),
            4096 * 4096 * 2
        );
        assert_eq!(exact.resident_bytes(4096, 4096, QuantType::Other(1)), 0);
        let cw = OffloadPlan {
            quant: QuantPolicy {
                group: 0,
                lossless: false,
                eight: false,
            },
            ..OffloadPlan::default()
        };
        assert_eq!(
            cw.resident_bytes(4096, 4096, QuantType::Q4_0),
            4096 * 4096 / 2 + 4096 * 2
        );
    }

    #[test]
    fn model_plan_respects_the_budget_and_explains_refusals() {
        let shapes: Vec<WeightShape> = [
            ("attn_q", 4096, 4096),
            ("ffn_up", 14336, 4096),
            ("tiny", 64, 64),
            ("head", 128256, 4096),
        ]
        .into_iter()
        .map(|(name, rows, cols)| WeightShape {
            name: name.to_owned(),
            rows,
            cols,
            quant: QuantType::Q4_0,
        })
        .collect();
        let settings = OffloadPlan::default();
        let unlimited = plan(&shapes, 512, &settings, 0);
        assert!(unlimited[0].offload && unlimited[1].offload && unlimited[3].offload);
        assert!(!unlimited[2].offload);
        assert_eq!(unlimited[2].reason, "projection too small");
        let budget = settings.resident_bytes(4096, 4096, QuantType::Q4_0)
            + settings.resident_bytes(14336, 4096, QuantType::Q4_0);
        let limited = plan(&shapes, 512, &settings, budget);
        assert!(limited[0].offload && limited[1].offload);
        assert!(!limited[3].offload);
        assert_eq!(limited[3].reason, "over the NPU weight budget");
        let short = plan(&shapes, 10, &settings, 0);
        assert!(short.iter().all(|e| !e.offload));
        assert_eq!(short[0].reason, "prompt too short for the NPU");
    }

    #[test]
    fn describe_mentions_the_assumptions() {
        let text = OffloadPlan::default().describe();
        assert!(text.contains("assumed rates"), "{text}");
    }
}
