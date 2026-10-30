//! Command-line / environment description of the parallel layout.
use crate::comm::tcp::RendezvousConfig;
use crate::comm::BackendKind;
use std::ops::Range;
use std::sync::Mutex;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpDispatch {
    /// Every rank computes its own experts for all tokens it holds; the partial outputs are summed with the all-reduce of the layer.
    AllReduce,
    /// Tokens are routed to the ranks that own their experts with an all-to-all and the results come back the same way.
    AllToAll,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParallelConfig {
    pub tp: usize,
    pub pp: usize,
    /// Experts are distributed over the tensor-parallel group instead of every expert being split.
    pub expert_parallel: bool,
    pub ep_dispatch: EpDispatch,
    /// Device ordinals of this node's ranks (`--devices 0,1,2,3`); empty = the first `local_world_size()` visible ones.
    pub devices: Vec<usize>,
    /// `None`: NCCL when a GPU backend runs and libnccl is installed, otherwise the peer-to-peer fallback; `local` on the CPU.
    pub backend: Option<BackendKind>,
    pub rendezvous: RendezvousConfig,
    /// Micro-batches of a pipeline step (0 = one per stage when the batch has enough sequences).
    pub microbatches: usize,
    /// Split the embedding table by vocabulary rows and sum the lookups (`FLLM_VOCAB_PARALLEL_EMBED`); off by default because the
    /// table is memory-mapped and shared by all ranks of the process anyway.
    pub vocab_parallel_embedding: bool,
    /// The device of every local rank as `(backend, number within that backend)`, in rank order (`--device cuda:0,vulkan:1`). Empty:
    /// the ranks use consecutive ordinals of the single backend `--device` names. Ranks of different backends exchange activations
    /// through host memory.
    pub rank_devices: Vec<(String, usize)>,
    /// Layers of every pipeline stage when the memory planner chose them (proportional to memory and speed); empty: an even split.
    pub stage_ranges: Vec<Range<usize>>,
    /// `--tensor-split 3,1`: relative share of the weights per device (overrides the speed the planner estimates); empty: automatic.
    pub split_ratio: Vec<f64>,
    /// `--tensor-parallel auto`: the size of the tensor-parallel groups is chosen from the devices (`group::auto_tensor_parallel`).
    pub tp_auto: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            tp: 1,
            pp: 1,
            expert_parallel: false,
            ep_dispatch: EpDispatch::AllReduce,
            devices: Vec::new(),
            backend: None,
            rendezvous: RendezvousConfig::single_node(),
            microbatches: 0,
            vocab_parallel_embedding: std::env::var("FLLM_VOCAB_PARALLEL_EMBED").is_ok_and(|v| matches!(v.trim(), "1" | "on" | "true")),
            rank_devices: Vec::new(),
            stage_ranges: Vec::new(),
            split_ratio: Vec::new(),
            tp_auto: false,
        }
    }
}

impl ParallelConfig {
    pub fn world_size(&self) -> usize {
        self.tp * self.pp
    }

    pub fn is_parallel(&self) -> bool {
        self.world_size() > 1
    }

    /// Ranks hosted by this process.
    pub fn local_world_size(&self) -> usize {
        self.world_size() / self.rendezvous.nnodes.max(1)
    }

    /// World rank of local rank `local`.
    pub fn world_rank(&self, local: usize) -> usize {
        self.rendezvous.node_rank * self.local_world_size() + local
    }

    /// Backend and number of the device of local rank `local` when `--device` listed them.
    pub fn rank_device(&self, local: usize) -> Option<&(String, usize)> {
        self.rank_devices.get(local)
    }

    /// Device ordinal of local rank `local`.
    pub fn device_of(&self, local: usize) -> usize {
        self.devices.get(local).copied().unwrap_or(local)
    }

    /// Pipeline stage of a world rank (stage-major numbering keeps a tensor-parallel group on neighbouring devices).
    pub fn stage_of(&self, rank: usize) -> usize {
        rank / self.tp
    }

    pub fn tp_rank_of(&self, rank: usize) -> usize {
        rank % self.tp
    }

    /// The layers of pipeline stage `stage`: contiguous ranges; the first stages take the remainder, so the last stage, which also
    /// computes the logits, never holds more layers than the others.
    pub fn stage_layers(&self, layers: usize, stage: usize) -> Range<usize> {
        if let Some(range) = self.stage_ranges.get(stage) {
            return range.clone();
        }
        let base = layers / self.pp;
        let extra = layers % self.pp;
        let start = stage * base + stage.min(extra);
        let len = base + usize::from(stage < extra);
        start..start + len
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.tp == 0 || self.pp == 0 {
            return Err("--tensor-parallel-size and --pipeline-parallel-size must be at least 1".into());
        }
        self.rendezvous.validate()?;
        let nodes = self.rendezvous.nnodes.max(1);
        if !self.world_size().is_multiple_of(nodes) {
            return Err(format!(
                "tensor x pipeline size {} x {} = {} ranks cannot be spread evenly over --nnodes {}",
                self.tp,
                self.pp,
                self.world_size(),
                nodes
            ));
        }
        if !self.devices.is_empty() {
            if self.devices.len() != self.local_world_size() {
                return Err(format!(
                    "--devices lists {} devices but this node hosts {} ranks (tensor {} x pipeline {} over {} node(s))",
                    self.devices.len(),
                    self.local_world_size(),
                    self.tp,
                    self.pp,
                    nodes
                ));
            }
            let mut seen = std::collections::HashSet::new();
            if let Some(dup) = self.devices.iter().find(|d| !seen.insert(**d)) {
                return Err(format!("--devices lists device {dup} twice"));
            }
        }
        if !self.rank_devices.is_empty() && self.rank_devices.len() != self.local_world_size() {
            return Err(format!(
                "--device lists {} devices but this node hosts {} ranks (tensor {} x pipeline {})",
                self.rank_devices.len(),
                self.local_world_size(),
                self.tp,
                self.pp
            ));
        }
        if !self.stage_ranges.is_empty() {
            if self.stage_ranges.len() != self.pp {
                return Err(format!("{} layer ranges for {} pipeline stages", self.stage_ranges.len(), self.pp));
            }
            let mut next = 0;
            for range in &self.stage_ranges {
                if range.start != next || range.is_empty() {
                    return Err("the layer ranges of the pipeline stages must be contiguous and not empty".into());
                }
                next = range.end;
            }
        }
        if self.expert_parallel && self.tp == 1 {
            return Err("--expert-parallel distributes experts over the tensor-parallel group: it needs --tensor-parallel-size of 2 or more".into());
        }
        Ok(())
    }

    /// Whether `flag` is one of ours.
    pub fn is_flag(flag: &str) -> bool {
        matches!(
            flag,
            "--tensor-parallel-size"
                | "--tensor-parallel"
                | "-tp"
                | "--pipeline-parallel-size"
                | "-pp"
                | "--expert-parallel"
                | "--enable-expert-parallel"
                | "--ep-dispatch"
                | "--devices"
                | "--distributed-backend"
                | "--nnodes"
                | "--node-rank"
                | "--master-addr"
                | "--master-port"
                | "--pipeline-microbatches"
                | "--tensor-split"
        )
    }

    /// Whether `flag` is followed by a value.
    pub fn takes_value(flag: &str) -> bool {
        !matches!(flag, "--expert-parallel" | "--enable-expert-parallel")
    }

    /// Apply one command-line flag (`value` is `None` for the switches).
    pub fn apply_flag(&mut self, flag: &str, value: Option<&str>) -> Result<(), String> {
        let number = |what: &str| -> Result<usize, String> {
            value.ok_or_else(|| format!("{what} needs a value"))?.trim().parse::<usize>().map_err(|_| format!("invalid value for {what}"))
        };
        match flag {
            "--tensor-parallel" | "--tensor-parallel-size" | "-tp" if value.is_some_and(|v| v.trim().eq_ignore_ascii_case("auto")) => self.tp_auto = true,
            "--tensor-parallel-size" | "--tensor-parallel" | "-tp" => self.tp = number(flag)?,
            "--pipeline-parallel-size" | "-pp" => self.pp = number(flag)?,
            "--expert-parallel" | "--enable-expert-parallel" => self.expert_parallel = true,
            "--ep-dispatch" => {
                self.ep_dispatch = match value.unwrap_or("").trim().to_ascii_lowercase().as_str() {
                    "allreduce" | "all-reduce" => EpDispatch::AllReduce,
                    "alltoall" | "all-to-all" => EpDispatch::AllToAll,
                    other => return Err(format!("unknown --ep-dispatch '{other}' (use allreduce or alltoall)")),
                }
            }
            "--devices" => {
                let list = value.ok_or("--devices needs a list such as 0,1,2,3")?;
                self.devices = list
                    .split(',')
                    .map(|part| part.trim().parse::<usize>().map_err(|_| format!("invalid device '{part}' in --devices")))
                    .collect::<Result<_, _>>()?;
            }
            "--distributed-backend" => self.backend = Some(BackendKind::parse(value.unwrap_or(""))?),
            "--nnodes" => self.rendezvous.nnodes = number(flag)?,
            "--node-rank" => self.rendezvous.node_rank = number(flag)?,
            "--master-addr" => self.rendezvous.master_addr = value.ok_or("--master-addr needs an address")?.to_owned(),
            "--master-port" => {
                self.rendezvous.master_port = value
                    .ok_or("--master-port needs a port")?
                    .trim()
                    .parse::<u16>()
                    .map_err(|_| "invalid --master-port".to_owned())?
            }
            "--pipeline-microbatches" => self.microbatches = number(flag)?,
            "--tensor-split" => {
                let list = value.ok_or("--tensor-split needs a list such as 3,1")?;
                self.split_ratio = list
                    .split(',')
                    .map(|part| part.trim().parse::<f64>().ok().filter(|r| r.is_finite() && *r > 0.0).ok_or_else(|| format!("invalid share '{part}' in --tensor-split (positive numbers)")))
                    .collect::<Result<_, _>>()?;
            }
            other => return Err(format!("unknown parallel option {other}")),
        }
        Ok(())
    }

    /// `--device a,b,c`: one rank per listed device. Without explicit sizes the model is split into one pipeline stage per device; with
    /// `--tensor-parallel-size N` (or `-pp M`) the list is grouped into `N` devices per stage.
    pub fn apply_device_list(&mut self, list: &[(String, usize)]) -> Result<(), String> {
        let n = list.len();
        if self.tp_auto {
            self.tp = crate::parallel::group::auto_tensor_parallel(list);
        }
        match (self.tp, self.pp) {
            (1, 1) => self.pp = n,
            (tp, 1) if n.is_multiple_of(tp) => self.pp = n / tp,
            (1, pp) if n.is_multiple_of(pp) => self.tp = n / pp,
            (tp, pp) if tp * pp == n => {}
            (tp, pp) => {
                return Err(format!(
                    "--device lists {n} devices, which tensor-parallel {tp} x pipeline-parallel {pp} cannot use (the sizes must multiply to the number of devices)"
                ))
            }
        }
        if !self.split_ratio.is_empty() && self.split_ratio.len() != n {
            return Err(format!("--tensor-split lists {} shares for {n} devices", self.split_ratio.len()));
        }
        self.rank_devices = list.to_vec();
        Ok(())
    }

    /// The layout as the Studio and the status routes show it: which devices run which layers. `None` for a single device.
    pub fn layout_json(&self) -> Option<serde_json::Value> {
        if !self.is_parallel() {
            return None;
        }
        let total = self.stage_ranges.last().map(|r| r.end);
        let stages: Vec<serde_json::Value> = (0..self.pp)
            .map(|stage| {
                let devices: Vec<String> = (0..self.tp)
                    .filter_map(|t| self.rank_devices.get(stage * self.tp + t))
                    .map(|(backend, index)| if backend == "cpu" { "cpu".to_owned() } else { format!("{backend}:{index}") })
                    .collect();
                let range = self.stage_ranges.get(stage);
                serde_json::json!({
                    "stage": stage,
                    "devices": devices,
                    "layers": range.map(|r| [r.start, r.end]),
                    "share": range.zip(total).map(|(r, t)| (r.len() as f64 / t.max(1) as f64 * 1000.0).round() / 1000.0),
                })
            })
            .collect();
        Some(serde_json::json!({
            "tensor_parallel": self.tp,
            "pipeline_parallel": self.pp,
            "expert_parallel": self.expert_parallel,
            "total_layers": total,
            "stages": stages,
        }))
    }

    pub fn describe(&self) -> String {
        format!(
            "tensor-parallel {} x pipeline-parallel {} = {} ranks{}{}",
            self.tp,
            self.pp,
            self.world_size(),
            if self.expert_parallel { ", expert-parallel" } else { "" },
            if self.rendezvous.nnodes > 1 { format!(", node {} of {}", self.rendezvous.node_rank, self.rendezvous.nnodes) } else { String::new() }
        )
    }
}

static GLOBAL: Mutex<Option<ParallelConfig>> = Mutex::new(None);

/// The layout the server was started with (set once by the command line before the model is attached).
pub fn set_global(config: ParallelConfig) {
    *GLOBAL.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(config);
}

pub fn global() -> Option<ParallelConfig> {
    GLOBAL.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone().filter(ParallelConfig::is_parallel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_are_parsed_and_validated() {
        let mut c = ParallelConfig::default();
        c.apply_flag("-tp", Some("4")).unwrap();
        c.apply_flag("--pipeline-parallel-size", Some("2")).unwrap();
        c.apply_flag("--enable-expert-parallel", None).unwrap();
        c.apply_flag("--devices", Some("0, 1,2,3,4,5,6,7")).unwrap();
        c.apply_flag("--distributed-backend", Some("p2p")).unwrap();
        c.apply_flag("--ep-dispatch", Some("alltoall")).unwrap();
        assert_eq!((c.world_size(), c.local_world_size()), (8, 8));
        assert_eq!(c.backend, Some(BackendKind::P2p));
        assert!(c.validate().is_ok());
        c.devices.pop();
        assert!(c.validate().unwrap_err().contains("--devices lists 7"));
        c.devices = vec![0, 1, 1, 3, 4, 5, 6, 7];
        assert!(c.validate().unwrap_err().contains("twice"));
        assert!(c.apply_flag("-tp", Some("x")).is_err());
        assert!(c.apply_flag("--distributed-backend", Some("mpi")).is_err());
    }

    #[test]
    fn expert_parallel_needs_a_tensor_group_and_nodes_must_divide_the_world() {
        let mut c = ParallelConfig { expert_parallel: true, ..Default::default() };
        assert!(c.validate().unwrap_err().contains("--expert-parallel"));
        c.tp = 2;
        assert!(c.validate().is_ok());
        c.rendezvous.nnodes = 3;
        c.rendezvous.master_addr = "x".into();
        assert!(c.validate().unwrap_err().contains("--nnodes"));
    }

    #[test]
    fn ranks_are_stage_major_and_layers_are_split_contiguously() {
        let c = ParallelConfig { tp: 2, pp: 3, ..Default::default() };
        assert_eq!((c.stage_of(5), c.tp_rank_of(5)), (2, 1));
        let ranges: Vec<_> = (0..3).map(|s| c.stage_layers(8, s)).collect();
        assert_eq!(ranges, vec![0..3, 3..6, 6..8]);
        let all: usize = ranges.iter().map(ExactSizeIterator::len).sum();
        assert_eq!(all, 8);
    }

    #[test]
    fn planned_ranges_replace_the_even_split_and_are_validated() {
        let mut c = ParallelConfig { pp: 3, ..Default::default() };
        c.stage_ranges = vec![0..10, 10..14, 14..16];
        assert!(c.validate().is_ok());
        assert_eq!((c.stage_layers(16, 0), c.stage_layers(16, 2)), (0..10, 14..16));
        c.stage_ranges = vec![0..10, 11..14, 14..16];
        assert!(c.validate().unwrap_err().contains("contiguous"));
        c.stage_ranges = vec![0..10, 10..16];
        assert!(c.validate().unwrap_err().contains("pipeline stages"));
        c.stage_ranges.clear();
        c.rank_devices = vec![("cuda".into(), 0), ("vulkan".into(), 1)];
        assert!(c.validate().unwrap_err().contains("--device lists 2"));
    }

    #[test]
    fn a_device_list_sets_the_layout() {
        let list: Vec<(String, usize)> = vec![("cuda".into(), 0), ("cuda".into(), 1), ("vulkan".into(), 0), ("cpu".into(), 0)];
        let mut c = ParallelConfig::default();
        c.apply_device_list(&list).unwrap();
        assert_eq!((c.tp, c.pp, c.world_size()), (1, 4, 4));
        assert!(c.validate().is_ok());
        let mut c = ParallelConfig { tp: 2, ..Default::default() };
        c.apply_device_list(&list).unwrap();
        assert_eq!((c.tp, c.pp), (2, 2));
        let mut c = ParallelConfig { tp: 3, ..Default::default() };
        assert!(c.apply_device_list(&list).unwrap_err().contains("multiply"));
        // `auto` picks layer splitting where no fast collective exists (here: no CUDA devices)
        let mut c = ParallelConfig::default();
        c.apply_flag("--tensor-parallel", Some("auto")).unwrap();
        c.apply_device_list(&list).unwrap();
        assert_eq!((c.tp, c.pp), (1, 4));
        let mut c = ParallelConfig::default();
        c.apply_flag("--tensor-split", Some("3, 1,1,1")).unwrap();
        assert_eq!(c.split_ratio, vec![3.0, 1.0, 1.0, 1.0]);
        assert!(c.apply_flag("--tensor-split", Some("3,x")).is_err());
        assert!(c.apply_flag("--tensor-split", Some("0,1")).is_err());
        c.apply_device_list(&list[..3]).unwrap_err();
    }

    #[test]
    fn the_layout_report_names_devices_layers_and_share() {
        assert!(ParallelConfig::default().layout_json().is_none());
        let mut c = ParallelConfig::default();
        c.apply_device_list(&[("cuda".into(), 0), ("vulkan".into(), 1), ("cpu".into(), 0)]).unwrap();
        c.stage_ranges = vec![0..20, 20..30, 30..32];
        let json = c.layout_json().unwrap();
        assert_eq!(json["pipeline_parallel"], 3);
        assert_eq!(json["stages"][1]["devices"][0], "vulkan:1");
        assert_eq!(json["stages"][2]["devices"][0], "cpu");
        assert_eq!(json["stages"][0]["layers"], serde_json::json!([0, 20]));
        assert_eq!(json["stages"][0]["share"], 0.625);
    }

    #[test]
    fn two_nodes_each_host_half_of_the_ranks() {
        let mut c = ParallelConfig { tp: 4, pp: 2, ..Default::default() };
        c.rendezvous = RendezvousConfig { nnodes: 2, node_rank: 1, master_addr: "10.0.0.1".into(), master_port: 29500 };
        assert!(c.validate().is_ok());
        assert_eq!(c.local_world_size(), 4);
        assert_eq!(c.world_rank(2), 6);
    }
}
