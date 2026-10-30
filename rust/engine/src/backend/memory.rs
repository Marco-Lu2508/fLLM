//! GPU VRAM memory planning, budget calculation, and layer offloading.
//!
//! A layer that does not fit on the device keeps its attention (and with it the KV cache) there while its
//! feed-forward weights stay in host memory; the feed-forward then runs on the CPU between the device layers
//! (the same mechanism that serves mixture-of-experts layers). `offload_layers` is the number of leading
//! layers whose weights live completely on the device.

use crate::model::GgufModel;

/// Memory plan detailing device budget, weight allocation, and KV cache sizing.
#[derive(Debug, Clone)]
pub struct MemoryPlan {
    pub total_vram_bytes: u64,
    pub free_vram_bytes: u64,
    pub budget_bytes: u64,
    pub utilization: f64,
    pub estimated_model_bytes: u64,
    pub bytes_per_layer: u64,
    pub non_layer_bytes: u64,
    pub scratchpad_bytes: u64,
    pub kv_cache_bytes: u64,
    pub kv_block_count: usize,
    /// KV blocks that would fit with every layer completely on the device (the context to choose when the requested
    /// one does not fit and decode speed matters more than context length).
    pub all_on_device_kv_blocks: usize,
    pub offload_layers: usize,
    pub total_layers: usize,
    pub marlin_q4k_dual: bool,
    pub marlin_q4k_extra_bytes: u64,
}

fn tensor_bytes(t: &fllm_gguf_probe::TensorInfo) -> u64 {
    let elem_count = t
        .dimensions
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d))
        .unwrap_or(0);
    // packed EXL2 / EXL3 blobs have no fixed block size: the blob is what the device holds (backends without a kernel expand to F16)
    if fllm_gguf_probe::exl::is_exl_type(t.type_id) {
        return t.payload_len;
    }
    let kind = super::QuantType::from_gguf_type_id(t.type_id);
    // sizes of the device layouts: Q6_K blocks are padded to 224 bytes, IQ4_XS becomes IQ4_NL blocks (144 per 256
    // values), MXFP4 blocks carry an fp16 scale (18 bytes)
    let block_bytes = match t.type_id {
        14 => 224,
        23 => 144,
        39 => 18,
        _ => kind.block_bytes() as u64,
    };
    elem_count.div_ceil(kind.block_size() as u64) * block_bytes
}

impl MemoryPlan {
    /// Compute a memory plan given the model configuration, available device memory,
    /// target utilization (e.g. 0.90), and desired offloaded layers ("auto" or specific count).
    pub fn plan(
        model: &GgufModel,
        free_vram: u64,
        total_vram: u64,
        utilization: f64,
        gpu_layers_arg: Option<&str>,
        max_tokens: usize,
    ) -> Result<Self, String> {
        Self::plan_for_backend(model, free_vram, total_vram, utilization, gpu_layers_arg, max_tokens, true)
    }

    /// `marlin_q4k_allowed`: the CUDA backend can keep a second, Marlin-ordered copy of Q4_K weights; every other backend has no use for it, and its
    /// memory (5/8 byte per weight) must not be taken out of the KV cache.
    pub fn plan_for_backend(
        model: &GgufModel,
        free_vram: u64,
        total_vram: u64,
        utilization: f64,
        gpu_layers_arg: Option<&str>,
        max_tokens: usize,
        marlin_q4k_allowed: bool,
    ) -> Result<Self, String> {
        let util = utilization.clamp(0.01, 1.0);
        let budget_bytes = ((free_vram as f64) * util) as u64;

        let total_layers = model.layer_count();
        let hidden = model.hidden_size();
        let vocab = model.vocab_size();

        // Weights per kind of tensor, from the GGUF index
        let (mut layer_attn, mut layer_ffn, mut total_model_bytes, mut head_bytes) = (0u64, 0u64, 0u64, 0u64);
        let mut tied_head = 0u64;
        for t in &model.mapping().index().tensors {
            let bytes = tensor_bytes(t);
            total_model_bytes += bytes;
            if t.name.starts_with("blk.") {
                if t.name.contains(".ffn_") {
                    layer_ffn += bytes;
                } else {
                    layer_attn += bytes;
                }
            } else if t.name == "output.weight" {
                head_bytes = bytes;
            } else if t.name == "token_embd.weight" {
                tied_head = bytes;
            }
        }
        // The output projection lives on the device (the embedding table does not); a float head is stored as Q6_K
        let head_bytes = if head_bytes > 0 { head_bytes } else { tied_head };
        let non_layer_bytes = if head_bytes >= (hidden * vocab) as u64 * 2 {
            (hidden * vocab) as u64 * 224 / 256
        } else {
            head_bytes
        };
        let layers = total_layers.max(1) as u64;
        // The experts of a mixture-of-experts layer are feed-forward weights like any other: the leading layers keep
        // them on the device, the others run on the host (decode) or stream over PCIe (prompts)
        let ffn_per_layer = layer_ffn / layers;
        let attn_per_layer = layer_attn / layers;
        let bytes_per_layer = attn_per_layer + ffn_per_layer;

        // Scratchpad memory estimate for activations:
        // max_batch * (hidden + ffn + vocab) * 4 bytes
        // residual / norm / attention buffers (about 12 hidden-sized vectors per token with the pool's slack), the gate / up /
        // activation buffers of a feed-forward, and the logits of the few rows that are sampled (not of every prompt token)
        let ff = model.config().ff.max(hidden);
        // batches of host-streamed models (more than 512 tokens) keep more buffers of the pool alive at once
        let (hid_mul, ff_mul) = if max_tokens > 512 { (28, 14) } else { (12, 6) };
        let scratchpad_bytes = ((max_tokens * (hidden * hid_mul + ff * ff_mul) + vocab * 16) * 4) as u64;

        // KV cache calculations per layer:
        let block_size = model.block_size();
        let (key_format, value_format) = model.kv_formats();
        // bytes of all layers together: layers differ in KV heads / head width and layers that share a cache add nothing
        let bytes_per_token = crate::kv_geometry::bytes_per_token(model.config(), key_format, value_format);
        // Every layer keeps its KV cache on the device, also when its feed-forward runs on the host
        let kv_block_bytes = (block_size as u64) * bytes_per_token;

        // Backend workspaces, the driver context and padded weight layouts
        let overhead_bytes = (768u64 << 20) + (layer_attn + layer_ffn) / 20 + recurrent_state_bytes(model);
        // Smallest pool worth giving up decode speed for: one full-context sequence. More concurrent sequences share the
        // pool (the scheduler preempts when it runs dry); offloading feed-forward layers to the host costs far more.
        let min_blocks = model.table_stride().max(1) as u64;
        let fixed = non_layer_bytes + scratchpad_bytes + overhead_bytes + min_blocks * kv_block_bytes;

        // Leading layers whose weights are completely on the device; the others keep only their attention there.
        let all_on_device = total_layers as u64 * bytes_per_layer;
        let offload_layers = match gpu_layers_arg {
            Some("auto") | None => {
                if budget_bytes > fixed + all_on_device {
                    total_layers
                } else {
                    let base = fixed + total_layers as u64 * attn_per_layer;
                    // even one full-context sequence does not fit next to the attention weights: keep only attention on
                    // the device and give the KV pool what is left (shorter than the context), as long as it holds a few blocks
                    let floor = non_layer_bytes + scratchpad_bytes + overhead_bytes + total_layers as u64 * attn_per_layer + 64 * kv_block_bytes;
                    match budget_bytes.checked_sub(base) {
                        Some(room) if ffn_per_layer > 0 => ((room / ffn_per_layer) as usize).min(total_layers),
                        Some(_) => total_layers,
                        None if budget_bytes >= floor => 0,
                        None => {
                            return Err(format!(
                                "the attention weights and KV cache of {} do not fit into {:.1} GB of device memory",
                                total_layers,
                                budget_bytes as f64 / 1e9
                            ))
                        }
                    }
                }
            }
            Some(s) => {
                let requested: usize = s
                    .parse()
                    .map_err(|_| format!("invalid --gpu-layers argument '{s}'"))?;
                requested.min(total_layers)
            }
        };

        // Remaining memory for the KV cache
        let weights_on_device =
            non_layer_bytes + total_layers as u64 * attn_per_layer + offload_layers as u64 * ffn_per_layer;
        let reserved_bytes = weights_on_device + scratchpad_bytes + overhead_bytes;

        // Check if Q4_K Marlin dual-weight layout can be enabled without starving the KV cache or forcing layer offload.
        // Marlin layout adds 160 bytes per 256 weights (= 5/8 bytes/element) for 64x256-aligned Q4_K tensors.
        let mut marlin_q4k_extra = 0u64;
        for t in &model.mapping().index().tensors {
            if t.type_id == 12 && t.dimensions.len() >= 2 {
                let cols = t.dimensions[0] as usize;
                let rows = t.dimensions[1] as usize;
                if cols.is_multiple_of(64) && rows.is_multiple_of(256) {
                    marlin_q4k_extra += (cols * rows * 5 / 8) as u64;
                }
            }
        }

        let min_kv_needed = min_blocks * kv_block_bytes;
        // Dual layout is enabled when:
        // 1. All layers are on device (offload_layers == total_layers).
        // 2. The model has Q4_K weights (marlin_q4k_extra > 0).
        // 3. Free VRAM has room for weights, dual Marlin layout, scratchpad, driver overhead (384 MB), and at least min_blocks KV cache.
        // (the same reserve as without it: the utilization budget, the backend overhead and the recurrent states; the KV pool used to get
        // everything up to the card's free memory here, and a busy step then ran out of memory)
        let needed_with_marlin = reserved_bytes + marlin_q4k_extra + min_kv_needed;
        let marlin_q4k_dual = marlin_q4k_allowed
            && offload_layers == total_layers
            && marlin_q4k_extra > 0
            && budget_bytes >= needed_with_marlin;

        let final_reserved = if marlin_q4k_dual {
            reserved_bytes + marlin_q4k_extra
        } else {
            reserved_bytes
        };
        let final_available_for_kv = budget_bytes.saturating_sub(final_reserved);

        let kv_block_count = final_available_for_kv
            .checked_div(kv_block_bytes)
            .map_or(model.pool_blocks(), |blocks| (blocks as usize).min(model.pool_blocks()));
        let kv_cache_bytes = (kv_block_count as u64) * kv_block_bytes;
        let all_reserved = non_layer_bytes + total_layers as u64 * bytes_per_layer + scratchpad_bytes + overhead_bytes
            + if marlin_q4k_dual { marlin_q4k_extra } else { 0 };
        // (of the budget, like the plan itself: counted against the free memory the context came out about a tenth too long, the plan with this
        // context then no longer held every layer and sent feed-forward weights to the host; seen on an integrated GPU with 5.5 GB)
        let all_on_device_kv_blocks = budget_bytes.saturating_sub(all_reserved).checked_div(kv_block_bytes).unwrap_or(0) as usize;

        let total_model_bytes = if marlin_q4k_dual {
            total_model_bytes + marlin_q4k_extra
        } else {
            total_model_bytes
        };

        Ok(Self {
            total_vram_bytes: total_vram,
            free_vram_bytes: free_vram,
            budget_bytes,
            utilization: util,
            estimated_model_bytes: total_model_bytes,
            bytes_per_layer,
            non_layer_bytes,
            scratchpad_bytes,
            kv_cache_bytes,
            kv_block_count,
            all_on_device_kv_blocks,
            offload_layers,
            total_layers,
            marlin_q4k_dual,
            marlin_q4k_extra_bytes: marlin_q4k_extra,
        })
    }

    /// Compute a per-device memory plan for tensor and pipeline parallel configurations.
    /// Each rank only accounts for its share of attention / FFN weights and pipeline layers.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_sharded(
        model: &GgufModel,
        free_vram: u64,
        total_vram: u64,
        utilization: f64,
        gpu_layers_arg: Option<&str>,
        max_tokens: usize,
        tp: usize,
        pp: usize,
        stage: usize,
    ) -> Result<Self, String> {
        let tp = tp.max(1);
        let pp = pp.max(1);
        let mut plan = Self::plan(model, free_vram, total_vram, utilization, gpu_layers_arg, max_tokens)?;
        
        // Scale layer weights by TP (each rank holds 1/TP of attention and FFN projection columns/rows)
        plan.bytes_per_layer /= tp as u64;
        
        // Pipeline stages divide the model layers
        let stage_layers = (plan.total_layers / pp) + usize::from(stage < (plan.total_layers % pp));
        plan.total_layers = stage_layers;
        plan.offload_layers = plan.offload_layers.min(stage_layers);
        plan.estimated_model_bytes = (plan.non_layer_bytes / tp as u64) + (plan.bytes_per_layer * plan.offload_layers as u64);

        // KV cache budget per rank: KV heads are sharded by TP
        plan.kv_cache_bytes /= tp as u64;
        Ok(plan)
    }

}

/// One device of a layer split (pipeline stages over several devices) as the memory planner sees it.
#[derive(Clone, Debug)]
pub struct PipelineDevice {
    /// `cuda:0 (NVIDIA RTX 2000 Ada)` for log lines.
    pub label: String,
    /// Memory the stage may use: the free memory times the utilisation (host memory for the CPU).
    pub budget_bytes: u64,
    /// Relative speed of reading weights (GB/s, or a user ratio): the share of weights follows it.
    pub speed: f64,
}

/// What one stage of a layer split holds.
#[derive(Clone, Debug)]
pub struct PipelineStage {
    pub device: usize,
    pub layers: std::ops::Range<usize>,
    pub weight_bytes: u64,
    /// Output head (last stage only).
    pub head_bytes: u64,
    /// KV cache of the common pool in this stage.
    pub kv_cache_bytes: u64,
    pub scratchpad_bytes: u64,
    pub budget_bytes: u64,
    /// KV blocks this stage alone could hold.
    pub max_kv_blocks: u64,
}

/// The split of a model over several devices and the KV pool they share (one block table: the pool has as many blocks as the smallest
/// stage can hold).
#[derive(Clone, Debug)]
pub struct PipelinePlan {
    pub stages: Vec<PipelineStage>,
    pub kv_blocks: usize,
    pub total_layers: usize,
    /// Blocks of a full-context sequence the plan had to give up (0: the requested context fits).
    pub context_blocks: usize,
}

impl PipelinePlan {
    pub fn ranges(&self) -> Vec<std::ops::Range<usize>> {
        self.stages.iter().map(|s| s.layers.clone()).collect()
    }
}

/// Sizes of the model for the layer-split planner: the weights and cache of every layer, the output head, and the fixed share of each
/// device (driver context, scratch, padded weight layouts are folded into the layer sizes).
pub fn pipeline_sizes(model: &GgufModel, max_tokens: usize, min_kv_blocks: u64) -> crate::parallel::split::ModelSizes {
    let layers = model.layer_count();
    let hidden = model.hidden_size();
    let vocab = model.vocab_size();
    let mut layer_bytes = vec![0u64; layers];
    let (mut head_bytes, mut tied_head) = (0u64, 0u64);
    for t in &model.mapping().index().tensors {
        let bytes = tensor_bytes(t);
        if let Some(rest) = t.name.strip_prefix("blk.") {
            if let Some(index) = rest.split('.').next().and_then(|n| n.parse::<usize>().ok()).filter(|&n| n < layers) {
                layer_bytes[index] += bytes;
            }
        } else if t.name == "output.weight" {
            head_bytes = bytes;
        } else if t.name == "token_embd.weight" {
            tied_head = bytes;
        }
    }
    let head = if head_bytes > 0 { head_bytes } else { tied_head };
    let head_bytes = if head >= (hidden * vocab) as u64 * 2 { (hidden * vocab) as u64 * 224 / 256 } else { head };
    // padded device layouts and backend workspaces: the single-device planner reserves a twentieth of the weights, and the recurrent
    // states of hybrid models are spread over the layers
    let recurrent = recurrent_state_bytes(model) / layers.max(1) as u64;
    for b in &mut layer_bytes {
        *b += *b / 20 + recurrent;
    }
    let ff = model.config().ff.max(hidden);
    let (hid_mul, ff_mul) = if max_tokens > 512 { (28, 14) } else { (12, 6) };
    let scratchpad_bytes = ((max_tokens * (hidden * hid_mul + ff * ff_mul) + vocab * 16) * 4) as u64;
    let (key_format, value_format) = model.kv_formats();
    let block_size = model.block_size() as u64;
    let kv_block_bytes = (0..layers)
        .map(|l| block_size * crate::kv_geometry::bytes_per_token_layer(model.config(), l, key_format, value_format))
        .collect();
    crate::parallel::split::ModelSizes {
        layer_bytes,
        kv_block_bytes,
        head_bytes,
        fixed_bytes: (768u64 << 20) + scratchpad_bytes,
        min_kv_blocks,
        // the split tries to leave room for a window of 4096 tokens (or one full context, if that is longer); the rest of the memory goes into
        // the pool up to the size the server asked for
        wanted_kv_blocks: (model.pool_blocks() as u64).min(min_kv_blocks.max(256)),
        pool_blocks: model.pool_blocks() as u64,
    }
}

/// Splits `model` into `devices.len()` contiguous stages and sizes the shared KV pool. When even one sequence of the full context does
/// not fit next to the weights, the context is given up in steps (the plan then reports how many blocks a sequence gets; the caller
/// lowers the context to it), down to 64 blocks.
pub fn plan_pipeline(model: &GgufModel, devices: &[PipelineDevice], max_tokens: usize) -> Result<PipelinePlan, String> {
    let split_devices: Vec<crate::parallel::split::SplitDevice> = devices
        .iter()
        .map(|d| crate::parallel::split::SplitDevice { label: d.label.clone(), budget_bytes: d.budget_bytes, speed: d.speed })
        .collect();
    let stride = model.table_stride().max(1) as u64;
    let mut last_error = String::new();
    for min_blocks in [stride, 512, 256, 64] {
        if min_blocks > stride {
            continue;
        }
        let sizes = pipeline_sizes(model, max_tokens, min_blocks);
        match crate::parallel::split::plan_split(&split_devices, &sizes) {
            Ok(split) => {
                let stages = split
                    .stages
                    .iter()
                    .map(|s| PipelineStage {
                        device: s.device,
                        layers: s.layers.clone(),
                        weight_bytes: s.weight_bytes,
                        head_bytes: s.head_bytes,
                        kv_cache_bytes: s.kv_block_bytes * split.kv_blocks,
                        scratchpad_bytes: sizes.fixed_bytes - (768u64 << 20),
                        budget_bytes: s.budget_bytes,
                        max_kv_blocks: s.max_kv_blocks,
                    })
                    .collect();
                return Ok(PipelinePlan {
                    stages,
                    kv_blocks: split.kv_blocks as usize,
                    total_layers: sizes.layer_bytes.len(),
                    context_blocks: if min_blocks < stride { min_blocks as usize } else { 0 },
                });
            }
            Err(e) => last_error = e,
        }
    }
    Err(last_error)
}

/// The split of a model over the devices of `--device a,b,c`, with what the planner knew about each device (for the log, `fllm-rs
/// devices --split` and the Studio).
#[derive(Clone, Debug)]
pub struct DeviceListPlan {
    pub plan: PipelinePlan,
    /// One entry per pipeline stage (a tensor-parallel group counts as one).
    pub stages: Vec<PipelineDevice>,
    /// One entry per listed device.
    pub devices: Vec<crate::devices::PlanningDevice>,
}

impl DeviceListPlan {
    /// The lines the log and `fllm-rs devices --split` print.
    pub fn describe(&self, block_size: usize) -> Vec<String> {
        let mut lines = vec![format!(
            "Layer split over {} stage(s): shared KV pool of {} blocks ({} tokens); each stage holds the cache of its own layers",
            self.plan.stages.len(),
            self.plan.kv_blocks,
            self.plan.kv_blocks * block_size
        )];
        for d in &self.devices {
            lines.push(format!(
                "  device {}: {:.2} GB of {:.2} GB usable | read speed {:.0} GB/s ({})",
                d.label,
                d.budget_bytes as f64 / 1e9,
                d.total_bytes as f64 / 1e9,
                d.speed_gbs,
                if d.speed_known { "reported or measured" } else { "estimate for this kind of device" }
            ));
        }
        for (n, stage) in self.plan.stages.iter().enumerate() {
            lines.push(format!(
                "  stage {n}: {} layers {}..{} of {} | weights {:.2} GB{} | KV cache {:.2} GB | scratch {:.0} MB | budget {:.2} GB",
                self.stages[stage.device].label,
                stage.layers.start,
                stage.layers.end,
                self.plan.total_layers,
                (stage.weight_bytes + stage.head_bytes) as f64 / 1e9,
                if stage.head_bytes > 0 { format!(" (output head {:.2} GB)", stage.head_bytes as f64 / 1e9) } else { String::new() },
                stage.kv_cache_bytes as f64 / 1e9,
                stage.scratchpad_bytes as f64 / 1e6,
                stage.budget_bytes as f64 / 1e9
            ));
        }
        lines
    }
}

/// Plan the split of `model` over `list` (backend and number of each device, in pipeline order; `tp` devices form one stage).
/// `split_ratio` (`--tensor-split`) replaces the estimated speeds; `reserve_first` takes memory away from the first device (a draft model).
pub fn plan_device_list(
    model: &GgufModel,
    list: &[(String, usize)],
    tp: usize,
    utilization: f64,
    split_ratio: &[f64],
    reserve_first: u64,
    max_tokens: usize,
) -> Result<DeviceListPlan, String> {
    let tp = tp.max(1);
    let mut devices = Vec::new();
    for (n, (backend, index)) in list.iter().enumerate() {
        let mut d = crate::devices::planning_device(backend, *index, utilization)?;
        if n == 0 {
            d.budget_bytes = d.budget_bytes.saturating_sub(reserve_first);
        }
        devices.push(d);
    }
    // a tensor-parallel group of `tp` devices is one pipeline stage: equal shards, so it holds `tp` times the smallest budget and reads
    // `tp` times as fast as its slowest member
    let stages: Vec<PipelineDevice> = devices
        .chunks(tp)
        .enumerate()
        .map(|(stage, group)| PipelineDevice {
            label: group.iter().map(|d| d.label.clone()).collect::<Vec<_>>().join(" + "),
            budget_bytes: group.iter().map(|d| d.budget_bytes).min().unwrap_or(0) * group.len() as u64,
            speed: group
                .iter()
                .enumerate()
                .map(|(k, d)| split_ratio.get(stage * tp + k).copied().unwrap_or(d.speed_gbs))
                .fold(f64::MAX, f64::min)
                * group.len() as f64,
        })
        .collect();
    let plan = plan_pipeline(model, &stages, max_tokens)?;
    Ok(DeviceListPlan { plan, stages, devices })
}

/// Device memory of the recurrent states/// Device memory of the recurrent states (Gated DeltaNet, Mamba-2, Mamba-1 layers): the decoder keeps up to two states per sequence slot
/// (`state_capacity`), each holding every recurrent layer. They are allocated while the server runs, so the KV pool must leave room for
/// them (16 sequences of Qwen3.5-2B need about 600 MB; without the reserve the pool took the whole card and the first busy step failed).
fn recurrent_state_bytes(model: &crate::model::GgufModel) -> u64 {
    let cfg = model.config();
    let mut layers = std::collections::BTreeSet::new();
    for t in &model.mapping().index().tensors {
        if let Some(rest) = t.name.strip_prefix("blk.") {
            if t.name.contains(".ssm_") {
                if let Some(index) = rest.split('.').next().and_then(|n| n.parse::<usize>().ok()) {
                    layers.insert(index);
                }
            }
        }
    }
    if layers.is_empty() {
        return 0;
    }
    let per_layer = if let Some(d) = &cfg.delta {
        let conv_dim = 2 * d.k_heads * d.k_dim + d.v_heads * d.v_dim;
        d.v_heads * d.k_dim * d.v_dim * 4 + conv_dim * d.conv_kernel.saturating_sub(1) * 4
    } else if let Some(m) = &cfg.mamba {
        m.inner * m.state * 4 + (m.inner + 2 * m.groups * m.state) * m.conv_kernel.saturating_sub(1) * 4
    } else if let Some(m) = &cfg.mamba1 {
        m.inner * m.state * 4 + m.inner * m.conv_kernel.saturating_sub(1) * 4
    } else {
        0
    };
    (layers.len() * per_layer.next_multiple_of(256) * model.max_sequences().max(1) * 2) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel::fixture::{scratch_path, TinySpec};

    fn model(case: &str) -> GgufModel {
        let path = scratch_path(&format!("memory-{case}"));
        TinySpec::named(case).unwrap().write(&path).unwrap();
        let model = GgufModel::load(path.to_str().unwrap(), 64, 2, 1).unwrap();
        let _ = std::fs::remove_file(&path);
        model
    }

    fn device(label: &str, gb: f64, speed: f64) -> PipelineDevice {
        PipelineDevice { label: label.to_owned(), budget_bytes: (gb * 1e9) as u64, speed }
    }

    #[test]
    fn the_sizes_add_up_to_the_file_and_the_cache() {
        let model = model("llama-q4k");
        let sizes = pipeline_sizes(&model, 512, 4);
        assert_eq!(sizes.layer_bytes.len(), model.layer_count());
        assert!(sizes.layer_bytes.iter().all(|&b| b > 0));
        let (key, value) = model.kv_formats();
        let per_token = crate::kv_geometry::bytes_per_token(model.config(), key, value);
        assert_eq!(sizes.kv_block_bytes.iter().sum::<u64>(), per_token * model.block_size() as u64);
        assert!(sizes.head_bytes > 0 && sizes.fixed_bytes > 768 << 20);
    }

    #[test]
    fn a_model_is_split_over_one_to_four_devices() {
        let model = model("llama-q4k");
        for n in 1..=4usize {
            let devices: Vec<_> = (0..n).map(|i| device(&format!("d{i}"), 4.0, 400.0)).collect();
            let plan = plan_pipeline(&model, &devices, 512).unwrap_or_else(|e| panic!("{n} devices: {e}"));
            assert_eq!(plan.stages.len(), n);
            let mut next = 0;
            for stage in &plan.stages {
                assert_eq!(stage.layers.start, next);
                assert!(!stage.layers.is_empty());
                next = stage.layers.end;
            }
            assert_eq!(next, model.layer_count());
            assert!(plan.kv_blocks >= 1);
            assert!(plan.stages.last().unwrap().head_bytes > 0);
            assert!(plan.stages.iter().rev().skip(1).all(|s| s.head_bytes == 0));
        }
    }

    #[test]
    fn faster_devices_take_more_layers_and_small_budgets_are_refused() {
        let model = model("llama-q4k");
        let plan = plan_pipeline(&model, &[device("fast", 4.0, 900.0), device("slow", 4.0, 100.0)], 512).unwrap();
        assert!(plan.stages[0].layers.len() >= plan.stages[1].layers.len());
        let error = plan_pipeline(&model, &[device("tiny0", 0.2, 400.0), device("tiny1", 0.2, 400.0)], 512).unwrap_err();
        assert!(error.contains("does not fit"), "{error}");
        // more devices than layers
        let many: Vec<_> = (0..9).map(|i| device(&format!("d{i}"), 2.0, 100.0)).collect();
        assert!(plan_pipeline(&model, &many, 512).unwrap_err().contains("every device needs at least one layer"));
    }
}
