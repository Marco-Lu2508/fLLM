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
        let needed_with_marlin = weights_on_device + marlin_q4k_extra + scratchpad_bytes + (384u64 << 20) + min_kv_needed;
        let marlin_q4k_dual = offload_layers == total_layers
            && marlin_q4k_extra > 0
            && free_vram >= needed_with_marlin;

        let final_reserved = if marlin_q4k_dual {
            weights_on_device + marlin_q4k_extra + scratchpad_bytes + (384u64 << 20)
        } else {
            reserved_bytes
        };
        let final_available_for_kv = if marlin_q4k_dual {
            free_vram.saturating_sub(final_reserved)
        } else {
            budget_bytes.saturating_sub(final_reserved)
        };

        let kv_block_count = final_available_for_kv
            .checked_div(kv_block_bytes)
            .map_or(model.pool_blocks(), |blocks| (blocks as usize).min(model.pool_blocks()));
        let kv_cache_bytes = (kv_block_count as u64) * kv_block_bytes;
        let all_reserved = non_layer_bytes + total_layers as u64 * bytes_per_layer + scratchpad_bytes + overhead_bytes
            + if marlin_q4k_dual { marlin_q4k_extra } else { 0 };
        let all_on_device_kv_blocks = free_vram.saturating_sub(all_reserved).checked_div(kv_block_bytes).unwrap_or(0) as usize;

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

/// Device memory of the recurrent states (Gated DeltaNet, Mamba-2, Mamba-1 layers): the decoder keeps up to two states per sequence slot
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
