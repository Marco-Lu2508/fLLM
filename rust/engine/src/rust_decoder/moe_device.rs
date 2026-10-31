//! Mixture-of-experts layers on the device without host work: router, top-k selection (softmax or sigmoid scores, selection bias, groups),
//! the expert-major sort, one grouped product per projection over all experts, the shared expert and the weighted combination.
//!
//! Nothing here reads data back, and every buffer size depends only on the number of rows (a step through such a layer is a candidate
//! for CUDA graphs, which `graph_eligible` does not accept for MoE layers yet). Layers or devices without the operations
//! (`Backend::moe_device_available`) keep the earlier paths: the softmax-only decode kernels of `moe_fast`, and routing on the host with one gated FFN per expert.

use super::*;
use crate::backend::{moe::moe_block_for, moe::moe_sort_bytes, BufferPool, MoeGlu, MoeGrouped, MoeRoute, MoeSort, Stream};

impl RustDecoder {
    /// Routing parameters of a layer for the device router.
    pub(super) fn device_route(&self, moe: &Moe) -> MoeRoute {
        let cfg = self.cfg.moe.as_ref().expect("MoE layer without MoE configuration");
        MoeRoute {
            sigmoid: matches!(cfg.gating, Gating::Sigmoid),
            normalize: cfg.normalize,
            scale: cfg.weight_scale,
            // LFM2-MoE's sigmoid router includes epsilon in the selected-weight denominator
            eps: if self.cfg.arch == "lfm2moe" { 1e-6 } else { 0.0 },
            groups: if cfg.groups > 1 { cfg.groups } else { 0 },
            groups_used: cfg.groups_used,
            group_top2: moe.bias.is_some() || cfg.group_top2,
        }
    }

    /// Bytes of the per-pair intermediate buffers (`act`, `y`) one chunk of a large batch may need: batches above that are cut into chunks of
    /// rows (`FLLM_MOE_CHUNK_MB`, default 256). The result does not depend on the cut.
    fn moe_chunk_bytes() -> usize {
        static MB: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *MB.get_or_init(|| std::env::var("FLLM_MOE_CHUNK_MB").ok().and_then(|v| v.trim().parse::<usize>().ok()).filter(|&m| m > 0).unwrap_or(256)) << 20
    }

    /// A whole mixture-of-experts layer for `rows` tokens with the device operations: `None` (nothing done) when the layer does not qualify
    /// for this batch size.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn moe_device(
        &self,
        stream: &mut dyn Stream,
        scratch: &mut BufferPool,
        dm: &DeviceMoe,
        fast: &DeviceMoeFast,
        dev: &DeviceMoeDevice,
        x: &crate::backend::DeviceBuffer,
        rows: usize,
    ) -> Result<Option<crate::backend::DeviceBuffer>, String> {
        let (Some(cfg), Some(_)) = (self.cfg.moe.as_ref(), self.device_glu_mode()) else {
            return Ok(None);
        };
        // one to four tokens through the integer dot-product kernels when the formats allow it; everything else through the grouped GEMM
        let decode = rows <= 4 && fast.dp4a && !cfg.input_scaled;
        if rows == 0 || (!decode && !dev.grouped) {
            return Ok(None);
        }
        let err = |e: crate::backend::BackendError| e.to_string();
        let hidden = self.cfg.hidden;
        let out = scratch.acquire(rows * hidden * 4).map_err(err)?;
        let per_row = cfg.used * hidden.max(fast.ff) * 4;
        let max_rows = (Self::moe_chunk_bytes() / per_row.max(1)).max(1);
        let mut start = 0;
        while start < rows {
            let n = (rows - start).min(max_rows);
            let xs = x.slice(start * hidden * 4, n * hidden * 4).map_err(err)?;
            let mut os = out.slice(start * hidden * 4, n * hidden * 4).map_err(err)?;
            self.moe_device_rows(stream, scratch, dm, fast, dev, &xs, &mut os, n, decode)?;
            start += n;
        }
        Ok(Some(out))
    }

    /// `moe_device` for one chunk of rows: the result goes to `out`.
    #[allow(clippy::too_many_arguments)]
    fn moe_device_rows(
        &self,
        stream: &mut dyn Stream,
        scratch: &mut BufferPool,
        dm: &DeviceMoe,
        fast: &DeviceMoeFast,
        dev: &DeviceMoeDevice,
        x: &crate::backend::DeviceBuffer,
        out: &mut crate::backend::DeviceBuffer,
        rows: usize,
        decode: bool,
    ) -> Result<(), String> {
        let (Some(cfg), Some(mode)) = (self.cfg.moe.as_ref(), self.device_glu_mode()) else {
            return Err("MoE layer without MoE configuration".into());
        };
        let err = |e: crate::backend::BackendError| e.to_string();
        let shared_mode = self.dense_glu_mode().unwrap_or(mode);
        let (scored, k, hidden, ff) = (cfg.experts, cfg.used, self.cfg.hidden, fast.ff);
        // LongCat-Flash: the router also scores `zero_experts` experts without weights that return their input; the tables hold the others
        let zero = cfg.zero_experts;
        let experts = scored - zero;
        let pairs = rows * k;
        let mut logits = scratch.acquire(rows * experts * 4).map_err(err)?;
        stream.quantized_linear(&mut logits, x, &fast.router, fast.router.bias.as_ref()).map_err(err)?;
        let mut ids = scratch.acquire(pairs * 4).map_err(err)?;
        let mut weights = scratch.acquire(pairs * 4).map_err(err)?;
        stream.moe_route_ex(&logits, dev.bias.as_ref(), &mut ids, &mut weights, rows, scored, k, &dev.route).map_err(err)?;
        // the weights of the zero experts leave the pairs: `zero_scale[r] * x[r]` is their whole contribution. The decode kernels index their tables by
        // id, so those pairs are redirected to expert 0 with weight 0; the sorted path skips ids beyond the local experts (their rows of `y` stay zero)
        let mut zero_scale = None;
        if zero > 0 {
            let mut zs = scratch.acquire(rows * 4).map_err(err)?;
            stream.moe_zero_fold(&mut zs, &mut ids, &mut weights, rows, k, experts, decode).map_err(err)?;
            zero_scale = Some(zs);
        }
        let mut y = scratch.acquire(pairs * hidden * 4).map_err(err)?;
        let mut temps = vec![logits];
        if decode {
            let mut gate = scratch.acquire(pairs * ff * 4).map_err(err)?;
            let mut up = scratch.acquire(pairs * ff * 4).map_err(err)?;
            let mut act = scratch.acquire(pairs * ff * 4).map_err(err)?;
            stream
                .moe_linear(&mut gate, x, &fast.gate, fast.gate_bias.as_ref(), &ids, fast.kinds[0], ff, hidden, pairs, k)
                .map_err(err)?;
            stream
                .moe_linear(&mut up, x, &fast.up, fast.up_bias.as_ref(), &ids, fast.kinds[1], ff, hidden, pairs, k)
                .map_err(err)?;
            Self::device_glu(stream, mode, &mut act, &gate, &up)?;
            stream
                .moe_linear(&mut y, &act, &fast.down, fast.down_bias.as_ref(), &ids, fast.kinds[2], hidden, ff, pairs, 1)
                .map_err(err)?;
            temps.extend([gate, up, act]);
        } else {
            let block = moe_block_for(pairs, experts);
            let (sorted, blocks, post) = moe_sort_bytes(pairs, experts, block);
            let mut sort = MoeSort {
                sorted: scratch.acquire(sorted).map_err(err)?,
                block_expert: scratch.acquire(blocks).map_err(err)?,
                post: scratch.acquire(post).map_err(err)?,
                pairs,
                block,
                max_blocks: blocks / 4,
            };
            stream.moe_align(&ids, &mut sort, 0, experts).map_err(err)?;
            // Llama 4 scales the expert input by the router score; every other model scales the output (in the combination)
            let input_weights = cfg.input_scaled.then_some(&weights);
            let mut act = scratch.acquire(pairs * ff * 4).map_err(err)?;
            let fuse = fast.kinds[0] == fast.kinds[1] && fast.gate_bias.is_some() == fast.up_bias.is_some() && !matches!(mode, GluMode::Clamped(_));
            if fuse {
                let glu = match mode {
                    GluMode::Swiglu => MoeGlu { mode: 0, alpha: 0.0, limit: 0.0 },
                    GluMode::Geglu => MoeGlu { mode: 1, alpha: 0.0, limit: 0.0 },
                    GluMode::SwigluOai(alpha, limit) => MoeGlu { mode: 2, alpha, limit },
                    GluMode::Clamped(_) => return Err("the fused expert kernels have no clamped activation".into()),
                };
                stream
                    .moe_grouped_linear(MoeGrouped {
                        out: &mut act,
                        x,
                        table: &fast.gate,
                        table_up: Some(&fast.up),
                        bias: fast.gate_bias.as_ref(),
                        bias_up: fast.up_bias.as_ref(),
                        input_weights,
                        sort: &sort,
                        quant_type: fast.kinds[0],
                        rows: ff,
                        cols: hidden,
                        x_div: k,
                        glu: Some(glu),
                    })
                    .map_err(err)?;
            } else {
                let mut gate = scratch.acquire(pairs * ff * 4).map_err(err)?;
                let mut up = scratch.acquire(pairs * ff * 4).map_err(err)?;
                for (out, table, bias, quant) in [(&mut gate, &fast.gate, &fast.gate_bias, fast.kinds[0]), (&mut up, &fast.up, &fast.up_bias, fast.kinds[1])] {
                    stream
                        .moe_grouped_linear(MoeGrouped {
                            out,
                            x,
                            table,
                            table_up: None,
                            bias: bias.as_ref(),
                            bias_up: None,
                            input_weights,
                            sort: &sort,
                            quant_type: quant,
                            rows: ff,
                            cols: hidden,
                            x_div: k,
                            glu: None,
                        })
                        .map_err(err)?;
                }
                Self::device_glu(stream, mode, &mut act, &gate, &up)?;
                temps.extend([gate, up]);
            }
            stream
                .moe_grouped_linear(MoeGrouped {
                    out: &mut y,
                    x: &act,
                    table: &fast.down,
                    table_up: None,
                    bias: fast.down_bias.as_ref(),
                    bias_up: None,
                    input_weights: None,
                    sort: &sort,
                    quant_type: fast.kinds[2],
                    rows: hidden,
                    cols: ff,
                    x_div: 1,
                    glu: None,
                })
                .map_err(err)?;
            temps.extend([act, sort.sorted, sort.block_expert, sort.post]);
        }
        // the shared expert (a dense gated FFN on the same input) joins in the combination, scaled by sigmoid(g . x) when it has a gate
        let mut shared = None;
        let mut gate_scale = None;
        if let Some(sh) = dm.shared.as_ref() {
            if let Some(ref s2_lock) = self.backend_stream2 {
                if let Ok(mut stream2) = s2_lock.try_lock() {
                    shared = Some(Self::gated_run(&mut **stream2, scratch, shared_mode, &ExpertSrc::Device(sh), x, rows)?);
                    if let Some(g) = dev.shared_gate.as_ref() {
                        let mut scale = scratch.acquire(rows * 4).map_err(err)?;
                        stream2.moe_gate_scale(&mut scale, x, g, rows, hidden).map_err(err)?;
                        gate_scale = Some(scale);
                    }
                    stream.wait_stream(&mut **stream2).map_err(err)?;
                } else {
                    shared = Some(Self::gated_run(stream, scratch, shared_mode, &ExpertSrc::Device(sh), x, rows)?);
                    if let Some(g) = dev.shared_gate.as_ref() {
                        let mut scale = scratch.acquire(rows * 4).map_err(err)?;
                        stream.moe_gate_scale(&mut scale, x, g, rows, hidden).map_err(err)?;
                        gate_scale = Some(scale);
                    }
                }
            } else {
                shared = Some(Self::gated_run(stream, scratch, shared_mode, &ExpertSrc::Device(sh), x, rows)?);
                if let Some(g) = dev.shared_gate.as_ref() {
                    let mut scale = scratch.acquire(rows * 4).map_err(err)?;
                    stream.moe_gate_scale(&mut scale, x, g, rows, hidden).map_err(err)?;
                    gate_scale = Some(scale);
                }
            }
        }
        if let Some(zs) = zero_scale {
            // no shared expert in such a router (`device_routing_ok`): its slot carries the zero experts' identity term
            stream.moe_combine_ex(out, &y, Some(&weights), Some(x), Some(&zs), rows, k, hidden).map_err(err)?;
            temps.push(zs);
        } else {
            stream
                .moe_combine_ex(out, &y, (!cfg.input_scaled).then_some(&weights), shared.as_ref(), gate_scale.as_ref(), rows, k, hidden)
                .map_err(err)?;
        }
        temps.extend([ids, weights, y]);
        temps.extend(shared);
        temps.extend(gate_scale);
        for b in temps {
            scratch.recycle(b);
        }
        Ok(())
    }
}
