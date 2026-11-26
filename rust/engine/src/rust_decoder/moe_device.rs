//! Mixture-of-experts layers on the device without host work: router, top-k selection (softmax or sigmoid scores, selection bias, groups),
//! the expert-major sort, one grouped product per projection over all experts, the shared expert and the weighted combination.
//!
//! Nothing here reads data back, and every buffer size depends only on the number of rows (a step through such a layer is a candidate
//! for CUDA graphs, which `graph_eligible` does not accept for MoE layers yet). Layers or devices without the operations
//! (`Backend::moe_device_available`) keep the earlier paths: the softmax-only decode kernels of `moe_fast`, and routing on the host with one gated FFN per expert.

use super::*;
use crate::backend::{moe::moe_block_for, moe::moe_sort_bytes, BufferPool, MoeGlu, MoeGrouped, MoeRoute, MoeSort, Stream};

fn moe_decode_max_rows() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("FLLM_MOE_DECODE_MAX_ROWS").ok().and_then(|v| v.trim().parse::<usize>().ok()).filter(|&n| n > 0).unwrap_or(4))
}

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
        layer_idx: usize,
        dm: &DeviceMoe,
        fast: &DeviceMoeFast,
        dev: &DeviceMoeDevice,
        x: &crate::backend::DeviceBuffer,
        rows: usize,
    ) -> Result<Option<crate::backend::DeviceBuffer>, String> {
        let (Some(cfg), Some(_)) = (self.cfg.moe.as_ref(), self.device_glu_mode(layer_idx)) else {
            return Ok(None);
        };
        if let Some(latent) = dev.latent.as_ref() {
            return self.moe_device_latent(stream, scratch, layer_idx, dm, fast, dev, latent, x, rows);
        }
        // small batches through the per-slot decode kernels when the formats allow it; larger batches through the grouped GEMM
        let decode_cap = moe_decode_max_rows();
        // a backend without grouped products (Vulkan) takes the per-pair decode kernels for every batch size
        let decode = (rows <= decode_cap || stream.moe_decode_any_rows()) && fast.dp4a && !cfg.input_scaled;
        if rows == 0 || (!decode && !dev.grouped) {
            return Ok(None);
        }
        let err = |e: crate::backend::BackendError| e.to_string();
        let hidden = self.cfg.hidden;
        let out = scratch.acquire(rows * hidden * 4).map_err(err)?;
        let narrow = stream.act_dtype().is_narrow();
        let native_f16 = stream.act_dtype() == crate::backend::act_dtype::ActDtype::F16
            && fast.dp4a
            && fast.kinds.iter().all(|q| matches!(q.to_gguf_type_id(), 0x8000_0110..=0x8000_0112))
            && fast.gate.is_some()
            && fast.kinds[0] == fast.kinds[1]
            && fast.gate_bias.is_some() == fast.up_bias.is_some()
            && self
                .device_glu_mode(layer_idx)
                .is_some_and(|m| !matches!(m, GluMode::Clamped(_) | GluMode::Act(_) | GluMode::Plain(_)));
        let mut x_f32 = if narrow && !native_f16 {
            let mut w = scratch.acquire(rows * hidden * 4).map_err(err)?;
            stream.widen_activations(&mut w, x).map_err(err)?;
            Some(w)
        } else {
            None
        };
        let per_row = cfg.used * hidden.max(fast.ff) * 4;
        let max_rows = (Self::moe_chunk_bytes() / per_row.max(1)).max(1);
        let mut start = 0;
        while start < rows {
            let n = (rows - start).min(max_rows);
            if native_f16 {
                let xs = x.slice(start * hidden * 2, n * hidden * 4).map_err(err)?;
                let mut os = out.slice(start * hidden * 2, n * hidden * 4).map_err(err)?;
                self.moe_device_rows(stream, scratch, layer_idx, dm, fast, dev, &xs, &xs, &mut os, n, decode, None)?;
                start += n;
                continue;
            }
            let xs = x_f32.as_ref().unwrap_or(x).slice(start * hidden * 4, n * hidden * 4).map_err(err)?;
            let mut tmp_shared_n = None;
            let x_shared = if !narrow {
                &xs
            } else if n == rows {
                x
            } else {
                let mut wn = scratch.acquire(n * hidden * 4).map_err(err)?;
                stream.narrow_activations(&mut wn, &xs).map_err(err)?;
                tmp_shared_n.insert(wn)
            };
            if narrow {
                let mut os_wide = scratch.acquire(n * hidden * 4).map_err(err)?;
                self.moe_device_rows(stream, scratch, layer_idx, dm, fast, dev, &xs, x_shared, &mut os_wide, n, decode, None)?;
                if n == rows {
                    let mut out_mut = out.slice(0, out.size).map_err(err)?;
                    stream.narrow_activations(&mut out_mut, &os_wide).map_err(err)?;
                } else {
                    let mut out_n = scratch.acquire(n * hidden * 4).map_err(err)?;
                    stream.narrow_activations(&mut out_n, &os_wide).map_err(err)?;
                    let elem = stream.act_dtype().bytes();
                    let mut dst_slice = out.slice(start * hidden * elem, n * hidden * elem).map_err(err)?;
                    let src_slice = out_n.slice(0, n * hidden * elem).map_err(err)?;
                    stream.copy_buffer(&mut dst_slice, &src_slice).map_err(err)?;
                    scratch.recycle(out_n);
                }
                scratch.recycle(os_wide);
            } else {
                let mut os = out.slice(start * hidden * 4, n * hidden * 4).map_err(err)?;
                self.moe_device_rows(stream, scratch, layer_idx, dm, fast, dev, &xs, &xs, &mut os, n, decode, None)?;
            }
            if let Some(wn) = tmp_shared_n {
                scratch.recycle(wn);
            }
            start += n;
        }
        if let Some(w) = x_f32.take() {
            scratch.recycle(w);
        }
        Ok(Some(out))
    }

    /// `moe_device` for one chunk of rows: the result goes to `out`.
    #[allow(clippy::too_many_arguments)]
    fn moe_device_rows(
        &self,
        stream: &mut dyn Stream,
        scratch: &mut BufferPool,
        layer_idx: usize,
        dm: &DeviceMoe,
        fast: &DeviceMoeFast,
        dev: &DeviceMoeDevice,
        x: &crate::backend::DeviceBuffer,
        x_shared: &crate::backend::DeviceBuffer,
        out: &mut crate::backend::DeviceBuffer,
        rows: usize,
        decode: bool,
        latent: Option<(&crate::backend::DeviceBuffer, usize, &mut Option<(Option<crate::backend::DeviceBuffer>, Option<crate::backend::DeviceBuffer>)>)>,
    ) -> Result<(), String> {
        let (Some(cfg), Some(mode)) = (self.cfg.moe.as_ref(), self.device_glu_mode(layer_idx)) else {
            return Err("MoE layer without MoE configuration".into());
        };
        let err = |e: crate::backend::BackendError| e.to_string();
        let saved_act = stream.act_dtype();
        let native_f16 = saved_act == crate::backend::act_dtype::ActDtype::F16
            && fast.dp4a
            && fast.kinds.iter().all(|q| matches!(q.to_gguf_type_id(), 0x8000_0110..=0x8000_0112))
            && fast.gate.is_some()
            && fast.kinds[0] == fast.kinds[1]
            && fast.gate_bias.is_some() == fast.up_bias.is_some()
            && !matches!(mode, GluMode::Clamped(_) | GluMode::Act(_) | GluMode::Plain(_));
        let shared_mode = self.dense_glu_mode(layer_idx).unwrap_or(mode);
        let plain_act = act_code(self.cfg.activation);
        let (scored, k, hidden, ff) = (cfg.experts, cfg.used, self.cfg.hidden, fast.ff);
        // latent-space experts (Nemotron-H, Kimi K3): the router and the shared expert read `x` (`hidden` wide), the routed experts the projection
        // `xe` of it, `width` wide; their combination stays `width` wide and the caller projects it back up
        let (xe, width, mut latent_out) = match latent {
            Some((xe, width, extra)) => (xe, width, Some(extra)),
            None => (x, hidden, None),
        };
        // LongCat-Flash: the router also scores `zero_experts` experts without weights that return their input; the tables hold the others
        let zero = cfg.zero_experts;
        let experts = scored - zero;
        let pairs = rows * k;
        let mut logits = scratch.acquire(rows * scored * 4).map_err(err)?;
        if native_f16 {
            let mut logits_narrow = scratch.acquire(rows * scored * 4).map_err(err)?;
            stream
                .quantized_linear(&mut logits_narrow, x, &fast.router, fast.router.bias.as_ref())
                .map_err(err)?;
            stream.widen_activations(&mut logits, &logits_narrow).map_err(err)?;
            scratch.recycle(logits_narrow);
        } else {
            if saved_act.is_narrow() {
                stream.set_act_dtype(crate::backend::act_dtype::ActDtype::F32);
            }
            let router_res = stream
                .quantized_linear(&mut logits, x, &fast.router, fast.router.bias.as_ref())
                .map_err(err);
            if let Err(e) = router_res {
                if saved_act.is_narrow() {
                    stream.set_act_dtype(saved_act);
                }
                return Err(e);
            }
        }
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
        let mut y = scratch.acquire(pairs * width * 4).map_err(err)?;
        if zero > 0 && !decode {
            stream.zero_buffer(&mut y).map_err(err)?;
        }
        let mut temps = vec![logits];
        if decode {
            let mut act = scratch.acquire(pairs * ff * 4).map_err(err)?;
            let fuse = fast.gate.is_some()
                && fast.kinds[0] == fast.kinds[1]
                && fast.gate_bias.is_some() == fast.up_bias.is_some()
                && !matches!(mode, GluMode::Clamped(_) | GluMode::Act(_) | GluMode::Plain(_));
            let fused_done = if let (true, Some(gate_tbl)) = (fuse, fast.gate.as_ref()) {
                let glu = match mode {
                    GluMode::Swiglu => MoeGlu { mode: 0, alpha: 0.0, limit: 0.0 },
                    GluMode::Geglu => MoeGlu { mode: 1, alpha: 0.0, limit: 0.0 },
                    GluMode::SwigluOai(alpha, limit) => MoeGlu { mode: 2, alpha, limit },
                    GluMode::Clamped(_) | GluMode::Act(_) | GluMode::Plain(_) => unreachable!(),
                };
                stream
                    .moe_linear_glu(
                        &mut act,
                        xe,
                        gate_tbl,
                        &fast.up,
                        fast.gate_bias.as_ref(),
                        fast.up_bias.as_ref(),
                        &ids,
                        fast.kinds[0],
                        ff,
                        width,
                        pairs,
                        k,
                        glu,
                    )
                    .map_err(err)?
            } else {
                false
            };
            if !fused_done {
                let mut up = scratch.acquire(pairs * ff * 4).map_err(err)?;
                stream
                    .moe_linear(&mut up, xe, &fast.up, fast.up_bias.as_ref(), &ids, fast.kinds[1], ff, width, pairs, k)
                    .map_err(err)?;
                if let Some(gate_tbl) = fast.gate.as_ref() {
                    let mut gate = scratch.acquire(pairs * ff * 4).map_err(err)?;
                    stream
                        .moe_linear(&mut gate, xe, gate_tbl, fast.gate_bias.as_ref(), &ids, fast.kinds[0], ff, width, pairs, k)
                        .map_err(err)?;
                    Self::device_glu(stream, mode, &mut act, &gate, &up)?;
                    temps.push(gate);
                } else {
                    stream.activation(&mut act, &up, plain_act).map_err(err)?;
                }
                temps.push(up);
            }
            stream
                .moe_linear(&mut y, &act, &fast.down, fast.down_bias.as_ref(), &ids, fast.kinds[2], width, ff, pairs, 1)
                .map_err(err)?;
            temps.push(act);
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
            let fuse = fast.gate.is_some()
                && fast.kinds[0] == fast.kinds[1]
                && fast.gate_bias.is_some() == fast.up_bias.is_some()
                && !matches!(mode, GluMode::Clamped(_) | GluMode::Act(_) | GluMode::Plain(_));
            if let (true, Some(gate_tbl)) = (fuse, fast.gate.as_ref()) {
                let glu = match mode {
                    GluMode::Swiglu => MoeGlu { mode: 0, alpha: 0.0, limit: 0.0 },
                    GluMode::Geglu => MoeGlu { mode: 1, alpha: 0.0, limit: 0.0 },
                    GluMode::SwigluOai(alpha, limit) => MoeGlu { mode: 2, alpha, limit },
                    GluMode::Clamped(_) | GluMode::Act(_) | GluMode::Plain(_) => {
                        return Err("the fused expert kernels have no clamped or custom activation".into());
                    }
                };
                stream
                    .moe_grouped_linear(MoeGrouped {
                        out: &mut act,
                        x: xe,
                        table: gate_tbl,
                        table_up: Some(&fast.up),
                        bias: fast.gate_bias.as_ref(),
                        bias_up: fast.up_bias.as_ref(),
                        input_weights,
                        sort: &sort,
                        quant_type: fast.kinds[0],
                        rows: ff,
                        cols: width,
                        x_div: k,
                        glu: Some(glu),
                    })
                    .map_err(err)?;
            } else if let Some(gate_tbl) = fast.gate.as_ref() {
                let mut gate = scratch.acquire(pairs * ff * 4).map_err(err)?;
                let mut up = scratch.acquire(pairs * ff * 4).map_err(err)?;
                for (out, table, bias, quant) in [(&mut gate, gate_tbl, &fast.gate_bias, fast.kinds[0]), (&mut up, &fast.up, &fast.up_bias, fast.kinds[1])] {
                    stream
                        .moe_grouped_linear(MoeGrouped {
                            out,
                            x: xe,
                            table,
                            table_up: None,
                            bias: bias.as_ref(),
                            bias_up: None,
                            input_weights,
                            sort: &sort,
                            quant_type: quant,
                            rows: ff,
                            cols: width,
                            x_div: k,
                            glu: None,
                        })
                        .map_err(err)?;
                }
                Self::device_glu(stream, mode, &mut act, &gate, &up)?;
                temps.extend([gate, up]);
            } else {
                let mut up = scratch.acquire(pairs * ff * 4).map_err(err)?;
                stream
                    .moe_grouped_linear(MoeGrouped {
                        out: &mut up,
                        x: xe,
                        table: &fast.up,
                        table_up: None,
                        bias: fast.up_bias.as_ref(),
                        bias_up: None,
                        input_weights,
                        sort: &sort,
                        quant_type: fast.kinds[1],
                        rows: ff,
                        cols: width,
                        x_div: k,
                        glu: None,
                    })
                    .map_err(err)?;
                stream.activation(&mut act, &up, plain_act).map_err(err)?;
                temps.push(up);
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
                    rows: width,
                    cols: ff,
                    x_div: 1,
                    glu: None,
                })
                .map_err(err)?;
            temps.extend([act, sort.sorted, sort.block_expert, sort.post]);
        }
        if saved_act.is_narrow() && !native_f16 {
            stream.set_act_dtype(saved_act);
        }
        // the shared expert (a dense gated FFN on the same input) joins in the combination, scaled by sigmoid(g . x) when it has a gate
        let mut shared = None;
        let mut gate_scale = None;
        if let Some(sh) = dm.shared.as_ref() {
            let sh_mode = if sh.gate.is_some() { shared_mode } else { GluMode::Plain(plain_act) };
            let use_s2 = !stream.graph_capturing() && (rows > graph_max_rows() || graph_switch() == GraphSwitch::Off);
            if let Some(s2_lock) = self.backend_stream2.as_ref().filter(|_| use_s2) {
                if let Ok(mut stream2) = s2_lock.try_lock() {
                    stream2.set_act_dtype(stream.act_dtype());
                    stream2.wait_stream(stream).map_err(err)?;
                    shared = Some(Self::gated_run(&mut **stream2, scratch, sh_mode, &ExpertSrc::Device(sh), x_shared, rows)?);
                    if let Some(g) = dev.shared_gate.as_ref() {
                        let mut scale = scratch.acquire(rows * 4).map_err(err)?;
                        stream2.moe_gate_scale(&mut scale, x, g, rows, hidden).map_err(err)?;
                        gate_scale = Some(scale);
                    }
                    stream.wait_stream(&mut **stream2).map_err(err)?;
                } else {
                    shared = Some(Self::gated_run(stream, scratch, sh_mode, &ExpertSrc::Device(sh), x_shared, rows)?);
                    if let Some(g) = dev.shared_gate.as_ref() {
                        let mut scale = scratch.acquire(rows * 4).map_err(err)?;
                        stream.moe_gate_scale(&mut scale, x, g, rows, hidden).map_err(err)?;
                        gate_scale = Some(scale);
                    }
                }
            } else {
                shared = Some(Self::gated_run(stream, scratch, sh_mode, &ExpertSrc::Device(sh), x_shared, rows)?);
                if let Some(g) = dev.shared_gate.as_ref() {
                    let mut scale = scratch.acquire(rows * 4).map_err(err)?;
                    stream.moe_gate_scale(&mut scale, x, g, rows, hidden).map_err(err)?;
                    gate_scale = Some(scale);
                }
            }
            if stream.act_dtype().is_narrow() && !native_f16 {
                if let Some(sh_buf) = shared.take() {
                    let mut wide = scratch.acquire(rows * hidden * 4).map_err(err)?;
                    stream.widen_activations(&mut wide, &sh_buf).map_err(err)?;
                    scratch.recycle(sh_buf);
                    shared = Some(wide);
                }
            }
        }
        if let Some(extra) = latent_out.take() {
            // the routed result stays in the latent space; the shared expert joins after the up projection (`moe_device_latent`)
            stream.moe_combine_ex(out, &y, (!cfg.input_scaled).then_some(&weights), None, None, rows, k, width).map_err(err)?;
            *extra = Some((shared.take(), gate_scale.take()));
            temps.extend([ids, weights, y]);
            for b in temps {
                scratch.recycle(b);
            }
            return Ok(());
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

    /// A mixture-of-experts layer with latent-space experts (Nemotron-H, Kimi K3) for `rows` tokens, F32 activations: the router reads `x`, the experts the
    /// projection `down(x)`, their combination is normalised (`latent_norm`) and projected back up, the shared expert joins at the end.
    #[allow(clippy::too_many_arguments)]
    fn moe_device_latent(
        &self,
        stream: &mut dyn Stream,
        scratch: &mut BufferPool,
        layer_idx: usize,
        dm: &DeviceMoe,
        fast: &DeviceMoeFast,
        dev: &DeviceMoeDevice,
        latent: &DeviceLatent,
        x: &crate::backend::DeviceBuffer,
        rows: usize,
    ) -> Result<Option<crate::backend::DeviceBuffer>, String> {
        let Some(cfg) = self.cfg.moe.as_ref() else {
            return Ok(None);
        };
        let err = |e: crate::backend::BackendError| e.to_string();
        if stream.act_dtype().is_narrow() {
            return Err("latent-space experts need F32 activations".into());
        }
        let decode = (rows <= moe_decode_max_rows() || stream.moe_decode_any_rows()) && fast.dp4a && !cfg.input_scaled;
        if rows == 0 || (!decode && !dev.grouped) {
            return Err("latent-space experts need the device MoE kernels for the formats of this layer".into());
        }
        let (hidden, width) = (self.cfg.hidden, latent.width);
        let out = scratch.acquire(rows * hidden * 4).map_err(err)?;
        let per_row = cfg.used * width.max(fast.ff) * 4;
        let max_rows = (Self::moe_chunk_bytes() / per_row.max(1)).max(1);
        let mut start = 0;
        while start < rows {
            let n = (rows - start).min(max_rows);
            let xs = x.slice(start * hidden * 4, n * hidden * 4).map_err(err)?;
            let mut os = out.slice(start * hidden * 4, n * hidden * 4).map_err(err)?;
            let mut xe = scratch.acquire(n * width * 4).map_err(err)?;
            stream.quantized_linear(&mut xe, &xs, &latent.down, latent.down.bias.as_ref()).map_err(err)?;
            let mut routed = scratch.acquire(n * width * 4).map_err(err)?;
            let mut extra = None;
            self.moe_device_rows(stream, scratch, layer_idx, dm, fast, dev, &xs, &xs, &mut routed, n, decode, Some((&xe, width, &mut extra)))?;
            if let Some(norm) = latent.norm.as_ref() {
                let mut normed = scratch.acquire(n * width * 4).map_err(err)?;
                stream.rms_norm(&mut normed, &routed, norm, self.cfg.eps).map_err(err)?;
                scratch.recycle(std::mem::replace(&mut routed, normed));
            }
            let mut up = scratch.acquire(n * hidden * 4).map_err(err)?;
            stream.quantized_linear(&mut up, &routed, &latent.up, latent.up.bias.as_ref()).map_err(err)?;
            let (shared, gate) = extra.unwrap_or((None, None));
            // out = up + gate * shared  (k = 1, no routing weights)
            stream.moe_combine_ex(&mut os, &up, None, shared.as_ref(), gate.as_ref(), n, 1, hidden).map_err(err)?;
            for b in [xe, routed, up].into_iter().chain(shared).chain(gate) {
                scratch.recycle(b);
            }
            start += n;
        }
        Ok(Some(out))
    }
}
