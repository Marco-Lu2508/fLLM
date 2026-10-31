//! Inkling on a device (the host pass is `inkling_layers` in `inkling.rs`; both read the same weights).
//!
//! A layer runs as the host does: attention with a convolution on the keys and the values (`Stream::conv_resid`, the history of the last `kernel - 1` raw inputs of
//! every channel in the sequence's device state buffer), the learned distance profile and log scaling inside the attention kernel (`Stream::rel_attention`, keys
//! and values in the ordinary paged KV cache), a convolution over the attention output, then the feed-forward block (dense and scaled, or the sigmoid-routed experts
//! with shared experts) and a convolution over its output. The routing of the experts is computed on the host from the normed stream (like every mixture of experts
//! on a device without device routing), the experts themselves run on the device.
use super::inkling::InkLayer;
use super::*;
use crate::backend::{DeltaRun, RelAttnParams};

/// Device copy of an Inkling layer: the projections, the small tensors and the experts. The dense feed-forward uses the generic device weights of the layer.
pub(super) struct DeviceInk {
    pub(super) q: crate::backend::DeviceWeight,
    pub(super) k: crate::backend::DeviceWeight,
    pub(super) v: crate::backend::DeviceWeight,
    pub(super) r: crate::backend::DeviceWeight,
    pub(super) o: crate::backend::DeviceWeight,
    pub(super) qn: crate::backend::DeviceBuffer,
    pub(super) kn: crate::backend::DeviceBuffer,
    pub(super) profile: crate::backend::DeviceBuffer,
    pub(super) k_conv: crate::backend::DeviceBuffer,
    pub(super) v_conv: crate::backend::DeviceBuffer,
    pub(super) attn_conv: crate::backend::DeviceBuffer,
    pub(super) ffn_conv: crate::backend::DeviceBuffer,
    pub(super) experts: Vec<DeviceExpert>,
    pub(super) shared: Vec<DeviceExpert>,
}

impl RustDecoder {
    /// Bytes of the four convolution histories of a layer (keys, values, attention output, feed-forward output), each rounded up to 256 bytes.
    fn ink_state_sizes(&self, il: &InkLayer) -> [usize; 4] {
        let Some(ic) = self.cfg.inkling.as_ref() else { return [0; 4] };
        let size = |channels: usize, taps: usize| (channels * (taps - 1) * 4).next_multiple_of(256);
        [size(il.kv_width(), ic.conv_kernel), size(il.kv_width(), ic.conv_kernel), size(self.cfg.hidden, ic.block_conv_kernel), size(self.cfg.hidden, ic.block_conv_kernel)]
    }

    /// Bytes one layer takes in a sequence's state buffer: the largest layer's (global and sliding layers have different key widths).
    pub(super) fn ink_layer_bytes(&self) -> usize {
        self.inkling.as_ref().map_or(0, |m| m.layers.iter().map(|il| self.ink_state_sizes(il).iter().sum::<usize>()).max().unwrap_or(0))
    }

    /// The attention parameters of a layer.
    pub(super) fn ink_attention_params(&self, il: &InkLayer) -> Option<RelAttnParams> {
        let ic = self.cfg.inkling.as_ref()?;
        Some(RelAttnParams {
            heads: il.heads,
            kv_heads: il.kv_heads,
            head_dim: il.hd,
            d_rel: ic.d_rel,
            extent: il.extent,
            window: if il.sliding { ic.window } else { 0 },
            log_floor: if il.sliding { 0 } else { ic.log_floor },
            log_alpha: ic.log_alpha,
            scale: 1.0 / il.hd as f32,
            stride: self.stride,
            block_size: self.block_size,
        })
    }

    /// Can the device run every layer of this model (the convolutions and the attention for the head widths of both kinds of layers)?
    pub(super) fn ink_device_possible(&self, backend: &dyn crate::backend::Backend) -> bool {
        let (Some(ic), Some(model)) = (self.cfg.inkling.as_ref(), self.inkling.as_deref()) else { return false };
        if self.par.is_some() || std::env::var("FLLM_INKLING").is_ok_and(|v| v.eq_ignore_ascii_case("host")) || !matches!(self.device_glu_mode(), Some(GluMode::Swiglu)) {
            return false;
        }
        let Ok(stream) = backend.create_stream() else { return false };
        stream.supports_conv_resid(self.cfg.hidden, ic.block_conv_kernel)
            && model.layers.iter().all(|il| {
                stream.supports_conv_resid(il.kv_width(), ic.conv_kernel) && self.ink_attention_params(il).is_some_and(|p| stream.supports_rel_attention(&p, self.kv_kind))
            })
    }

    /// One Inkling layer on the device: `x` is the residual stream (rows x hidden).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn inkling_layer_device(
        &self,
        stream: &mut dyn crate::backend::Stream,
        scratch: &mut crate::backend::BufferPool,
        i: usize,
        kvi: usize,
        dev_layer: &DeviceLayerWeights,
        ink: &DeviceInk,
        x: &mut crate::backend::DeviceBuffer,
        (slots, tables, lengths): (&[usize], &[i64], &[i64]),
        plan: &[DeltaPlanRun],
        dev_kv: &crate::backend::DeviceKvCache,
    ) -> Result<(), String> {
        let err = |e: crate::backend::BackendError| e.to_string();
        let (ic, model) = (self.cfg.inkling.as_ref().ok_or("Inkling configuration missing")?, self.inkling.as_deref().ok_or("Inkling weights missing")?);
        let il = model.layers.get(i).ok_or("Inkling layer out of range")?;
        let (rows, hidden, eps) = (lengths.len(), self.cfg.hidden, self.cfg.eps);
        let (heads, hd, kvw) = (il.heads, il.hd, il.kv_width());
        let sizes = self.ink_state_sizes(il);
        let base = i * self.ink_layer_bytes();
        let offsets = [base, base + sizes[0], base + sizes[0] + sizes[1], base + sizes[0] + sizes[1] + sizes[2]];
        let resets = self.delta_claim(i, plan)?;
        let runs: Vec<DeltaRun<'_>> = plan.iter().zip(&resets).map(|(p, &reset)| DeltaRun { state: &p.buf, first_row: p.start, rows: p.n, reset }).collect();

        let mut normed = scratch.acquire(rows * hidden * 4).map_err(err)?;
        let attn_norm = dev_layer.attn_norm.as_ref().ok_or("Inkling layer without an attention norm")?;
        device_norm(stream, self.cfg.norm, &mut normed, x, attn_norm, dev_layer.attn_norm_b.as_ref(), eps, self.groups)?;
        let mut q = scratch.acquire(rows * heads * hd * 4).map_err(err)?;
        stream.quantized_linear(&mut q, &normed, &ink.q, ink.q.bias.as_ref()).map_err(err)?;
        device_rms_inplace(stream, scratch, &mut q, Some(&ink.qn), eps, 1)?;
        let mut k_raw = scratch.acquire(rows * kvw * 4).map_err(err)?;
        stream.quantized_linear(&mut k_raw, &normed, &ink.k, ink.k.bias.as_ref()).map_err(err)?;
        let mut v_raw = scratch.acquire(rows * kvw * 4).map_err(err)?;
        stream.quantized_linear(&mut v_raw, &normed, &ink.v, ink.v.bias.as_ref()).map_err(err)?;
        let mut rel = scratch.acquire(rows * heads * ic.d_rel * 4).map_err(err)?;
        stream.quantized_linear(&mut rel, &normed, &ink.r, ink.r.bias.as_ref()).map_err(err)?;
        scratch.recycle(normed);
        // keys and values: the convolution adds the raw input back; the keys are normalised per head
        let mut k = scratch.acquire(rows * kvw * 4).map_err(err)?;
        stream.conv_resid(kvw, ic.conv_kernel, &ink.k_conv, &runs, offsets[0], &k_raw, &mut k).map_err(err)?;
        device_rms_inplace(stream, scratch, &mut k, Some(&ink.kn), eps, 1)?;
        let mut v = scratch.acquire(rows * kvw * 4).map_err(err)?;
        stream.conv_resid(kvw, ic.conv_kernel, &ink.v_conv, &runs, offsets[1], &v_raw, &mut v).map_err(err)?;
        stream.kv_cache_write(dev_kv, kvi, slots, &k, &v).map_err(err)?;
        let params = self.ink_attention_params(il).ok_or("Inkling attention parameters")?;
        let mut attn = scratch.acquire(rows * heads * hd * 4).map_err(err)?;
        stream.rel_attention(&mut attn, &q, &rel, &ink.profile, dev_kv, tables, lengths, kvi, &params).map_err(err)?;
        for buf in [q, k_raw, v_raw, rel, k, v] {
            scratch.recycle(buf);
        }
        // the attention output, convolved, joins the stream
        let mut projected = scratch.acquire(rows * hidden * 4).map_err(err)?;
        stream.quantized_linear(&mut projected, &attn, &ink.o, ink.o.bias.as_ref()).map_err(err)?;
        scratch.recycle(attn);
        let mut convolved = scratch.acquire(rows * hidden * 4).map_err(err)?;
        stream.conv_resid(hidden, ic.block_conv_kernel, &ink.attn_conv, &runs, offsets[2], &projected, &mut convolved).map_err(err)?;
        stream.residual_add(x, &convolved, 1.0).map_err(err)?;
        // feed-forward block
        let mut f = scratch.acquire(rows * hidden * 4).map_err(err)?;
        let ffn_norm = dev_layer.ffn_norm.as_ref().ok_or("Inkling layer without a feed-forward norm")?;
        device_norm(stream, self.cfg.norm, &mut f, x, ffn_norm, dev_layer.ffn_norm_b.as_ref(), eps, self.groups)?;
        let mlp = if il.moe.is_some() { self.ink_moe_device(stream, scratch, il, ink, &f, rows)? } else { self.ink_dense_device(stream, scratch, il, dev_layer, &f, rows)? };
        scratch.recycle(f);
        stream.conv_resid(hidden, ic.block_conv_kernel, &ink.ffn_conv, &runs, offsets[3], &mlp, &mut convolved).map_err(err)?;
        stream.residual_add(x, &convolved, 1.0).map_err(err)?;
        for buf in [projected, convolved, mlp] {
            scratch.recycle(buf);
        }
        Ok(())
    }

    /// `scale * down(silu(gate(f)) * up(f))` of a dense layer.
    fn ink_dense_device(
        &self,
        stream: &mut dyn crate::backend::Stream,
        scratch: &mut crate::backend::BufferPool,
        il: &InkLayer,
        dev_layer: &DeviceLayerWeights,
        f: &crate::backend::DeviceBuffer,
        rows: usize,
    ) -> Result<crate::backend::DeviceBuffer, String> {
        let err = |e: crate::backend::BackendError| e.to_string();
        let (Some(gate), Some(up), Some(down)) = (&dev_layer.ffn_gate, &dev_layer.ffn_up, &dev_layer.ffn_down) else {
            return Err("Inkling dense layer without gate, up and down projections on the device".into());
        };
        let ff = gate.rows;
        let mut g = scratch.acquire(rows * ff * 4).map_err(err)?;
        let mut u = scratch.acquire(rows * ff * 4).map_err(err)?;
        let mut act = scratch.acquire(rows * ff * 4).map_err(err)?;
        stream.quantized_linear(&mut g, f, gate, gate.bias.as_ref()).map_err(err)?;
        stream.quantized_linear(&mut u, f, up, up.bias.as_ref()).map_err(err)?;
        stream.swiglu(&mut act, &g, &u).map_err(err)?;
        let mut out = scratch.acquire(rows * self.cfg.hidden * 4).map_err(err)?;
        stream.quantized_linear(&mut out, &act, down, down.bias.as_ref()).map_err(err)?;
        if il.dense_scale != 1.0 {
            stream.scale_inplace(&mut out, il.dense_scale).map_err(err)?;
        }
        for buf in [g, u, act] {
            scratch.recycle(buf);
        }
        Ok(out)
    }

    /// The routed and shared experts for the normed rows `f`: routing on the host (the weights of a token's chosen and shared experts come out of one softmax over
    /// log-sigmoid logits), every expert that got tokens runs on the device on its tokens and adds its weighted result.
    fn ink_moe_device(
        &self,
        stream: &mut dyn crate::backend::Stream,
        scratch: &mut crate::backend::BufferPool,
        il: &InkLayer,
        ink: &DeviceInk,
        f: &crate::backend::DeviceBuffer,
        rows: usize,
    ) -> Result<crate::backend::DeviceBuffer, String> {
        let err = |e: crate::backend::BackendError| e.to_string();
        let (ic, im) = (self.cfg.inkling.as_ref().ok_or("Inkling configuration missing")?, il.moe.as_ref().ok_or("Inkling layer without experts")?);
        let hidden = self.cfg.hidden;
        let mut host_in = vec![0.0f32; rows * hidden];
        stream.download(cast_f32_to_u8_mut(&mut host_in), f).map_err(err)?;
        let mut assigned: Vec<Vec<(i32, f32)>> = vec![Vec::new(); ic.routed];
        let mut shared_weights = vec![vec![0.0f32; rows]; ic.shared];
        for t in 0..rows {
            let (order, weights) = self.ink_route(im, &host_in[t * hidden..(t + 1) * hidden], ic)?;
            for (n, &e) in order.iter().enumerate() {
                assigned[e].push((t as i32, weights[n]));
            }
            for (s, w) in shared_weights.iter_mut().enumerate() {
                w[t] = weights[order.len() + s];
            }
        }
        let mode = self.device_glu_mode().ok_or("Inkling feed-forward without a device activation")?;
        let mut acc = scratch.acquire(rows * hidden * 4).map_err(err)?;
        stream.zero_buffer(&mut acc).map_err(err)?;
        for (e, tokens) in assigned.iter().enumerate().filter(|(_, t)| !t.is_empty()) {
            let idx: Vec<i32> = tokens.iter().map(|&(t, _)| t).collect();
            let weights: Vec<f32> = tokens.iter().map(|&(_, w)| w).collect();
            // every token goes to this expert, in order: no gather needed
            let direct = idx.len() == rows && idx.iter().enumerate().all(|(n, &t)| t as usize == n);
            let mut gathered = None;
            if !direct {
                let mut xe = scratch.acquire(idx.len() * hidden * 4).map_err(err)?;
                stream.gather_rows(&mut xe, f, &idx, None, hidden).map_err(err)?;
                gathered = Some(xe);
            }
            let result = Self::gated_run(stream, scratch, mode, &ExpertSrc::Device(&ink.experts[e]), gathered.as_ref().unwrap_or(f), idx.len())?;
            stream.scatter_add_rows(&mut acc, &result, &idx, Some(&weights), hidden).map_err(err)?;
            scratch.recycle(result);
            if let Some(xe) = gathered {
                scratch.recycle(xe);
            }
        }
        let all: Vec<i32> = (0..rows as i32).collect();
        for (expert, weights) in ink.shared.iter().zip(&shared_weights) {
            let result = Self::gated_run(stream, scratch, mode, &ExpertSrc::Device(expert), f, rows)?;
            stream.scatter_add_rows(&mut acc, &result, &all, Some(weights), hidden).map_err(err)?;
            scratch.recycle(result);
        }
        Ok(acc)
    }
}
