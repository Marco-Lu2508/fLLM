//! Kimi Delta Attention layers in the decoder (host path): weights, state handling and the call into `cpu_kda`.
//!
//! GGUF tensors of a KDA layer `blk.N`: `attn_q`, `attn_k`, `attn_v` (the three projections), `ssm_conv1d_{q,k,v}.weight`
//! (depthwise short convolutions, no bias), `ssm_f_a` / `ssm_f_b` (low-rank forget gate) or `ssm_f` (one projection, Ling-3.0),
//! `ssm_dt.bias`, `ssm_a` (`-exp(A_log)` per head, the usual Mamba convention), `ssm_beta` (write strength), `ssm_g_a` / `ssm_g_b`
//! (low-rank output gate) or `ssm_g` (one projection), `ssm_o_norm.weight` and `attn_output`.
use super::*;

pub(super) struct Kda {
    q: Linear,
    k: Linear,
    v: Linear,
    /// `None`: `forget_up` projects the layer input directly.
    forget_down: Option<Linear>,
    forget_up: Linear,
    beta: Linear,
    /// `None`: `gate_up` projects the layer input directly.
    gate_down: Option<Linear>,
    gate_up: Linear,
    /// q, k, v convolution weights one after the other: `3 * heads * head_dim` rows of `kernel` taps.
    conv: Vec<f32>,
    dt_bias: Vec<f32>,
    /// `exp(A_log)` per head.
    rate: Vec<f32>,
    norm: Vec<f32>,
}

impl Kda {
    /// `None` when layer `layer` is not a KDA layer.
    pub(super) fn load(
        cfg: &ModelConfig,
        mapping: &MappedGguf,
        p: &str,
        layer: usize,
        linear: &dyn Fn(&str, usize, usize) -> Result<Linear, String>,
    ) -> Result<Option<Self>, String> {
        let Some(kc) = &cfg.hybrid.kda else { return Ok(None) };
        if !kc.layers.get(layer).copied().unwrap_or(false) {
            return Ok(None);
        }
        let (h, width) = (cfg.hidden, kc.heads * kc.head_dim);
        let mut conv = Vec::with_capacity(3 * width * kc.conv_kernel);
        for part in ["q", "k", "v"] {
            conv.extend(vector(mapping, &format!("{p}.ssm_conv1d_{part}.weight"), width * kc.conv_kernel)?);
        }
        let a = vector(mapping, &format!("{p}.ssm_a"), kc.heads)?;
        let low_forget = has_tensor(mapping, &format!("{p}.ssm_f_a.weight"));
        let low_gate = has_tensor(mapping, &format!("{p}.ssm_g_a.weight"));
        Ok(Some(Self {
            q: linear(&format!("{p}.attn_q"), width, h)?,
            k: linear(&format!("{p}.attn_k"), width, h)?,
            v: linear(&format!("{p}.attn_v"), width, h)?,
            forget_down: if low_forget { Some(linear(&format!("{p}.ssm_f_a"), kc.head_dim, h)?) } else { None },
            forget_up: if low_forget { linear(&format!("{p}.ssm_f_b"), width, kc.head_dim)? } else { linear(&format!("{p}.ssm_f"), width, h)? },
            beta: linear(&format!("{p}.ssm_beta"), kc.heads, h)?,
            gate_down: if low_gate { Some(linear(&format!("{p}.ssm_g_a"), kc.head_dim, h)?) } else { None },
            gate_up: if low_gate { linear(&format!("{p}.ssm_g_b"), width, kc.head_dim)? } else { linear(&format!("{p}.ssm_g"), width, h)? },
            conv,
            dt_bias: vector(mapping, &format!("{p}.ssm_dt.bias"), width)?,
            rate: a.iter().map(|v| -v).collect(),
            norm: vector(mapping, &format!("{p}.ssm_o_norm.weight"), kc.head_dim)?,
        }))
    }

    pub(super) fn linears_mut(&mut self) -> Vec<&mut Linear> {
        let mut all = vec![&mut self.q, &mut self.k, &mut self.v];
        all.extend(self.forget_down.iter_mut());
        all.extend([&mut self.forget_up, &mut self.beta]);
        all.extend(self.gate_down.iter_mut());
        all.push(&mut self.gate_up);
        all
    }
}

/// Device copy of a KDA layer (projections as device weights, the small tensors as FP32 buffers).
pub(super) struct DeviceKda {
    q: crate::backend::DeviceWeight,
    k: crate::backend::DeviceWeight,
    v: crate::backend::DeviceWeight,
    forget_down: Option<crate::backend::DeviceWeight>,
    forget_up: crate::backend::DeviceWeight,
    beta: crate::backend::DeviceWeight,
    gate_down: Option<crate::backend::DeviceWeight>,
    gate_up: crate::backend::DeviceWeight,
    conv: crate::backend::DeviceBuffer,
    dt_bias: crate::backend::DeviceBuffer,
    rate: crate::backend::DeviceBuffer,
    norm: crate::backend::DeviceBuffer,
    /// Index among the KDA layers (selects the slice of a sequence's state buffer).
    ordinal: usize,
}

impl Kda {
    /// Uploads the layer. `linear` uploads a projection, `vector` a small FP32 vector.
    pub(super) fn upload(
        &self,
        ordinal: usize,
        linear: &dyn Fn(&str, &Linear) -> Result<crate::backend::DeviceWeight, String>,
        vector: &dyn Fn(&[f32]) -> Result<crate::backend::DeviceBuffer, String>,
        name: &str,
    ) -> Result<DeviceKda, String> {
        let part = |tag: &str, l: &Linear| linear(&format!("{name}.{tag}"), l);
        Ok(DeviceKda {
            q: part("q", &self.q)?,
            k: part("k", &self.k)?,
            v: part("v", &self.v)?,
            forget_down: self.forget_down.as_ref().map(|l| part("forget_down", l)).transpose()?,
            forget_up: part("forget_up", &self.forget_up)?,
            beta: part("beta", &self.beta)?,
            gate_down: self.gate_down.as_ref().map(|l| part("gate_down", l)).transpose()?,
            gate_up: part("gate_up", &self.gate_up)?,
            conv: vector(&self.conv)?,
            dt_bias: vector(&self.dt_bias)?,
            rate: vector(&self.rate)?,
            norm: vector(&self.norm)?,
            ordinal,
        })
    }
}

impl RustDecoder {
    /// One KDA layer on the device: the projections (q, k and v are packed into the `[q | k | v]` rows of the kernel), the convolution and recurrence
    /// kernels, the output projection added to the residual stream.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn kda_device(
        &self,
        stream: &mut dyn crate::backend::Stream,
        scratch: &mut crate::backend::BufferPool,
        shape: &crate::backend::KdaShape,
        dk: &DeviceKda,
        dev_layer: &DeviceLayerWeights,
        plan: &[DeltaPlanRun],
        normed: &crate::backend::DeviceBuffer,
        x: &mut crate::backend::DeviceBuffer,
        rows: usize,
    ) -> Result<(), String> {
        let err = |e: crate::backend::BackendError| e.to_string();
        let resets = self.delta_claim(dk.ordinal, plan)?;
        let (width, conv_dim) = (shape.heads * shape.head_dim, shape.conv_dim());
        let project = |stream: &mut dyn crate::backend::Stream,
                       scratch: &mut crate::backend::BufferPool,
                       w: &crate::backend::DeviceWeight,
                       input: &crate::backend::DeviceBuffer|
         -> Result<crate::backend::DeviceBuffer, String> {
            let mut out = scratch.acquire(rows * w.rows * 4).map_err(err)?;
            stream.quantized_linear(&mut out, input, w, w.bias.as_ref()).map_err(err)?;
            Ok(out)
        };
        let mut qkv = scratch.acquire(rows * conv_dim * 4).map_err(err)?;
        for (part, w) in [&dk.q, &dk.k, &dk.v].into_iter().enumerate() {
            let block = project(stream, scratch, w, normed)?;
            stream.copy_columns(&mut qkv, conv_dim, part * width, &block, rows, width).map_err(err)?;
            scratch.recycle(block);
        }
        let low_rank = |stream: &mut dyn crate::backend::Stream,
                        scratch: &mut crate::backend::BufferPool,
                        down: Option<&crate::backend::DeviceWeight>,
                        up: &crate::backend::DeviceWeight|
         -> Result<crate::backend::DeviceBuffer, String> {
            match down {
                Some(down) => {
                    let mid = project(stream, scratch, down, normed)?;
                    let out = project(stream, scratch, up, &mid)?;
                    scratch.recycle(mid);
                    Ok(out)
                }
                None => project(stream, scratch, up, normed),
            }
        };
        let forget = low_rank(stream, scratch, dk.forget_down.as_ref(), &dk.forget_up)?;
        let beta = project(stream, scratch, &dk.beta, normed)?;
        let gate = low_rank(stream, scratch, dk.gate_down.as_ref(), &dk.gate_up)?;
        let mut mixed = scratch.acquire(rows * conv_dim * 4).map_err(err)?;
        let mut out = scratch.acquire(rows * width * 4).map_err(err)?;
        let runs: Vec<crate::backend::DeltaRun<'_>> =
            plan.iter().zip(&resets).map(|(p, &reset)| crate::backend::DeltaRun { state: &p.buf, first_row: p.start, rows: p.n, reset }).collect();
        let weights = crate::backend::KdaWeights { conv: &dk.conv, dt_bias: &dk.dt_bias, rate: &dk.rate, norm: &dk.norm };
        stream
            .kda_layer(shape, &weights, &runs, dk.ordinal * shape.layer_state_bytes(), &qkv, &forget, &beta, &gate, &mut mixed, &mut out)
            .map_err(err)?;
        let residual_scale = self.cfg.residual_scale;
        let out_w = dev_layer.out.as_ref().ok_or("KDA layer without a device output projection")?;
        let fused = dev_layer.post_attn_norm.is_none() && stream.quantized_linear_add(x, &out, out_w, out_w.bias.as_ref(), residual_scale).map_err(err)?;
        if !fused {
            let mut branch = scratch.acquire(rows * self.cfg.hidden * 4).map_err(err)?;
            stream.quantized_linear(&mut branch, &out, out_w, out_w.bias.as_ref()).map_err(err)?;
            device_post_norm_inplace(
                stream,
                scratch,
                self.cfg.norm,
                &mut branch,
                dev_layer.post_attn_norm.as_ref(),
                dev_layer.post_attn_norm_b.as_ref(),
                self.cfg.post_eps,
                self.groups,
            )?;
            stream.residual_add(x, &branch, residual_scale).map_err(err)?;
            scratch.recycle(branch);
        }
        for buf in [qkv, forget, beta, gate, mixed, out] {
            scratch.recycle(buf);
        }
        Ok(())
    }

    /// Walks the runs of consecutive tokens of each sequence in a batch (sequences are identified by their first KV block) and calls
    /// `f(start, end, state)` for each with the layer's recurrent state, which `blank` creates for a sequence that starts here. The
    /// state is checked to hold exactly the tokens before the run and counts the run in.
    pub(super) fn for_each_run(
        &self,
        layer: usize,
        tables: &[i64],
        lengths: &[i64],
        blank: &dyn Fn() -> LayerState,
        mut f: impl FnMut(usize, usize, &mut LayerState),
    ) -> Result<(), String> {
        let rows = lengths.len();
        let mut store = self.states.lock().map_err(|_| "recurrent state lock poisoned")?;
        let mut start = 0;
        while start < rows {
            let first = tables[start * self.stride];
            let position = lengths[start] as usize - 1;
            let mut end = start + 1;
            while end < rows && tables[end * self.stride] == first && lengths[end] == lengths[end - 1] + 1 {
                end += 1;
            }
            store.clock += 1;
            let clock = store.clock;
            if !store.map.contains_key(&first) && store.map.len() >= self.state_capacity {
                if let Some(oldest) = store.map.iter().min_by_key(|(_, s)| s.used).map(|(k, _)| *k) {
                    store.map.remove(&oldest);
                }
            }
            let layers = self.cfg.layers;
            let seq = store.map.entry(first).or_insert_with(|| SeqState { used: clock, layers: (0..layers).map(|_| None).collect() });
            seq.used = clock;
            if position == 0 {
                seq.layers[layer] = None;
            }
            let state = seq.layers[layer].get_or_insert_with(blank);
            if state.len != position {
                return Err(format!("recurrent state out of sync (holds {} tokens, got position {position})", state.len));
            }
            state.len += end - start;
            f(start, end, state);
            start = end;
        }
        Ok(())
    }

    /// KDA mixer for a batch of tokens; the output projection is applied here.
    pub(super) fn kda_forward(
        &self,
        k: &Kda,
        out_projection: &Linear,
        layer: usize,
        x: &[f32],
        tables: &[i64],
        lengths: &[i64],
    ) -> Result<Vec<f32>, String> {
        let kc = self.cfg.hybrid.kda.as_ref().ok_or("missing KDA configuration")?;
        let rows = lengths.len();
        let width = kc.heads * kc.head_dim;
        let params = crate::cpu_kda::Layer {
            heads: kc.heads,
            k_dim: kc.head_dim,
            v_dim: kc.head_dim,
            conv_kernel: kc.conv_kernel,
            eps: self.cfg.eps,
            lower_bound: kc.lower_bound,
            conv: &k.conv,
            dt_bias: &k.dt_bias,
            rate: &k.rate,
            norm: &k.norm,
        };
        let conv_dim = params.conv_dim();
        let mut parts = Linear::forward_group(&[&k.q, &k.k, &k.v], x)?.into_iter();
        let (q, kk, v) = parts.next().zip(parts.next()).zip(parts.next()).map(|((a, b), c)| (a, b, c)).ok_or("KDA projection group")?;
        let mut qkv = Vec::with_capacity(rows * conv_dim);
        for r in 0..rows {
            qkv.extend_from_slice(&q[r * width..(r + 1) * width]);
            qkv.extend_from_slice(&kk[r * width..(r + 1) * width]);
            qkv.extend_from_slice(&v[r * width..(r + 1) * width]);
        }
        let forget = match &k.forget_down {
            Some(down) => k.forget_up.forward(&down.forward(x)?)?,
            None => k.forget_up.forward(x)?,
        };
        let beta = k.beta.forward(x)?;
        let gate = match &k.gate_down {
            Some(down) => k.gate_up.forward(&down.forward(x)?)?,
            None => k.gate_up.forward(x)?,
        };
        let mut out = vec![0.0f32; rows * width];
        let blank = || LayerState { len: 0, conv: vec![0.0; conv_dim * (kc.conv_kernel - 1)], rec: vec![0.0; kc.heads * kc.head_dim * kc.head_dim] };
        self.for_each_run(layer, tables, lengths, &blank, |start, end, state| {
            let n = end - start;
            // Chunked prefill on CPU for KDA
            const CHUNK_SIZE: usize = 64;
            for chunk_start in (0..n).step_by(CHUNK_SIZE) {
                let chunk_len = (n - chunk_start).min(CHUNK_SIZE);
                let cs = start + chunk_start;
                let ce = cs + chunk_len;
                crate::cpu_kda::run(
                    &params,
                    &mut state.conv,
                    &mut state.rec,
                    (
                        &qkv[cs * conv_dim..ce * conv_dim],
                        &forget[cs * width..ce * width],
                        &beta[cs * kc.heads..ce * kc.heads],
                        &gate[cs * width..ce * width],
                    ),
                    &mut out[cs * width..ce * width],
                    chunk_len,
                );
            }
        })?;
        out_projection.forward(&out)
    }
}
