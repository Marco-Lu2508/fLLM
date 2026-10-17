//! Kimi Delta Attention layers in the decoder (host path): weights, state handling and the call into `cpu_kda`.
//!
//! GGUF tensors of a KDA layer `blk.N`: `attn_q`, `attn_k`, `attn_v` (the three projections), `ssm_conv1d_{q,k,v}.weight`
//! (depthwise short convolutions, no bias), `ssm_f_a` / `ssm_f_b` (low-rank forget gate), `ssm_dt.bias`, `ssm_a`
//! (`-exp(A_log)` per head, the usual Mamba convention), `ssm_beta` (write strength), `ssm_g_a` / `ssm_g_b` (low-rank output
//! gate), `ssm_o_norm.weight` and `attn_output`.
use super::*;

pub(super) struct Kda {
    q: Linear,
    k: Linear,
    v: Linear,
    forget_down: Linear,
    forget_up: Linear,
    beta: Linear,
    gate_down: Linear,
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
        Ok(Some(Self {
            q: linear(&format!("{p}.attn_q"), width, h)?,
            k: linear(&format!("{p}.attn_k"), width, h)?,
            v: linear(&format!("{p}.attn_v"), width, h)?,
            forget_down: linear(&format!("{p}.ssm_f_a"), kc.head_dim, h)?,
            forget_up: linear(&format!("{p}.ssm_f_b"), width, kc.head_dim)?,
            beta: linear(&format!("{p}.ssm_beta"), kc.heads, h)?,
            gate_down: linear(&format!("{p}.ssm_g_a"), kc.head_dim, h)?,
            gate_up: linear(&format!("{p}.ssm_g_b"), width, kc.head_dim)?,
            conv,
            dt_bias: vector(mapping, &format!("{p}.ssm_dt.bias"), width)?,
            rate: a.iter().map(|v| -v).collect(),
            norm: vector(mapping, &format!("{p}.ssm_o_norm.weight"), kc.head_dim)?,
        }))
    }

    pub(super) fn linears_mut(&mut self) -> [&mut Linear; 8] {
        [
            &mut self.q,
            &mut self.k,
            &mut self.v,
            &mut self.forget_down,
            &mut self.forget_up,
            &mut self.beta,
            &mut self.gate_down,
            &mut self.gate_up,
        ]
    }
}

impl RustDecoder {
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
        let forget = k.forget_up.forward(&k.forget_down.forward(x)?)?;
        let beta = k.beta.forward(x)?;
        let gate = k.gate_up.forward(&k.gate_down.forward(x)?)?;
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
