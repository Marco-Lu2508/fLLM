//! Inkling on the host (see `rust/checkpoint/src/inkling.rs` for the model and the tensor names).
//!
//! A token passes a layer completely before the next token of the same sequence does, because four short convolutions carry state from
//! token to token: on the keys and the values of the attention (`kernel` taps), and on the output of the attention block and of the
//! feed-forward block (`block kernel` taps). Everything a sequence keeps is in the per-layer state: `conv` holds the last inputs of the four
//! convolutions (oldest first), `rec` the keys and values of all earlier tokens (`k | v` per token, after convolution and normalisation).
use super::*;

/// Evaluates a `Result` inside a token loop that returns `()`: the first error is kept and the token is abandoned.
macro_rules! tri {
    ($e:expr, $error:ident) => {
        match $e {
            Ok(v) => v,
            Err(e) => {
                $error.get_or_insert(e);
                return;
            }
        }
    };
}

/// Depthwise causal convolution whose result is added to its input: `x + sum_j w[c][j] * x[t - (taps - 1) + j]`.
struct Conv {
    taps: usize,
    weights: Vec<f32>,
}

impl Conv {
    fn load(mapping: &MappedGguf, name: &str, channels: usize, taps: usize) -> Result<Self, String> {
        Ok(Self { taps, weights: vector(mapping, name, channels * taps)? })
    }

    /// One token: `state` holds the `taps - 1` previous inputs oldest first; it is advanced.
    fn step(&self, state: &mut [f32], x: &[f32]) -> Vec<f32> {
        let (taps, c) = (self.taps, x.len());
        let mut out = vec![0.0f32; c];
        for ch in 0..c {
            let mut sum = 0.0;
            for j in 0..taps - 1 {
                sum += self.weights[ch * taps + j] * state[j * c + ch];
            }
            sum += self.weights[ch * taps + taps - 1] * x[ch];
            out[ch] = sum + x[ch];
        }
        if taps > 1 {
            state.copy_within(c.., 0);
            let last = (taps - 2) * c;
            state[last..last + c].copy_from_slice(x);
        }
        out
    }
}

struct InkMoe {
    router: Linear,
    scale: f32,
    bias: Vec<f32>,
    experts: Vec<Expert>,
    shared: Vec<Expert>,
}

struct InkLayer {
    sliding: bool,
    heads: usize,
    kv_heads: usize,
    hd: usize,
    q: Linear,
    k: Linear,
    v: Linear,
    r: Linear,
    o: Linear,
    qn: Vec<f32>,
    kn: Vec<f32>,
    /// `d_rel x extent` distance profiles.
    profile: Vec<f32>,
    extent: usize,
    k_conv: Conv,
    v_conv: Conv,
    attn_conv: Conv,
    ffn_conv: Conv,
    moe: Option<InkMoe>,
    /// Scale of a dense feed-forward.
    dense_scale: f32,
}

pub(super) struct InklingModel {
    layers: Vec<InkLayer>,
}

impl InkLayer {
    fn kv_width(&self) -> usize {
        self.kv_heads * self.hd
    }
    /// Offsets of the four convolution states in `LayerState::conv`: keys, values, attention output, feed-forward output.
    fn offsets(&self, hidden: usize) -> [usize; 5] {
        let (k, b) = (self.k_conv.taps - 1, self.attn_conv.taps - 1);
        let kv = self.kv_width();
        let a = k * kv;
        let c = a + k * kv;
        let d = c + b * hidden;
        [0, a, c, d, d + b * hidden]
    }
    fn blank(&self, hidden: usize) -> LayerState {
        LayerState { len: 0, conv: vec![0.0; self.offsets(hidden)[4]], rec: Vec::new() }
    }
}

impl InklingModel {
    pub(super) fn load(model: &GgufModel, cfg: &ModelConfig) -> Result<Self, String> {
        let ic = cfg.inkling.as_ref().ok_or("not an Inkling model")?;
        let mapping = model.mapping();
        let hidden = cfg.hidden;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("blk.{i}");
            let sliding = ic.sliding[i];
            let (heads, kv_heads, hd) = if sliding { (ic.swa_heads, ic.swa_kv_heads, ic.swa_dim) } else { (cfg.heads, ic.kv_heads, ic.head_dim) };
            let load = |name: &str, rows: usize, cols: usize| -> Result<Linear, String> {
                let mut l = Linear::load(mapping, &format!("{p}.{name}"), rows, cols)?;
                l.replace_weights(model.adapted_projection(&format!("{p}.{name}.weight")))?;
                l.enable_fast_cpu();
                Ok(l)
            };
            let extent = if sliding { ic.window } else { ic.rel_extent };
            let moe = if ic.sparse[i] {
                let rows = |name: &str, e: usize, n: usize, cols: usize| -> Result<Linear, String> {
                    let mut l = Linear::load_rows(mapping, &format!("{p}.{name}.weight"), None, e * n, n, cols)?;
                    l.enable_fast_cpu();
                    Ok(l)
                };
                let expert = |prefix: &str, e: usize| -> Result<Expert, String> {
                    let (gate, up, down) = if prefix == "exps" {
                        ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
                    } else {
                        ("ffn_shared_gate", "ffn_shared_up", "ffn_shared_down")
                    };
                    Ok(Expert { gate: Some(rows(gate, e, ic.moe_ff, hidden)?), up: rows(up, e, ic.moe_ff, hidden)?, down: rows(down, e, hidden, ic.moe_ff)? })
                };
                Some(InkMoe {
                    router: load("ffn_router", ic.routed + ic.shared, hidden)?,
                    scale: vector(mapping, &format!("{p}.ffn_router_scale.weight"), 1)?[0],
                    bias: vector(mapping, &format!("{p}.ffn_router_bias.bias"), ic.routed)?,
                    experts: (0..ic.routed).map(|e| expert("exps", e)).collect::<Result<_, _>>()?,
                    shared: (0..ic.shared).map(|e| expert("shared", e)).collect::<Result<_, _>>()?,
                })
            } else {
                None
            };
            layers.push(InkLayer {
                sliding,
                heads,
                kv_heads,
                hd,
                q: load("attn_query", heads * hd, hidden)?,
                k: load("attn_key", kv_heads * hd, hidden)?,
                v: load("attn_value", kv_heads * hd, hidden)?,
                r: load("attn_rel", heads * ic.d_rel, hidden)?,
                o: load("attn_out", hidden, heads * hd)?,
                qn: vector(mapping, &format!("{p}.attn_qn.weight"), hd)?,
                kn: vector(mapping, &format!("{p}.attn_kn.weight"), hd)?,
                profile: vector(mapping, &format!("{p}.attn_rel_profile.weight"), ic.d_rel * extent)?,
                extent,
                k_conv: Conv::load(mapping, &format!("{p}.attn_k_sconv.weight"), kv_heads * hd, ic.conv_kernel)?,
                v_conv: Conv::load(mapping, &format!("{p}.attn_v_sconv.weight"), kv_heads * hd, ic.conv_kernel)?,
                attn_conv: Conv::load(mapping, &format!("{p}.attn_sconv.weight"), hidden, ic.block_conv_kernel)?,
                ffn_conv: Conv::load(mapping, &format!("{p}.ffn_sconv.weight"), hidden, ic.block_conv_kernel)?,
                moe,
                dense_scale: if ic.sparse[i] { 1.0 } else { vector(mapping, &format!("{p}.ffn_scale.weight"), 1)?[0] },
            });
        }
        Ok(Self { layers })
    }
}

impl RustDecoder {
    /// All layers of an Inkling model for a batch of embeddings (rows x hidden); the result is the final stream before the output norm.
    pub(super) fn inkling_layers(&self, x: &[f32], positions: &[usize], tables: &[i64], lengths: &[i64]) -> Result<Vec<f32>, String> {
        let (cfg, model) = (&self.cfg, self.inkling.as_deref().ok_or("Inkling weights missing")?);
        let ic = cfg.inkling.as_ref().ok_or("Inkling configuration missing")?;
        let (rows, hidden, eps) = (positions.len(), cfg.hidden, cfg.eps);
        let mut x = x.to_vec();
        for (i, (layer, il)) in self.layers.iter().zip(&model.layers).enumerate() {
            let mut normed = x.clone();
            if let Some(norm) = &layer.attn_norm {
                self.norm(&mut normed, norm);
            }
            let (hd, heads, kv_heads) = (il.hd, il.heads, il.kv_heads);
            let (kvw, group) = (il.kv_width(), il.heads / il.kv_heads);
            let mut q = il.q.forward(&normed)?;
            rms(&mut q, &il.qn, eps);
            let k_raw = il.k.forward(&normed)?;
            let v_raw = il.v.forward(&normed)?;
            let rel = il.r.forward(&normed)?;
            let offsets = il.offsets(hidden);
            let scale = 1.0 / hd as f32;
            let mut error: Option<String> = None;
            self.for_each_run(i, tables, lengths, &|| il.blank(hidden), |start, end, state| {
                for t in start..end {
                    // keys and values: convolution (with the input added back), keys are normalised per head
                    let mut k = il.k_conv.step(&mut state.conv[offsets[0]..offsets[1]], &k_raw[t * kvw..(t + 1) * kvw]);
                    rms(&mut k, &il.kn, eps);
                    let v = il.v_conv.step(&mut state.conv[offsets[1]..offsets[2]], &v_raw[t * kvw..(t + 1) * kvw]);
                    state.rec.extend_from_slice(&k);
                    state.rec.extend_from_slice(&v);
                    let n = state.rec.len() / (2 * kvw);
                    let position = n - 1;
                    let tau = match ic.log_floor {
                        floor if floor > 0 && !il.sliding => 1.0 + ic.log_alpha * ((position + 1) as f32 / floor as f32).max(1.0).ln(),
                        _ => 1.0,
                    };
                    let first = if il.sliding { n.saturating_sub(ic.window) } else { 0 };
                    let mut attn = vec![0.0f32; heads * hd];
                    for h in 0..heads {
                        let kh = h / group;
                        let query: Vec<f32> = q[(t * heads + h) * hd..(t * heads + h + 1) * hd].iter().map(|v| v * tau).collect();
                        let relative = &rel[(t * heads + h) * ic.d_rel..(t * heads + h + 1) * ic.d_rel];
                        let mut scores = Vec::with_capacity(n - first);
                        for j in first..n {
                            let key = &state.rec[j * 2 * kvw + kh * hd..j * 2 * kvw + (kh + 1) * hd];
                            let distance = position - j;
                            let bias = if distance < il.extent { (0..ic.d_rel).map(|d| relative[d] * il.profile[d * il.extent + distance]).sum::<f32>() * tau } else { 0.0 };
                            scores.push(compute::dot(&query, key) * scale + bias);
                        }
                        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let mut sum = 0.0;
                        for s in &mut scores {
                            *s = (*s - max).exp();
                            sum += *s;
                        }
                        let out = &mut attn[h * hd..(h + 1) * hd];
                        for (s, j) in scores.iter().zip(first..n) {
                            let value = &state.rec[j * 2 * kvw + kvw + kh * hd..j * 2 * kvw + kvw + (kh + 1) * hd];
                            let w = s / sum;
                            out.iter_mut().zip(value).for_each(|(o, v)| *o += w * v);
                        }
                    }
                    let _ = kv_heads;
                    let projected = tri!(il.o.forward(&attn), error);
                    let projected = il.attn_conv.step(&mut state.conv[offsets[2]..offsets[3]], &projected);
                    let row = &mut x[t * hidden..(t + 1) * hidden];
                    row.iter_mut().zip(&projected).for_each(|(a, b)| *a += b);
                    // feed-forward
                    let mut f = row.to_vec();
                    if let Some(norm) = &layer.ffn_norm {
                        self.norm(&mut f, norm);
                    }
                    let mlp = match &il.moe {
                        Some(moe) => tri!(self.ink_moe(moe, &f, ic), error),
                        None => tri!(self.ink_dense(layer, &f, il.dense_scale), error),
                    };
                    let mlp = il.ffn_conv.step(&mut state.conv[offsets[3]..offsets[4]], &mlp);
                    row.iter_mut().zip(&mlp).for_each(|(a, b)| *a += b);
                }
            })?;
            if let Some(e) = error {
                return Err(e);
            }
            let _ = rows;
        }
        Ok(x)
    }

    fn ink_dense(&self, layer: &Layer, f: &[f32], scale: f32) -> Result<Vec<f32>, String> {
        let Ffn::Gated { gate, up, down } = &layer.ffn else {
            return Err("Inkling dense layer without gate, up and down projections".into());
        };
        let mut g = gate.forward(f)?;
        let u = up.forward(f)?;
        self.gate_activation_clamped(&mut g, &u, false, false);
        let mut out = down.forward(&g)?;
        out.iter_mut().for_each(|v| *v *= scale);
        Ok(out)
    }

    /// Sigmoid routing: the best `used` routed experts by score plus bias, then the log-sigmoid logits of the chosen experts and of the shared
    /// experts are normalised together (a softmax over the log-probabilities) and scaled by `route_scale * global_scale`.
    fn ink_moe(&self, moe: &InkMoe, f: &[f32], ic: &crate::config::InklingConfig) -> Result<Vec<f32>, String> {
        let logits = moe.router.forward(f)?;
        let routed = &logits[..ic.routed];
        let scores: Vec<f32> = routed.iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect();
        let mut order: Vec<usize> = (0..ic.routed).collect();
        let choice = |i: usize| scores[i] + moe.bias[i];
        order.sort_by(|&a, &b| choice(b).total_cmp(&choice(a)).then(a.cmp(&b)));
        order.truncate(ic.used);
        let log_sigmoid = |l: f32| -(1.0 + (-l).exp()).ln();
        let chosen_logits: Vec<f32> = order.iter().map(|&i| routed[i]).chain(logits[ic.routed..].iter().copied()).collect();
        let logp: Vec<f32> = chosen_logits.iter().map(|&l| log_sigmoid(l)).collect();
        let max = logp.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lse = max + logp.iter().map(|l| (l - max).exp()).sum::<f32>().ln();
        let weights: Vec<f32> = logp.iter().map(|l| (l - lse).exp() * ic.route_scale * moe.scale).collect();
        let hidden = f.len();
        let mut out = vec![0.0f32; hidden];
        let mut workspace = compute::ProjectionGroupWorkspace::default();
        for (n, &e) in order.iter().enumerate() {
            let result = self.ink_expert(&moe.experts[e], f, &mut workspace)?;
            out.iter_mut().zip(&result).for_each(|(o, r)| *o += weights[n] * r);
        }
        for (s, expert) in moe.shared.iter().enumerate() {
            let result = self.ink_expert(expert, f, &mut workspace)?;
            let gamma = weights[ic.used + s];
            out.iter_mut().zip(&result).for_each(|(o, r)| *o += gamma * r);
        }
        Ok(out)
    }

    fn ink_expert(&self, e: &Expert, f: &[f32], workspace: &mut compute::ProjectionGroupWorkspace) -> Result<Vec<f32>, String> {
        self.expert_forward_clamped(e, f, workspace, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convolution_adds_its_input_and_slides_the_state() {
        // two channels, three taps: out[t] = x[t] + w0 * x[t - 2] + w1 * x[t - 1] + w2 * x[t]
        let conv = Conv { taps: 3, weights: vec![1.0, 2.0, 3.0, 0.5, 0.0, -1.0] };
        let mut state = vec![0.0f32; 2 * 2];
        let a = conv.step(&mut state, &[1.0, 10.0]);
        assert_eq!(a, vec![1.0 + 3.0, -10.0 + 10.0]);
        let b = conv.step(&mut state, &[2.0, 20.0]);
        // channel 0: x1 + 1*0 + 2*1 + 3*2; channel 1: x1 + 0.5*0 + 0*10 - 20
        assert_eq!(b, vec![2.0 + 2.0 + 6.0, 20.0 - 20.0 + 0.0]);
        let c = conv.step(&mut state, &[0.0, 0.0]);
        assert_eq!(c, vec![1.0 * 1.0 + 2.0 * 2.0, 0.5 * 10.0 + 0.0 * 20.0]);
    }
}
