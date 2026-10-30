//! MiniMax-M3 on the host (see `rust/checkpoint/src/minimax3.rs` for the model and the tensor names).
//!
//! Keys and values (after the norm and the rotary embedding) and, on block-sparse layers, the indexer keys of all earlier tokens of a
//! sequence are kept in the per-layer state (`rec`: `k | v | indexer key` per token).
use super::*;

/// `(up + 1) * gate * sigmoid(alpha * gate)` after clamping, over a gate and an up vector.
fn swiglu_alpha(gate: &mut [f32], up: &[f32], alpha: f32, limit: f32) {
    for (g, u) in gate.iter_mut().zip(up) {
        let gate = g.min(limit);
        let up = u.clamp(-limit, limit);
        *g = (up + 1.0) * gate / (1.0 + (-alpha * gate).exp());
    }
}

struct M3Moe {
    router: Linear,
    bias: Vec<f32>,
    experts: Vec<Expert>,
    shared: Expert,
}

struct M3Indexer {
    q: Linear,
    k: Linear,
    qn: Vec<f32>,
    kn: Vec<f32>,
}

struct M3Layer {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    qn: Vec<f32>,
    kn: Vec<f32>,
    indexer: Option<M3Indexer>,
    moe: Option<M3Moe>,
}

pub(super) struct Minimax3Model {
    layers: Vec<M3Layer>,
    freq: Vec<f32>,
}

impl Minimax3Model {
    pub(super) fn load(model: &GgufModel, cfg: &ModelConfig) -> Result<Self, String> {
        let mc = cfg.minimax3.as_ref().ok_or("not a MiniMax-M3 model")?;
        let mapping = model.mapping();
        let hidden = cfg.hidden;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("blk.{i}");
            let load = |name: &str, rows: usize, cols: usize| -> Result<Linear, String> {
                let mut l = Linear::load(mapping, &format!("{p}.{name}"), rows, cols)?;
                l.replace_weights(model.adapted_projection(&format!("{p}.{name}.weight")))?;
                l.enable_fast_cpu();
                Ok(l)
            };
            let rows = |name: &str, e: usize, n: usize, cols: usize| -> Result<Linear, String> {
                let mut l = Linear::load_rows(mapping, &format!("{p}.{name}.weight"), None, e * n, n, cols)?;
                l.enable_fast_cpu();
                Ok(l)
            };
            let moe = if mc.sparse_mlp[i] {
                let (ff, sff) = (mc.moe_ff, mc.shared_ff);
                Some(M3Moe {
                    router: load("ffn_router", mc.experts, hidden)?,
                    bias: vector(mapping, &format!("{p}.ffn_router_bias.bias"), mc.experts)?,
                    experts: (0..mc.experts)
                        .map(|e| Ok(Expert { gate: Some(rows("ffn_gate_exps", e, ff, hidden)?), up: rows("ffn_up_exps", e, ff, hidden)?, down: rows("ffn_down_exps", e, hidden, ff)? }))
                        .collect::<Result<_, String>>()?,
                    shared: Expert { gate: Some(rows("ffn_gate_shexp", 0, sff, hidden)?), up: rows("ffn_up_shexp", 0, sff, hidden)?, down: rows("ffn_shared_down", 0, hidden, sff)? },
                })
            } else {
                None
            };
            let indexer = if mc.sparse_attn[i] {
                Some(M3Indexer {
                    q: load("indexer.attn_q", mc.idx_heads * mc.idx_dim, hidden)?,
                    k: load("indexer.attn_k", mc.idx_dim, hidden)?,
                    qn: vector(mapping, &format!("{p}.indexer.q_norm.weight"), mc.idx_dim)?,
                    kn: vector(mapping, &format!("{p}.indexer.k_norm.weight"), mc.idx_dim)?,
                })
            } else {
                None
            };
            layers.push(M3Layer {
                q: load("attn_query", cfg.heads * mc.head_dim, hidden)?,
                k: load("attn_key", mc.kv_heads * mc.head_dim, hidden)?,
                v: load("attn_value", mc.kv_heads * mc.head_dim, hidden)?,
                o: load("attn_out", hidden, cfg.heads * mc.head_dim)?,
                qn: vector(mapping, &format!("{p}.attn_qn.weight"), mc.head_dim)?,
                kn: vector(mapping, &format!("{p}.attn_kn.weight"), mc.head_dim)?,
                indexer,
                moe,
            });
        }
        let half = mc.rot / 2;
        let freq = (0..half).map(|i| mc.theta.powf(-2.0 * i as f32 / mc.rot as f32)).collect();
        Ok(Self { layers, freq })
    }
}

/// Half-split rotary pairs `(i, i + rot / 2)` of the first `rot` values of `v` at `position`.
fn rotate_half_pairs(v: &mut [f32], rot: usize, position: usize, freq: &[f32]) {
    let half = rot / 2;
    for (i, &f) in freq.iter().enumerate().take(half) {
        let angle = position as f32 * f;
        let (c, s) = (angle.cos(), angle.sin());
        let (a, b) = (v[i], v[i + half]);
        v[i] = a * c - b * s;
        v[i + half] = b * c + a * s;
    }
}

impl RustDecoder {
    /// All layers of a MiniMax-M3 model for a batch of embeddings (rows x hidden); the result is the stream before the output norm.
    pub(super) fn minimax3_layers(&self, x: &[f32], positions: &[usize], tables: &[i64], lengths: &[i64]) -> Result<Vec<f32>, String> {
        let (cfg, model) = (&self.cfg, self.minimax3.as_deref().ok_or("MiniMax-M3 weights missing")?);
        let mc = cfg.minimax3.as_ref().ok_or("MiniMax-M3 configuration missing")?;
        let (hidden, eps, heads) = (cfg.hidden, cfg.eps, cfg.heads);
        let (hd, kv, rot) = (mc.head_dim, mc.kv_heads, mc.rot);
        let group = heads / kv;
        let mut x = x.to_vec();
        for (i, (layer, ml)) in self.layers.iter().zip(&model.layers).enumerate() {
            let mut normed = x.clone();
            if let Some(norm) = &layer.attn_norm {
                self.norm(&mut normed, norm);
            }
            let mut q = ml.q.forward(&normed)?;
            rms(&mut q, &ml.qn, eps);
            let mut k = ml.k.forward(&normed)?;
            rms(&mut k, &ml.kn, eps);
            let v = ml.v.forward(&normed)?;
            let (mut iq, mut ik) = match &ml.indexer {
                Some(ix) => {
                    let mut iq = ix.q.forward(&normed)?;
                    rms(&mut iq, &ix.qn, eps);
                    let mut ik = ix.k.forward(&normed)?;
                    rms(&mut ik, &ix.kn, eps);
                    (iq, ik)
                }
                None => (Vec::new(), Vec::new()),
            };
            for (t, &position) in positions.iter().enumerate() {
                for h in 0..heads {
                    rotate_half_pairs(&mut q[(t * heads + h) * hd..(t * heads + h + 1) * hd], rot, position, &model.freq);
                }
                for h in 0..kv {
                    rotate_half_pairs(&mut k[(t * kv + h) * hd..(t * kv + h + 1) * hd], rot, position, &model.freq);
                }
                if ml.indexer.is_some() {
                    for h in 0..mc.idx_heads {
                        rotate_half_pairs(&mut iq[(t * mc.idx_heads + h) * mc.idx_dim..(t * mc.idx_heads + h + 1) * mc.idx_dim], rot.min(mc.idx_dim), position, &model.freq);
                    }
                    rotate_half_pairs(&mut ik[t * mc.idx_dim..(t + 1) * mc.idx_dim], rot.min(mc.idx_dim), position, &model.freq);
                }
            }
            let width = 2 * kv * hd + if ml.indexer.is_some() { mc.idx_dim } else { 0 };
            let mut attn = vec![0.0f32; positions.len() * heads * hd];
            let scale = (hd as f32).powf(-0.5);
            self.for_each_run(i, tables, lengths, &|| LayerState { len: 0, conv: Vec::new(), rec: Vec::new() }, |start, end, state| {
                for t in start..end {
                    state.rec.extend_from_slice(&k[t * kv * hd..(t + 1) * kv * hd]);
                    state.rec.extend_from_slice(&v[t * kv * hd..(t + 1) * kv * hd]);
                    if ml.indexer.is_some() {
                        state.rec.extend_from_slice(&ik[t * mc.idx_dim..(t + 1) * mc.idx_dim]);
                    }
                    let n = state.rec.len() / width;
                    let position = n - 1;
                    // block selection per indexer head
                    // `FLLM_DSA=dense` attends to every key (an approximation: the model selects blocks)
                    let selected: Vec<Vec<bool>> = if ml.indexer.is_some() && !self.dsa_dense {
                        let blocks = n.div_ceil(mc.block);
                        (0..mc.idx_heads)
                            .map(|h| {
                                let query = &iq[(t * mc.idx_heads + h) * mc.idx_dim..(t * mc.idx_heads + h + 1) * mc.idx_dim];
                                let mut score = vec![f32::NEG_INFINITY; blocks];
                                for j in 0..n {
                                    let key = &state.rec[j * width + 2 * kv * hd..(j + 1) * width];
                                    let s = compute::dot(query, key);
                                    score[j / mc.block] = score[j / mc.block].max(s);
                                }
                                let q_block = position / mc.block;
                                for l in 0..mc.local {
                                    score[q_block.saturating_sub(l)] = f32::INFINITY;
                                }
                                let mut order: Vec<usize> = (0..blocks).collect();
                                order.sort_by(|&a, &b| score[b].total_cmp(&score[a]).then(a.cmp(&b)));
                                let mut keep = vec![false; blocks];
                                for &b in order.iter().take(mc.topk.min(blocks)) {
                                    if score[b] > f32::NEG_INFINITY {
                                        keep[b] = true;
                                    }
                                }
                                keep
                            })
                            .collect()
                    } else {
                        Vec::new()
                    };
                    for h in 0..heads {
                        let kh = h / group;
                        let query = &q[(t * heads + h) * hd..(t * heads + h + 1) * hd];
                        let keep = (!selected.is_empty()).then(|| &selected[h / (heads / mc.idx_heads)]);
                        let visible: Vec<usize> = (0..n).filter(|&j| keep.is_none_or(|k| k[j / mc.block])).collect();
                        let mut scores: Vec<f32> = visible.iter().map(|&j| compute::dot(query, &state.rec[j * width + kh * hd..j * width + (kh + 1) * hd]) * scale).collect();
                        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let mut sum = 0.0;
                        for s in &mut scores {
                            *s = (*s - max).exp();
                            sum += *s;
                        }
                        let out = &mut attn[(t * heads + h) * hd..(t * heads + h + 1) * hd];
                        for (s, &j) in scores.iter().zip(&visible) {
                            let value = &state.rec[j * width + kv * hd + kh * hd..j * width + kv * hd + (kh + 1) * hd];
                            out.iter_mut().zip(value).for_each(|(o, x)| *o += s / sum * x);
                        }
                    }
                }
            })?;
            let projected = ml.o.forward(&attn)?;
            x.iter_mut().zip(&projected).for_each(|(a, b)| *a += b);
            let mut f = x.clone();
            if let Some(norm) = &layer.ffn_norm {
                self.norm(&mut f, norm);
            }
            let mlp = match &ml.moe {
                Some(moe) => self.m3_moe(moe, &f, mc)?,
                None => {
                    let Ffn::Gated { gate, up, down } = &layer.ffn else {
                        return Err("MiniMax-M3 dense layer without gate, up and down projections".into());
                    };
                    let mut g = gate.forward(&f)?;
                    let u = up.forward(&f)?;
                    swiglu_alpha(&mut g, &u, mc.alpha, mc.limit);
                    down.forward(&g)?
                }
            };
            x.iter_mut().zip(&mlp).for_each(|(a, b)| *a += b);
            let _ = hidden;
        }
        Ok(x)
    }

    fn m3_expert(&self, e: &Expert, f: &[f32], mc: &crate::config::Minimax3Config) -> Result<Vec<f32>, String> {
        let gate = e.gate.as_ref().ok_or("expert without a gate projection")?;
        let mut g = gate.forward(f)?;
        let u = e.up.forward(f)?;
        swiglu_alpha(&mut g, &u, mc.alpha, mc.limit);
        e.down.forward(&g)
    }

    /// Sigmoid routing with a selection bias, weights renormalised over the chosen experts, routed output scaled, plus the shared expert.
    fn m3_moe(&self, moe: &M3Moe, f: &[f32], mc: &crate::config::Minimax3Config) -> Result<Vec<f32>, String> {
        let hidden = self.cfg.hidden;
        let rows = f.len() / hidden;
        let logits = moe.router.forward(f)?;
        let mut out = vec![0.0f32; f.len()];
        for t in 0..rows {
            let row = &f[t * hidden..(t + 1) * hidden];
            let scores: Vec<f32> = logits[t * mc.experts..(t + 1) * mc.experts].iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect();
            let mut order: Vec<usize> = (0..mc.experts).collect();
            order.sort_by(|&a, &b| (scores[b] + moe.bias[b]).total_cmp(&(scores[a] + moe.bias[a])).then(a.cmp(&b)));
            order.truncate(mc.used);
            let total: f32 = order.iter().map(|&e| scores[e]).sum();
            let o = &mut out[t * hidden..(t + 1) * hidden];
            for &e in &order {
                let result = self.m3_expert(&moe.experts[e], row, mc)?;
                o.iter_mut().zip(&result).for_each(|(a, r)| *a += scores[e] / total * mc.route_scale * r);
            }
            let shared = self.m3_expert(&moe.shared, row, mc)?;
            o.iter_mut().zip(&shared).for_each(|(a, r)| *a += r);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swiglu_clamps_and_adds_one_to_the_up_projection() {
        let mut gate = [10.0f32, -1.0, 0.0];
        swiglu_alpha(&mut gate, &[0.5, 100.0, -9.0], 1.702, 3.0);
        // gate 10 -> 3, up 0.5; gate -1, up 100 -> 3; gate 0, up -9 -> -3
        let glu = |g: f32| g / (1.0 + (-1.702 * g).exp());
        assert!((gate[0] - 1.5 * glu(3.0)).abs() < 1e-6);
        assert!((gate[1] - 4.0 * glu(-1.0)).abs() < 1e-6);
        assert!(gate[2].abs() < 1e-7);
    }

    #[test]
    fn half_split_rotation_pairs_i_with_i_plus_half_and_keeps_the_tail() {
        let mut v = [1.0f32, 0.0, 0.0, 1.0, 5.0, 6.0];
        // rot 4: pairs (0, 2) and (1, 3); a quarter turn for the first pair, none for the second
        rotate_half_pairs(&mut v, 4, 1, &[std::f32::consts::FRAC_PI_2, 0.0]);
        assert!((v[0] - 0.0).abs() < 1e-6 && (v[2] - 1.0).abs() < 1e-6);
        assert!((v[1] - 0.0).abs() < 1e-6 && (v[3] - 1.0).abs() < 1e-6);
        assert_eq!(&v[4..], &[5.0, 6.0]);
    }
}
