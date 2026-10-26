//! Gemma 4 additions to the host decoder: per-layer embeddings (PLE), the expert block that runs next to the dense MLP,
//! and the per-layer output scalar. The per-layer attention geometry and the shared-KV layers are handled by the
//! common attention code (`ModelConfig::layer_*`).
//!
//! Layer order of a Gemma 4 block (`x` = residual stream), as in the `transformers` reference:
//!
//! ```text
//! x += post_attention_norm(attn(input_norm(x)))
//! d  = post_ffw_norm_1(mlp(ffn_norm(x)))                       -- dense MLP
//! e  = post_ffw_norm_2(experts(pre_ffw_norm_2(x), router(x)))  -- only when the layer has experts
//! x += post_ffw_norm(d + e)
//! x += post_norm(proj(act(inp_gate(x)) * per_layer_input))     -- only with per-layer embeddings
//! x *= layer_output_scale
//! ```
use super::*;

/// Per-layer input path of one layer (`inp_gate`, `proj`, `post_norm`).
pub(super) struct PleLayer {
    gate: Linear,
    proj: Linear,
    norm: Norm,
}

/// Expert block of a Gemma 4 layer. The router normalises the residual without a learned weight, scales it by
/// `router_scale / sqrt(hidden)`, takes a softmax over all experts, keeps the best `used`, renormalises them and multiplies each
/// by its `expert_scale`.
pub(super) struct GemmaMoe {
    router: Linear,
    router_scale: Vec<f32>,
    expert_scale: Vec<f32>,
    experts: Vec<Expert>,
    used: usize,
    pre_norm: Norm,
    post_norm_dense: Norm,
    post_norm_experts: Norm,
}

/// Gemma 3n AltUp predict / correct modules of one layer.
pub(super) struct AltupLayer {
    predict: Linear,
    correct: Linear,
    router: Linear,
    router_norm: Norm,
    correct_scale: Vec<f32>,
}

/// Gemma 3n LAuReL: low-rank residual `x + norm(right(left(x)))`.
pub(super) struct Laurel {
    left: Linear,
    right: Linear,
    norm: Norm,
}

pub(super) struct GemmaLayer {
    scalar: Option<f32>,
    ple: Option<PleLayer>,
    moe: Option<GemmaMoe>,
    altup: Option<AltupLayer>,
    laurel: Option<Laurel>,
    /// Gaussian top-k factor of the gate activations (0: dense).
    sparsity: f32,
}

/// Gemma 3n projections that create and merge the parallel residual streams.
pub(super) struct AltupGlobal {
    up: Vec<Linear>,
    down: Vec<Linear>,
    streams: usize,
    active: usize,
}

/// Token-identity and projection tables of the per-layer embeddings.
pub(super) struct PleGlobal {
    table: Embedding,
    projection: Linear,
    norm: Vec<f32>,
    pub(super) dim: usize,
}

fn tensor_exists(mapping: &MappedGguf, name: &str) -> bool {
    has_tensor(mapping, name)
}

/// Load the Gemma additions of layer `i` (`None` for other architectures).
pub(super) fn load_layer(model: &GgufModel, cfg: &ModelConfig, p: &str) -> Result<Option<Box<GemmaLayer>>, String> {
    let Some(gemma) = &cfg.gemma else {
        return Ok(None);
    };
    let mapping = model.mapping();
    let hidden = cfg.hidden;
    let linear = |name: &str, rows: usize, cols: usize| -> Result<Linear, String> {
        let mut l = Linear::load(mapping, name, rows, cols)?;
        l.replace_weights(model.adapted_projection(&format!("{name}.weight")))?;
        l.enable_fast_cpu();
        Ok(l)
    };
    let scalar = if gemma.layer_scalar {
        let name = format!("{p}.layer_output_scale.weight");
        Some(vector(mapping, &name, 1)?[0])
    } else {
        None
    };
    let ple = if gemma.ple_dim > 0 && tensor_exists(mapping, &format!("{p}.inp_gate.weight")) {
        Some(PleLayer {
            gate: linear(&format!("{p}.inp_gate"), gemma.ple_dim, hidden)?,
            proj: linear(&format!("{p}.proj"), hidden, gemma.ple_dim)?,
            norm: load_norm(mapping, &format!("{p}.post_norm"), hidden)?.ok_or("missing per-layer post_norm")?,
        })
    } else {
        None
    };
    let moe = if let (Some(m), true) = (&cfg.moe, tensor_exists(mapping, &format!("{p}.pre_ffw_norm_2.weight"))) {
        let width = m.expert_ff;
        let mut experts = Vec::with_capacity(m.experts);
        for e in 0..m.experts {
            let rows = |kind: &str, first: usize, n: usize, cols: usize| -> Result<Linear, String> {
                let name = if kind == "down" { "ffn_down_exps" } else { "ffn_gate_up_exps" };
                let mut l = Linear::load_rows(mapping, &format!("{p}.{name}.weight"), None, first, n, cols)?;
                l.enable_fast_cpu();
                Ok(l)
            };
            experts.push(Expert {
                gate: Some(rows("gate", e * 2 * width, width, hidden)?),
                up: rows("up", e * 2 * width + width, width, hidden)?,
                down: rows("down", e * hidden, hidden, width)?,
            });
        }
        let norm = |base: &str| -> Result<Norm, String> {
            load_norm(mapping, &format!("{p}.{base}"), hidden)?.ok_or_else(|| format!("missing {p}.{base}.weight"))
        };
        Some(GemmaMoe {
            router: linear(&format!("{p}.ffn_gate_inp"), m.experts, hidden)?,
            router_scale: vector(mapping, &format!("{p}.ffn_gate_inp.scale"), hidden)?,
            expert_scale: vector(mapping, &format!("{p}.ffn_down_exps.scale"), m.experts)?,
            experts,
            used: m.used,
            pre_norm: norm("pre_ffw_norm_2")?,
            post_norm_dense: norm("post_ffw_norm_1")?,
            post_norm_experts: norm("post_ffw_norm_2")?,
        })
    } else {
        None
    };
    let layer: usize = p.strip_prefix("blk.").and_then(|s| s.parse().ok()).unwrap_or(0);
    let altup = if gemma.altup_inputs > 0 {
        let n = gemma.altup_inputs;
        Some(AltupLayer {
            predict: linear(&format!("{p}.altup_predict_coef"), n * n, n)?,
            correct: linear(&format!("{p}.altup_correct_coef"), n, n)?,
            router: linear(&format!("{p}.altup_router"), n, hidden)?,
            router_norm: load_norm(mapping, &format!("{p}.altup_router_norm"), hidden)?.ok_or("missing altup_router_norm")?,
            correct_scale: vector(mapping, &format!("{p}.altup_correct_scale.weight"), hidden)?,
        })
    } else {
        None
    };
    let laurel = if gemma.laurel_rank > 0 {
        Some(Laurel {
            left: linear(&format!("{p}.laurel_l"), gemma.laurel_rank, hidden)?,
            right: linear(&format!("{p}.laurel_r"), hidden, gemma.laurel_rank)?,
            norm: load_norm(mapping, &format!("{p}.laurel_post_norm"), hidden)?.ok_or("missing laurel_post_norm")?,
        })
    } else {
        None
    };
    let sparsity = gemma.sparsity.get(layer).copied().unwrap_or(0.0);
    Ok(Some(Box::new(GemmaLayer { scalar, ple, moe, altup, laurel, sparsity })))
}

/// Load the model-wide per-layer embedding tensors.
pub(super) fn load_global(model: &GgufModel, cfg: &ModelConfig) -> Result<Option<PleGlobal>, String> {
    let Some(gemma) = &cfg.gemma else {
        return Ok(None);
    };
    if gemma.ple_dim == 0 {
        return Ok(None);
    }
    let mapping = model.mapping();
    let total = cfg.layers * gemma.ple_dim;
    let mut projection = Linear::load(mapping, "per_layer_model_proj", total, cfg.hidden)?;
    projection.replace_weights(model.adapted_projection("per_layer_model_proj.weight"))?;
    projection.enable_fast_cpu();
    Ok(Some(PleGlobal {
        table: Embedding::load_with_cols(model, "per_layer_token_embd.weight", total)?,
        projection,
        norm: vector(mapping, "per_layer_proj_norm.weight", gemma.ple_dim)?,
        dim: gemma.ple_dim,
    }))
}

impl PleGlobal {
    /// The projection of the embeddings to the per-layer inputs.
    pub(super) fn projection(&self) -> &Linear {
        &self.projection
    }
}

impl GemmaLayer {
    /// What the device forward pass does not run of this layer (it has no kernels for these parts yet), if anything. The per-layer
    /// inputs and the output scale run on the device.
    pub(super) fn device_gap(&self) -> Option<&'static str> {
        if self.altup.is_some() {
            Some("Gemma AltUp residual streams")
        } else if self.laurel.is_some() {
            Some("Gemma LAuReL residual")
        } else if self.moe.is_some() {
            Some("the Gemma mixture-of-experts branch")
        } else if self.sparsity > 0.0 {
            Some("Gemma activation sparsity")
        } else {
            None
        }
    }

    /// Gate, projection and norm of the per-layer input path.
    pub(super) fn ple_parts(&self) -> Option<(&Linear, &Linear, &Norm)> {
        self.ple.as_ref().map(|p| (&p.gate, &p.proj, &p.norm))
    }

    /// The layer output scale.
    pub(super) fn output_scale(&self) -> Option<f32> {
        self.scalar
    }
}

impl RustDecoder {
    /// Per-layer inputs of a batch: `rows x (layers * ple_dim)`. `embedded` are the (scaled) input embeddings.
    pub(super) fn ple_inputs(&self, ids: &[i64], embedded: &[f32]) -> Result<Option<Vec<f32>>, String> {
        let Some(ple) = &self.ple else {
            return Ok(None);
        };
        let projected = ple.projection.forward(embedded)?;
        self.ple_inputs_from(ids, projected)
    }

    /// [`Self::ple_inputs`] from the projection of the embeddings (`rows x (layers * ple_dim)`), computed wherever it was.
    pub(super) fn ple_inputs_from(&self, ids: &[i64], mut projected: Vec<f32>) -> Result<Option<Vec<f32>>, String> {
        let Some(ple) = &self.ple else {
            return Ok(None);
        };
        let cfg = &self.cfg;
        let total = cfg.layers * ple.dim;
        let rows = ids.len();
        if projected.len() != rows * total {
            return Err("per-layer projection of the wrong size".into());
        }
        let root = (cfg.hidden as f32).powf(-0.5);
        projected.iter_mut().for_each(|v| *v *= root);
        rms(&mut projected, &ple.norm, cfg.eps);
        let token_scale = (ple.dim as f32).sqrt();
        let mix = 0.5f32.sqrt();
        for (r, &id) in ids.iter().enumerate() {
            let identity = ple.table.row(id as usize)?;
            for (v, e) in projected[r * total..(r + 1) * total].iter_mut().zip(&identity) {
                *v = (*v + e * token_scale) * mix;
            }
        }
        debug_assert_eq!(projected.len(), rows * total);
        Ok(Some(projected))
    }

    /// `dense = post_ffw_norm_1(dense) + post_ffw_norm_2(experts(pre_ffw_norm_2(residual), router(residual)))`.
    pub(super) fn gemma_moe_branch(
        &self,
        moe: &GemmaMoe,
        residual: &[f32],
        dense: &mut [f32],
        workspace: &mut compute::ProjectionGroupWorkspace,
    ) -> Result<(), String> {
        let cfg = &self.cfg;
        let hidden = cfg.hidden;
        let rows = residual.len() / hidden;
        self.post_norm(dense, &moe.post_norm_dense);
        // router input: weightless RMS norm of the residual, learned scale, 1 / sqrt(hidden)
        let mut routed_in = residual.to_vec();
        let root = (hidden as f32).powf(-0.5);
        let scale: Vec<f32> = moe.router_scale.iter().map(|s| s * root).collect();
        rms(&mut routed_in, &scale, cfg.eps);
        let logits = moe.router.forward(&routed_in)?;
        let experts = moe.experts.len();
        let mut assigned: Vec<Vec<(usize, f32)>> = vec![Vec::new(); experts];
        for t in 0..rows {
            let row = &logits[t * experts..(t + 1) * experts];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let probs: Vec<f32> = row.iter().map(|x| (x - max).exp()).collect();
            let sum: f32 = probs.iter().sum();
            let mut order: Vec<usize> = (0..experts).collect();
            order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));
            let chosen = &order[..moe.used.min(experts)];
            let kept: f32 = chosen.iter().map(|&e| probs[e] / sum).sum();
            for &e in chosen {
                assigned[e].push((t, probs[e] / sum / kept * moe.expert_scale[e]));
            }
        }
        let mut expert_in = residual.to_vec();
        self.norm(&mut expert_in, &moe.pre_norm);
        let mut out = vec![0.0f32; rows * hidden];
        for (expert, tokens) in moe.experts.iter().zip(&assigned) {
            if tokens.is_empty() {
                continue;
            }
            let mut gathered = Vec::with_capacity(tokens.len() * hidden);
            for &(t, _) in tokens {
                gathered.extend_from_slice(&expert_in[t * hidden..(t + 1) * hidden]);
            }
            let result = self.expert_forward(expert, &gathered, workspace)?;
            for (i, &(t, w)) in tokens.iter().enumerate() {
                for (o, r) in out[t * hidden..(t + 1) * hidden].iter_mut().zip(&result[i * hidden..(i + 1) * hidden]) {
                    *o += w * r;
                }
            }
        }
        self.post_norm(&mut out, &moe.post_norm_experts);
        for (d, o) in dense.iter_mut().zip(&out) {
            *d += o;
        }
        Ok(())
    }

    /// Per-layer input path and output scalar after the feed-forward residual add of layer `layer`.
    pub(super) fn gemma_tail(&self, layer: usize, g: &GemmaLayer, x: &mut [f32], inputs: Option<&[f32]>) -> Result<(), String> {
        let cfg = &self.cfg;
        let hidden = cfg.hidden;
        if let (Some(ple), Some(inputs), Some(global)) = (&g.ple, inputs, &self.ple) {
            let rows = x.len() / hidden;
            let total = cfg.layers * global.dim;
            let mut gated = ple.gate.forward(x)?;
            let activation = cfg.activation;
            for r in 0..rows {
                let row = &mut gated[r * global.dim..(r + 1) * global.dim];
                let input = &inputs[r * total + layer * global.dim..r * total + (layer + 1) * global.dim];
                for (v, i) in row.iter_mut().zip(input) {
                    *v = activate(activation, *v) * i;
                }
            }
            let mut back = ple.proj.forward(&gated)?;
            self.post_norm(&mut back, &ple.norm);
            for (v, b) in x.iter_mut().zip(&back) {
                *v += b;
            }
        }
        if let Some(scalar) = g.scalar {
            x.iter_mut().for_each(|v| *v *= scalar);
        }
        Ok(())
    }
}

impl GemmaLayer {
    pub(super) fn has_moe(&self) -> Option<&GemmaMoe> {
        self.moe.as_ref()
    }
}

/// Load the Gemma 3n stream projections.
pub(super) fn load_altup(model: &GgufModel, cfg: &ModelConfig) -> Result<Option<AltupGlobal>, String> {
    let Some(gemma) = cfg.gemma.as_ref().filter(|g| g.altup_inputs > 0) else {
        return Ok(None);
    };
    let mapping = model.mapping();
    let hidden = cfg.hidden;
    let streams = gemma.altup_inputs;
    let stack = |name: &str| -> Result<Vec<Linear>, String> {
        (0..streams - 1)
            .map(|i| {
                let mut l = Linear::load_rows(mapping, name, None, i * hidden, hidden, hidden)?;
                l.enable_fast_cpu();
                Ok(l)
            })
            .collect()
    };
    if gemma.altup_active >= streams {
        return Err("altup.active_idx is out of range".into());
    }
    Ok(Some(AltupGlobal {
        up: stack("altup_proj.weight")?,
        down: stack("altup_unembd_proj.weight")?,
        streams,
        active: gemma.altup_active,
    }))
}

/// Rescale every row of `x` to the root-mean-square of the matching row of `target` (`max(mean square, 1e-5)` in the divisor).
fn match_magnitude(x: &mut [f32], target: &[f32], width: usize) {
    for (row, reference) in x.chunks_exact_mut(width).zip(target.chunks_exact(width)) {
        let wanted = (reference.iter().map(|v| v * v).sum::<f32>() / width as f32).sqrt();
        let own = (row.iter().map(|v| v * v).sum::<f32>() / width as f32).max(1e-5).sqrt();
        row.iter_mut().for_each(|v| *v = *v * wanted / own);
    }
}

impl RustDecoder {
    /// Gemma 3n: the residual stream of every row becomes several parallel streams (`[stream][row][hidden]`).
    pub(super) fn altup_expand(&self, x: &[f32]) -> Result<Vec<f32>, String> {
        let Some(alt) = &self.altup else {
            return Ok(Vec::new());
        };
        let hidden = self.cfg.hidden;
        let mut streams = x.to_vec();
        for projection in &alt.up {
            let mut next = projection.forward(x)?;
            match_magnitude(&mut next, x, hidden);
            streams.extend(next);
        }
        Ok(streams)
    }

    /// Gemma 3n: merge the streams into one residual stream again (mean of the unembedded streams).
    pub(super) fn altup_merge(&self, streams: &[f32]) -> Result<Vec<f32>, String> {
        let Some(alt) = &self.altup else {
            return Err("no AltUp projections".into());
        };
        let hidden = self.cfg.hidden;
        let len = streams.len() / alt.streams;
        let first = &streams[..len];
        let mut sum = first.to_vec();
        for (i, projection) in alt.down.iter().enumerate() {
            let mut next = projection.forward(&streams[(i + 1) * len..(i + 2) * len])?;
            match_magnitude(&mut next, first, hidden);
            for (s, n) in sum.iter_mut().zip(&next) {
                *s += n;
            }
        }
        let count = alt.streams as f32;
        sum.iter_mut().for_each(|v| *v /= count);
        Ok(sum)
    }

    /// Router modalities `tanh(router(norm(x) / hidden))`, `rows x streams`.
    fn altup_modalities(&self, layer: &AltupLayer, x: &[f32]) -> Result<Vec<f32>, String> {
        let hidden = self.cfg.hidden;
        let mut input = x.to_vec();
        self.norm(&mut input, &layer.router_norm);
        input.iter_mut().for_each(|v| *v /= hidden as f32);
        let mut modalities = layer.router.forward(&input)?;
        modalities.iter_mut().for_each(|v| *v = v.tanh());
        Ok(modalities)
    }

    /// `predictions[o] = streams[o] + sum_i coef[o][i] * streams[i]` with per-row coefficients from the active stream.
    pub(super) fn altup_predict(&self, layer: &AltupLayer, streams: &[f32], rows: usize) -> Result<Vec<f32>, String> {
        let alt = self.altup.as_ref().ok_or("no AltUp projections")?;
        let (hidden, n) = (self.cfg.hidden, alt.streams);
        let len = rows * hidden;
        let active = &streams[alt.active * len..(alt.active + 1) * len];
        let modalities = self.altup_modalities(layer, active)?;
        let coefs = layer.predict.forward(&modalities)?;
        let mut out = streams.to_vec();
        for r in 0..rows {
            for o in 0..n {
                let dst = &mut out[o * len + r * hidden..o * len + (r + 1) * hidden];
                for i in 0..n {
                    let c = coefs[r * n * n + o * n + i];
                    let src = &streams[i * len + r * hidden..i * len + (r + 1) * hidden];
                    for (d, s) in dst.iter_mut().zip(src) {
                        *d += c * s;
                    }
                }
            }
        }
        Ok(out)
    }

    /// `corrected[i] = predictions[i] + (activated - predictions[active]) * (1 + coef_i)`.
    pub(super) fn altup_correct(
        &self,
        layer: &AltupLayer,
        predictions: &[f32],
        activated: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>, String> {
        let alt = self.altup.as_ref().ok_or("no AltUp projections")?;
        let (hidden, n) = (self.cfg.hidden, alt.streams);
        let len = rows * hidden;
        let modalities = self.altup_modalities(layer, activated)?;
        let coefs = layer.correct.forward(&modalities)?;
        let mut out = predictions.to_vec();
        let active = &predictions[alt.active * len..(alt.active + 1) * len];
        for r in 0..rows {
            let (new, old) = (&activated[r * hidden..(r + 1) * hidden], &active[r * hidden..(r + 1) * hidden]);
            for s in 0..n {
                let c = coefs[r * n + s] + 1.0;
                let dst = &mut out[s * len + r * hidden..s * len + (r + 1) * hidden];
                for ((d, a), p) in dst.iter_mut().zip(new).zip(old) {
                    *d += (a - p) * c;
                }
            }
        }
        Ok(out)
    }

    /// `x + norm(right(left(x)))` of a LAuReL block.
    pub(super) fn laurel_forward(&self, laurel: &Laurel, normed: &[f32]) -> Result<Vec<f32>, String> {
        let mut back = laurel.right.forward(&laurel.left.forward(normed)?)?;
        self.post_norm(&mut back, &laurel.norm);
        for (b, x) in back.iter_mut().zip(normed) {
            *b += x;
        }
        Ok(back)
    }

    /// Gemma 3n tail of a layer: AltUp correction, per-layer input, and the update of the non-active streams.
    pub(super) fn altup_tail(
        &self,
        layer: usize,
        g: &GemmaLayer,
        activated: &[f32],
        predictions: &[f32],
        streams: &mut Vec<f32>,
        inputs: Option<&[f32]>,
    ) -> Result<(), String> {
        let cfg = &self.cfg;
        let hidden = cfg.hidden;
        let rows = activated.len() / hidden;
        let alt = self.altup.as_ref().ok_or("no AltUp projections")?;
        let layer_alt = g.altup.as_ref().ok_or("layer without AltUp")?;
        let mut corrected = self.altup_correct(layer_alt, predictions, activated, rows)?;
        let len = rows * hidden;
        if let (Some(ple), Some(inputs), Some(global)) = (&g.ple, inputs, &self.ple) {
            let mut first = corrected[alt.active * len..(alt.active + 1) * len].to_vec();
            if cfg.gemma.as_ref().is_some_and(|c| c.altup_correct_scale) {
                for row in first.chunks_exact_mut(hidden) {
                    for (v, s) in row.iter_mut().zip(&layer_alt.correct_scale) {
                        *v *= s;
                    }
                }
            }
            let total = cfg.layers * global.dim;
            let mut gated = ple.gate.forward(&first)?;
            let activation = cfg.activation;
            for r in 0..rows {
                let row = &mut gated[r * global.dim..(r + 1) * global.dim];
                let input = &inputs[r * total + layer * global.dim..r * total + (layer + 1) * global.dim];
                for (v, i) in row.iter_mut().zip(input) {
                    *v = activate(activation, *v) * i;
                }
            }
            let mut back = ple.proj.forward(&gated)?;
            self.post_norm(&mut back, &ple.norm);
            for s in 1..alt.streams {
                for (v, b) in corrected[s * len..(s + 1) * len].iter_mut().zip(&back) {
                    *v += b;
                }
            }
        }
        *streams = corrected;
        Ok(())
    }
}

/// `relu(x - (mean + std * factor))` over each row of `width` values (population standard deviation).
pub(super) fn gaussian_top_k(values: &mut [f32], width: usize, factor: f32) {
    for row in values.chunks_exact_mut(width) {
        let mean = row.iter().sum::<f32>() / width as f32;
        let variance = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / width as f32;
        let cutoff = mean + variance.sqrt() * factor;
        row.iter_mut().for_each(|v| *v = (*v - cutoff).max(0.0));
    }
}

impl GemmaLayer {
    pub(super) fn laurel(&self) -> Option<&Laurel> {
        self.laurel.as_ref()
    }
    pub(super) fn sparsity(&self) -> f32 {
        self.sparsity
    }
}

impl AltupGlobal {
    pub(super) fn active(&self) -> usize {
        self.active
    }
}

impl GemmaLayer {
    pub(super) fn altup(&self) -> Option<&AltupLayer> {
        self.altup.as_ref()
    }
}
