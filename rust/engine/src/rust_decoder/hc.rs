//! Models built from the ordinary layers of the decoder (`Layer`: latent attention with the lightning indexer, Kimi delta attention,
//! dense or MoE feed-forwards) wrapped in hyper-connections: Hy4 (independent connections, `hc_magnitude` set) and GLM-5-Next
//! (manifold-constrained connections with a Sinkhorn-projected mixing matrix, an unweighted mean of the streams at the end).
//! Every block reads a weighted sum of the `hc_mult` residual streams and adds its output, scaled per stream, to the (mixed) streams
//! (`deepseek4::Hc`).
use super::deepseek4::{combine_streams, Hc};
use super::*;

/// The mixing weights of a block: `(post, comb, collapsed input)`; independent connections have no `comb`.
type BlockWeights = (Vec<f32>, Option<Vec<f32>>, Vec<f32>);

pub(super) struct HcModel {
    attn: Vec<Hc>,
    ffn: Vec<Hc>,
    /// `None`: the final collapse is the mean of the streams.
    head: Option<Hc>,
}

impl HcModel {
    pub(super) fn load(mapping: &MappedGguf, cfg: &ModelConfig) -> Result<Self, String> {
        let h = cfg.hc.as_ref().ok_or("model without hyper-connections")?;
        let (hc, hidden) = (h.hc_mult, cfg.hidden);
        let (outputs, scales) = if h.magnitude.is_some() { (2 * hc, 2) } else { ((2 + hc) * hc, 3) };
        let mut attn = Vec::with_capacity(cfg.layers);
        let mut ffn = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            attn.push(Hc::load(mapping, &format!("blk.{i}.hc_attn"), hc, hidden, outputs, scales)?);
            ffn.push(Hc::load(mapping, &format!("blk.{i}.hc_ffn"), hc, hidden, outputs, scales)?);
        }
        let head = if h.head_mean { None } else { Some(Hc::load(mapping, "output_hc", hc, hidden, hc, 1)?) };
        Ok(Self { attn, ffn, head })
    }
}

impl RustDecoder {
    /// All layers of such a model for a batch: `x` holds the embeddings (rows x hidden); the result is the collapsed stream.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn hc_layers(&self, x: &[f32], positions: &[usize], tables: &[i64], lengths: &[i64], key: &mut [u16], value: &mut [u16]) -> Result<Vec<f32>, String> {
        let (cfg, model) = (&self.cfg, self.hc.as_deref().ok_or("hyper-connection weights missing")?);
        let h = cfg.hc.as_ref().ok_or("hyper-connection configuration missing")?;
        let (rows, hidden, hc) = (positions.len(), cfg.hidden, h.hc_mult);
        let mut streams = Vec::with_capacity(rows * hc * hidden);
        for r in 0..rows {
            for _ in 0..hc {
                streams.extend_from_slice(&x[r * hidden..(r + 1) * hidden]);
            }
        }
        let mut workspace = compute::ProjectionGroupWorkspace::default();
        // the mixing weights of a block: (post, comb, collapsed input)
        let weights = |c: &Hc, streams: &[f32]| -> Result<BlockWeights, String> {
            match h.magnitude {
                Some(m) => c.compute_independent(streams, rows, hc, hidden, cfg.eps, h.eps, m).map(|(post, collapsed)| (post, None, collapsed)),
                None => c.compute(streams, rows, hc, hidden, cfg.eps, h.eps, h.iters).map(|(post, comb, collapsed)| (post, Some(comb), collapsed)),
            }
        };
        for (i, layer) in self.layers.iter().enumerate() {
            let (post, comb, mut normed) = weights(&model.attn[i], &streams)?;
            if let Some(norm) = &layer.attn_norm {
                self.norm(&mut normed, norm);
            }
            let attn = match &layer.attn {
                Attn::Kda(mixer) => self.kda_forward(mixer, &layer.out, i, &normed, tables, lengths)?,
                _ => self.hc_attention(i, layer, &normed, positions, tables, lengths, key, value)?,
            };
            dump_layer("branch", i, &attn);
            combine_streams(&mut streams, &post, comb.as_deref(), &attn, rows, hc, hidden);
            let (post, comb, mut normed) = weights(&model.ffn[i], &streams)?;
            if let Some(norm) = &layer.ffn_norm {
                self.norm(&mut normed, norm);
            }
            let mlp = self.ffn_forward(layer, &normed, &mut workspace)?;
            combine_streams(&mut streams, &post, comb.as_deref(), &mlp, rows, hc, hidden);
            dump_layer("ffn", i, &streams);
        }
        match &model.head {
            Some(head) => head.collapse_head(&streams, rows, hc, hidden, cfg.eps, h.eps),
            None => {
                let mut out = vec![0.0f32; rows * hidden];
                for r in 0..rows {
                    for n in 0..hc {
                        let stream = &streams[(r * hc + n) * hidden..(r * hc + n + 1) * hidden];
                        out[r * hidden..(r + 1) * hidden].iter_mut().zip(stream).for_each(|(o, s)| *o += s / hc as f32);
                    }
                }
                Ok(out)
            }
        }
    }

    /// Latent attention with the indexer selection and the optional sigmoid output gate; returns the projected output (rows x hidden).
    #[allow(clippy::too_many_arguments)]
    fn hc_attention(&self, i: usize, layer: &Layer, normed: &[f32], positions: &[usize], tables: &[i64], lengths: &[i64], key: &mut [u16], value: &mut [u16]) -> Result<Vec<f32>, String> {
        let cfg = &self.cfg;
        let (Attn::Mla(m), Some(mc)) = (&layer.attn, cfg.mla.as_ref()) else {
            return Err("a hyper-connection model layer that is neither delta nor latent attention".into());
        };
        let table = &self.rope[layer.rope_table];
        let sparse = cfg.indexer.is_some() && !self.dsa_dense && (layer.indexer.is_some() || layer.dsa_shared);
        let (qkv, q_low) = self.mla_qkv_low(m, normed, positions, table, sparse && layer.indexer.is_some())?;
        let [q, k, v] = qkv;
        let (heads, dk, dv) = (cfg.heads, mc.nope + mc.rope, mc.value);
        let (kv_k, kv_v) = (heads * dk, heads * dv);
        let (fmt_k, fmt_v) = self.kv_formats();
        let (key_region, value_region) = (self.key_region(i), self.value_region(i));
        for (r, &position) in positions.iter().enumerate() {
            let block = tables[r * self.stride + position / self.block_size] as usize;
            let slot = block * self.block_size + position % self.block_size;
            self.write_cache(key, key_region, slot, &k[r * kv_k..(r + 1) * kv_k], fmt_k);
            self.write_cache(value, value_region, slot, &v[r * kv_v..(r + 1) * kv_v], fmt_v);
        }
        let selection = if sparse { Some(self.dsa_select(i, layer, normed, q_low.as_deref(), positions, table, tables, lengths)?) } else { None };
        let mut attn = self.attention_selected(i, &q, key, value, tables, lengths, selection.as_deref());
        if let Some(gate) = &layer.attn_gate {
            let g = gate.forward(normed)?;
            attn.iter_mut().zip(&g).for_each(|(a, g)| *a *= 1.0 / (1.0 + (-g).exp()));
        }
        layer.out.forward(&attn)
    }
}
