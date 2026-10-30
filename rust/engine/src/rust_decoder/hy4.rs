//! Hy4 on the host: latent attention with the lightning indexer, a sigmoid output gate and sinks (the layers of `Layer`), wrapped in
//! independent hyper-connections: every block reads a weighted sum of the `hc_mult` residual streams and adds its output, scaled per
//! stream, to each of them (no mixing matrix, see `deepseek4::Hc::compute_independent`).
use super::deepseek4::{combine_streams, Hc};
use super::*;

pub(super) struct Hy4Model {
    attn: Vec<Hc>,
    ffn: Vec<Hc>,
    head: Hc,
}

impl Hy4Model {
    pub(super) fn load(mapping: &MappedGguf, cfg: &ModelConfig) -> Result<Self, String> {
        let h = cfg.hy4.as_ref().ok_or("not a Hy4 model")?;
        let (hc, hidden) = (h.hc_mult, cfg.hidden);
        let mut attn = Vec::with_capacity(cfg.layers);
        let mut ffn = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            attn.push(Hc::load(mapping, &format!("blk.{i}.hc_attn"), hc, hidden, 2 * hc, 2)?);
            ffn.push(Hc::load(mapping, &format!("blk.{i}.hc_ffn"), hc, hidden, 2 * hc, 2)?);
        }
        Ok(Self { attn, ffn, head: Hc::load(mapping, "output_hc", hc, hidden, hc, 1)? })
    }
}

impl RustDecoder {
    /// All layers of a Hy4 model for a batch: `x` holds the embeddings (rows x hidden); the result is the collapsed stream.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn hy4_layers(&self, x: &[f32], positions: &[usize], tables: &[i64], lengths: &[i64], key: &mut [u16], value: &mut [u16]) -> Result<Vec<f32>, String> {
        let (cfg, model) = (&self.cfg, self.hy4.as_deref().ok_or("Hy4 weights missing")?);
        let h = cfg.hy4.as_ref().ok_or("Hy4 configuration missing")?;
        let (rows, hidden, hc) = (positions.len(), cfg.hidden, h.hc_mult);
        let mut streams = Vec::with_capacity(rows * hc * hidden);
        for r in 0..rows {
            for _ in 0..hc {
                streams.extend_from_slice(&x[r * hidden..(r + 1) * hidden]);
            }
        }
        let mut workspace = compute::ProjectionGroupWorkspace::default();
        for (i, layer) in self.layers.iter().enumerate() {
            let (post, mut normed) = model.attn[i].compute_independent(&streams, rows, hc, hidden, cfg.eps, h.eps, h.magnitude)?;
            if let Some(norm) = &layer.attn_norm {
                self.norm(&mut normed, norm);
            }
            let attn = self.hy4_attention(i, layer, &normed, positions, tables, lengths, key, value)?;
            combine_streams(&mut streams, &post, None, &attn, rows, hc, hidden);
            let (post, mut normed) = model.ffn[i].compute_independent(&streams, rows, hc, hidden, cfg.eps, h.eps, h.magnitude)?;
            if let Some(norm) = &layer.ffn_norm {
                self.norm(&mut normed, norm);
            }
            let mlp = self.ffn_forward(layer, &normed, &mut workspace)?;
            combine_streams(&mut streams, &post, None, &mlp, rows, hc, hidden);
        }
        model.head.collapse_head(&streams, rows, hc, hidden, cfg.eps, h.eps)
    }

    /// Latent attention with the indexer selection and the sigmoid output gate; returns the projected output (rows x hidden).
    #[allow(clippy::too_many_arguments)]
    fn hy4_attention(&self, i: usize, layer: &Layer, normed: &[f32], positions: &[usize], tables: &[i64], lengths: &[i64], key: &mut [u16], value: &mut [u16]) -> Result<Vec<f32>, String> {
        let cfg = &self.cfg;
        let (Attn::Mla(m), Some(mc)) = (&layer.attn, cfg.mla.as_ref()) else {
            return Err("Hy4 layer without latent attention".into());
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
