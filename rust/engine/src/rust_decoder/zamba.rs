//! Zamba2: the shared transformer block that starts some of the Mamba-2 layers.
//!
//! A call of the block (reference: transformers `Zamba2HybridLayer`) works on the concatenation `[x, embeddings]` of the residual
//! stream and the embedding output: RMS norm over the doubled width, attention (queries, keys and values carry per-call low-rank
//! adapters, softmax scale `1 / sqrt(head_dim / 2)`, optional rotary embeddings), a norm, a GELU-gated MLP whose gate and up
//! projections carry a per-call adapter, and a per-call linear layer back to the hidden width. The result is added to the
//! residual stream in front of the layer's Mamba-2 mixer; the block itself adds no residual. Every call writes its keys and
//! values into the KV layer of the Mamba layer it belongs to.
use super::*;

/// Low-rank correction `B (A x)` of a shared projection.
pub(super) struct Adapter {
    down: Linear,
    up: Linear,
}

impl Adapter {
    fn apply(&self, input: &[f32], out: &mut [f32]) -> Result<(), String> {
        let low = self.down.forward(input)?;
        for (o, a) in out.iter_mut().zip(self.up.forward(&low)?) {
            *o += a;
        }
        Ok(())
    }
}

/// One call of a shared block: the shared weights (loaded from the block's tensors), the call's adapters and linear layer.
pub(super) struct ZambaCall {
    attn_norm: Vec<f32>,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    qkv_adapters: Option<[Adapter; 3]>,
    ffn_norm: Vec<f32>,
    gate: Linear,
    up: Linear,
    down: Linear,
    ffn_adapter: Option<(Linear, Linear, Linear)>,
    /// Maps the block output to the hidden width before it joins the Mamba input.
    linear: Linear,
}

impl ZambaCall {
    pub(super) fn load(
        cfg: &ModelConfig,
        mapping: &MappedGguf,
        layer: usize,
        linear: &dyn Fn(&str, usize, usize) -> Result<Linear, String>,
    ) -> Result<Option<Box<Self>>, String> {
        let Some(z) = &cfg.hybrid.zamba else { return Ok(None) };
        let Some(block) = z.block_of(layer) else { return Ok(None) };
        let (h, wide) = (cfg.hidden, 2 * cfg.hidden);
        let (qr, kr) = (cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim);
        let shared = format!("shared.{block}");
        let own = format!("blk.{layer}");
        let adapter = |a: &str, b: &str, inner: usize, outer: usize| -> Result<Adapter, String> {
            Ok(Adapter { down: linear(&format!("{own}.{a}"), z.rank, inner)?, up: linear(&format!("{own}.{b}"), outer, z.rank)? })
        };
        let qkv_adapters = if z.attention_adapters {
            Some([
                adapter("adapter_q_a", "adapter_q_b", wide, qr)?,
                adapter("adapter_k_a", "adapter_k_b", wide, kr)?,
                adapter("adapter_v_a", "adapter_v_b", wide, kr)?,
            ])
        } else {
            None
        };
        // the gate and up projections are the two halves of one tensor, and so are the halves of their adapter
        let ff = cfg.ff;
        let gate_up = format!("{shared}.ffn_gate_up.weight");
        let gate_up_bias = format!("{shared}.ffn_gate_up.bias");
        let half = |name: &str, first: usize| {
            let bias_arg = if has_tensor(mapping, &gate_up_bias) {
                Some(gate_up_bias.as_str())
            } else {
                None
            };
            Linear::load_rows(mapping, name, bias_arg, first, ff, h)
        };
        let ffn_adapter = if has_tensor(mapping, &format!("{own}.adapter_ffn_a.weight")) {
            let up_b = format!("{own}.adapter_ffn_b.weight");
            Some((
                linear(&format!("{own}.adapter_ffn_a"), z.rank, h)?,
                Linear::load_rows(mapping, &up_b, None, 0, ff, z.rank)?,
                Linear::load_rows(mapping, &up_b, None, ff, ff, z.rank)?,
            ))
        } else {
            None
        };
        Ok(Some(Box::new(Self {
            attn_norm: vector(mapping, &format!("{shared}.attn_norm.weight"), wide)?,
            q: linear(&format!("{shared}.attn_q"), qr, wide)?,
            k: linear(&format!("{shared}.attn_k"), kr, wide)?,
            v: linear(&format!("{shared}.attn_v"), kr, wide)?,
            o: linear(&format!("{shared}.attn_output"), h, qr)?,
            qkv_adapters,
            ffn_norm: vector(mapping, &format!("{shared}.ffn_norm.weight"), h)?,
            gate: half(&gate_up, 0)?,
            up: half(&gate_up, ff)?,
            down: linear(&format!("{shared}.ffn_down"), h, ff)?,
            ffn_adapter,
            linear: linear(&format!("{own}.zamba_linear"), h, h)?,
        })))
    }
}

impl RustDecoder {
    /// Output of the shared-block call of layer `i` for the residual stream `x` (rows x hidden), to be added to `x` in front of the
    /// layer's Mamba mixer.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn zamba_call(
        &self,
        call: &ZambaCall,
        i: usize,
        x: &[f32],
        positions: &[usize],
        tables: &[i64],
        lengths: &[i64],
        key: &mut [u16],
        value: &mut [u16],
    ) -> Result<Vec<f32>, String> {
        let cfg = &self.cfg;
        let (h, rows) = (cfg.hidden, positions.len());
        let (dk, dv) = (cfg.head_dim, cfg.value_dim);
        let original = self.embed_stash.get(rows * h)?;
        let mut wide = Vec::with_capacity(rows * 2 * h);
        for (row, orig) in x.chunks_exact(h).zip(original.chunks_exact(h)) {
            wide.extend_from_slice(row);
            wide.extend_from_slice(orig);
        }
        rms(&mut wide, &call.attn_norm, cfg.eps);
        let mut q = call.q.forward(&wide)?;
        let mut k = call.k.forward(&wide)?;
        let mut v = call.v.forward(&wide)?;
        if let Some([qa, ka, va]) = &call.qkv_adapters {
            qa.apply(&wide, &mut q)?;
            ka.apply(&wide, &mut k)?;
            va.apply(&wide, &mut v)?;
        }
        if cfg.layer_rotates(i) {
            let table = &self.rope[0];
            apply_rope(&mut q, cfg.heads, dk, cfg.rope.dims, cfg.rope.neox, positions, table);
            apply_rope(&mut k, cfg.kv_heads, dk, cfg.rope.dims, cfg.rope.neox, positions, table);
        }
        let (fmt_k, fmt_v) = self.kv_formats();
        let (kv_k, kv_v) = (cfg.kv_heads * dk, cfg.kv_heads * dv);
        for (r, &position) in positions.iter().enumerate() {
            let block = tables[r * self.stride + position / self.block_size] as usize;
            let slot = block * self.block_size + position % self.block_size;
            self.write_cache(key, self.key_region(i), slot, &k[r * kv_k..(r + 1) * kv_k], fmt_k);
            self.write_cache(value, self.value_region(i), slot, &v[r * kv_v..(r + 1) * kv_v], fmt_v);
        }
        let attn = self.attention_reference(i, &q, key, value, tables, lengths);
        let mut y = call.o.forward(&attn)?;
        rms(&mut y, &call.ffn_norm, cfg.eps);
        let mut gate = call.gate.forward(&y)?;
        let mut up = call.up.forward(&y)?;
        if let Some((a, gate_b, up_b)) = &call.ffn_adapter {
            let low = a.forward(&y)?;
            for (g, d) in gate.iter_mut().zip(gate_b.forward(&low)?) {
                *g += d;
            }
            for (u, d) in up.iter_mut().zip(up_b.forward(&low)?) {
                *u += d;
            }
        }
        for (g, u) in gate.iter_mut().zip(&up) {
            *g = activate(cfg.activation, *g) * u;
        }
        call.linear.forward(&call.down.forward(&gate)?)
    }
}
