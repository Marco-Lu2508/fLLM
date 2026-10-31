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

type Dev = crate::backend::DeviceBuffer;
type Weight = crate::backend::DeviceWeight;

/// Device copy of one call of a shared block (the host keeps one `ZambaCall` per layer, so the shared weights are uploaded once per call).
pub(super) struct DeviceZamba {
    attn_norm: Dev,
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    qkv_adapters: Option<[(Weight, Weight); 3]>,
    ffn_norm: Dev,
    gate: Weight,
    up: Weight,
    down: Weight,
    ffn_adapter: Option<(Weight, Weight, Weight)>,
    linear: Weight,
}

impl ZambaCall {
    /// Uploads the call. `linear` uploads a projection, `vector` a small FP32 vector.
    pub(super) fn upload(
        &self,
        name: &str,
        linear: &dyn Fn(&str, &Linear) -> Result<Weight, String>,
        vector: &dyn Fn(&[f32]) -> Result<Dev, String>,
    ) -> Result<DeviceZamba, String> {
        let part = |tag: &str, l: &Linear| linear(&format!("{name}.{tag}"), l);
        let pair = |tag: &str, a: &Adapter| -> Result<(Weight, Weight), String> { Ok((part(&format!("{tag}_down"), &a.down)?, part(&format!("{tag}_up"), &a.up)?)) };
        let qkv_adapters = match &self.qkv_adapters {
            Some([q, k, v]) => Some([pair("q_adapter", q)?, pair("k_adapter", k)?, pair("v_adapter", v)?]),
            None => None,
        };
        let ffn_adapter = match &self.ffn_adapter {
            Some((a, g, u)) => Some((part("ffn_adapter_a", a)?, part("ffn_adapter_gate", g)?, part("ffn_adapter_up", u)?)),
            None => None,
        };
        Ok(DeviceZamba {
            attn_norm: vector(&self.attn_norm)?,
            q: part("q", &self.q)?,
            k: part("k", &self.k)?,
            v: part("v", &self.v)?,
            o: part("o", &self.o)?,
            qkv_adapters,
            ffn_norm: vector(&self.ffn_norm)?,
            gate: part("gate", &self.gate)?,
            up: part("up", &self.up)?,
            down: part("down", &self.down)?,
            ffn_adapter,
            linear: part("linear", &self.linear)?,
        })
    }
}

fn project(stream: &mut dyn crate::backend::Stream, scratch: &mut crate::backend::BufferPool, w: &Weight, input: &Dev, rows: usize) -> Result<Dev, String> {
    let mut out = scratch.acquire(rows * w.rows * 4).map_err(|e| e.to_string())?;
    stream.quantized_linear(&mut out, input, w, w.bias.as_ref()).map_err(|e| e.to_string())?;
    Ok(out)
}

impl RustDecoder {
    /// Device counterpart of [`RustDecoder::zamba_call`]: the output of the shared-block call of layer `i` for the residual stream `x`
    /// (`rows x hidden`), to be added to `x` in front of the layer's Mamba-2 mixer. The call writes its keys and values into the cache layer
    /// `kvi` of the device KV cache.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn zamba_device(
        &self,
        stream: &mut dyn crate::backend::Stream,
        scratch: &mut crate::backend::BufferPool,
        dz: &DeviceZamba,
        (i, kvi): (usize, usize),
        layer: &Layer,
        x: &Dev,
        (positions, slots, tables, lengths): (&[usize], &[usize], &[i64], &[i64]),
        dev_kv: &crate::backend::DeviceKvCache,
        dev_weights: &DeviceModelWeights,
    ) -> Result<Dev, String> {
        let err = |e: crate::backend::BackendError| e.to_string();
        let cfg = &self.cfg;
        let (h, rows) = (cfg.hidden, positions.len());
        let wide = 2 * h;
        let (dk, dv) = (cfg.head_dim, cfg.value_dim);
        // [x | embeddings], normed over the doubled width
        let original = self.embed_stash.get(rows * h)?;
        let mut orig_buf = scratch.acquire(rows * h * 4).map_err(err)?;
        stream.upload(&mut orig_buf, cast_f32_to_u8(&original)).map_err(err)?;
        let mut cat = scratch.acquire(rows * wide * 4).map_err(err)?;
        stream.copy_columns(&mut cat, wide, 0, x, rows, h).map_err(err)?;
        stream.copy_columns(&mut cat, wide, h, &orig_buf, rows, h).map_err(err)?;
        scratch.recycle(orig_buf);
        let mut normed = scratch.acquire(rows * wide * 4).map_err(err)?;
        stream.rms_norm(&mut normed, &cat, &dz.attn_norm, cfg.eps).map_err(err)?;
        scratch.recycle(cat);
        let mut q = project(stream, scratch, &dz.q, &normed, rows)?;
        let mut k = project(stream, scratch, &dz.k, &normed, rows)?;
        let mut v = project(stream, scratch, &dz.v, &normed, rows)?;
        if let Some(adapters) = &dz.qkv_adapters {
            for (target, (down, up)) in [&mut q, &mut k, &mut v].into_iter().zip(adapters) {
                let low = project(stream, scratch, down, &normed, rows)?;
                let add = project(stream, scratch, up, &low, rows)?;
                stream.residual_add(target, &add, 1.0).map_err(err)?;
                scratch.recycle(low);
                scratch.recycle(add);
            }
        }
        scratch.recycle(normed);
        if cfg.layer_rotates(i) {
            let (cos, sin) = &dev_weights.rope[0];
            let params = crate::backend::RopeParams {
                heads: cfg.heads,
                kv_heads: cfg.kv_heads,
                dim: dk,
                rope_dim: cfg.rope.dims,
                neox: cfg.rope.neox,
                cos: Arc::clone(cos),
                sin: Arc::clone(sin),
            };
            stream.rope(&mut q, &mut k, positions, &params).map_err(err)?;
        }
        stream.kv_cache_write(dev_kv, kvi, slots, &k, &v).map_err(err)?;
        scratch.recycle(k);
        scratch.recycle(v);
        let mut attn = scratch.acquire(rows * cfg.heads * dv * 4).map_err(err)?;
        let params = crate::backend::AttentionParams {
            num_heads: cfg.heads,
            num_kv_heads: cfg.kv_heads,
            head_dim: dk,
            value_dim: dv,
            stride: self.stride,
            block_size: self.block_size,
            scale: cfg.attention_scale.unwrap_or(1.0 / (dk as f32).sqrt()),
            softcap: cfg.attention_softcap,
            window: layer.window,
            chunk: layer.chunk,
            sinks: None,
            alibi_slopes: dev_weights.alibi.clone(),
        };
        let read_layer = crate::kv_geometry::KvPoolLayout::owner(cfg, i) - self.kv_base();
        stream.attention_paged(&mut attn, &q, dev_kv, tables, lengths, read_layer, &params).map_err(err)?;
        scratch.recycle(q);
        let y = project(stream, scratch, &dz.o, &attn, rows)?;
        scratch.recycle(attn);
        let mut y_norm = scratch.acquire(rows * h * 4).map_err(err)?;
        stream.rms_norm(&mut y_norm, &y, &dz.ffn_norm, cfg.eps).map_err(err)?;
        scratch.recycle(y);
        let mut gate = project(stream, scratch, &dz.gate, &y_norm, rows)?;
        let mut up = project(stream, scratch, &dz.up, &y_norm, rows)?;
        if let Some((a, gate_b, up_b)) = &dz.ffn_adapter {
            let low = project(stream, scratch, a, &y_norm, rows)?;
            for (target, b) in [(&mut gate, gate_b), (&mut up, up_b)] {
                let add = project(stream, scratch, b, &low, rows)?;
                stream.residual_add(target, &add, 1.0).map_err(err)?;
                scratch.recycle(add);
            }
            scratch.recycle(low);
        }
        scratch.recycle(y_norm);
        let mut act = scratch.acquire(rows * cfg.ff * 4).map_err(err)?;
        match cfg.activation {
            Activation::GeluTanh => stream.geglu(&mut act, &gate, &up).map_err(err)?,
            Activation::Silu => stream.swiglu(&mut act, &gate, &up).map_err(err)?,
            other => stream.activation_glu(&mut act, &gate, &up, act_code(other)).map_err(err)?,
        }
        scratch.recycle(gate);
        scratch.recycle(up);
        let down = project(stream, scratch, &dz.down, &act, rows)?;
        scratch.recycle(act);
        let out = project(stream, scratch, &dz.linear, &down, rows)?;
        scratch.recycle(down);
        Ok(out)
    }
}
