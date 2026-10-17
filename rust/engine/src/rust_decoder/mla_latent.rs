//! One multi-head latent attention layer on the device with the latent cache (`FLLM_MLA=latent`, see `crate::mla_latent`).
//!
//! The projections are the same GEMMs as in the expanded path; what changes is the middle: the cache row `[latent | rotary key]` replaces the
//! rebuilt per-head keys and values, the queries are absorbed into the latent space, and the output is expanded after the attention.
use super::*;
use crate::backend::{AttentionParams, BufferPool, DeviceBuffer, DeviceKvCache, MlaShape, RopeParams, Stream};
use crate::config::MlaConfig;

/// Temporaries of one chunk of rows (absorbed queries and FP16 attention output) stay below this many bytes.
const CHUNK_BUDGET_BYTES: usize = 48 << 20;

impl RustDecoder {
    /// The attention part of an MLA layer with the latent cache: returns the `rows x (heads * value)` FP32 input of the output projection.
    /// `normed` is the normalized layer input; the row of every token is written to the cache slot `slots[r]` on the way.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn device_mla_latent(
        &self,
        stream: &mut dyn Stream,
        scratch: &mut BufferPool,
        dm: &DeviceMla,
        mc: &MlaConfig,
        layer: &Layer,
        layer_idx: usize,
        dev_weights: &DeviceModelWeights,
        dev_kv: &DeviceKvCache,
        normed: &DeviceBuffer,
        positions: &[usize],
        slots: &[usize],
        tables: &[i64],
        lengths: &[i64],
    ) -> Result<DeviceBuffer, String> {
        let err = |e: crate::backend::BackendError| e.to_string();
        let cfg = &self.cfg;
        let rows = positions.len();
        let heads = cfg.heads;
        let shape = MlaShape { heads, nope: mc.nope, rope: mc.rope, value: mc.value, kv_lora: mc.kv_lora };
        let dk = mc.nope + mc.rope;
        let width = mc.kv_lora + mc.rope;
        let w_kvb = dm.kv_b_dense.as_ref().ok_or("latent MLA: the dense kv_b matrix is not on the device")?;
        let (cos, sin) = &dev_weights.rope[layer.rope_table];
        let rope_params = RopeParams {
            heads,
            kv_heads: 1,
            dim: mc.rope,
            rope_dim: mc.rope,
            neox: cfg.rope.neox,
            cos: Arc::clone(cos),
            sin: Arc::clone(sin),
        };

        // query projection (low-rank when the model has q_lora)
        let mut q_full = scratch.acquire(rows * heads * dk * 4).map_err(err)?;
        if let Some((down, norm)) = &dm.q_a {
            let mut low = scratch.acquire(rows * down.rows * 4).map_err(err)?;
            stream.quantized_linear(&mut low, normed, down, down.bias.as_ref()).map_err(err)?;
            device_rms_inplace(stream, scratch, &mut low, Some(norm), cfg.eps)?;
            stream.quantized_linear(&mut q_full, &low, &dm.q, dm.q.bias.as_ref()).map_err(err)?;
            scratch.recycle(low);
        } else {
            stream.quantized_linear(&mut q_full, normed, &dm.q, dm.q.bias.as_ref()).map_err(err)?;
        }

        // cache row of every token: the normalized latent and the rotated shared rope key
        let mut kv_a = scratch.acquire(rows * width * 4).map_err(err)?;
        stream.quantized_linear(&mut kv_a, normed, &dm.kv_a, dm.kv_a.bias.as_ref()).map_err(err)?;
        let mut latent = scratch.acquire(rows * mc.kv_lora * 4).map_err(err)?;
        stream.mla_latent(&mut latent, &kv_a, rows, mc.kv_lora, width).map_err(err)?;
        device_rms_inplace(stream, scratch, &mut latent, Some(&dm.kv_a_norm), cfg.eps)?;
        let mut k_row = scratch.acquire(rows * width * 4).map_err(err)?;
        stream.mla_latent_kv(&mut k_row, &kv_a, &latent, positions, &shape, &rope_params).map_err(err)?;
        // the value pool is a placeholder (the values are the first kv_lora values of the key row)
        let v_dummy = scratch.acquire(rows * dev_kv.kv_heads * dev_kv.value_dim * 4).map_err(err)?;
        stream.kv_cache_write(dev_kv, layer_idx, slots, &k_row, &v_dummy).map_err(err)?;
        for buf in [kv_a, latent, k_row, v_dummy] {
            scratch.recycle(buf);
        }

        // attention in chunks of rows: absorbed queries, attention over the cache, expansion of the output
        let params = AttentionParams {
            num_heads: heads,
            num_kv_heads: 1,
            head_dim: width,
            value_dim: crate::mla_latent::LATENT_VALUE_DIM,
            stride: self.stride,
            block_size: self.block_size,
            scale: cfg.attention_scale.unwrap_or(1.0 / (dk as f32).sqrt()),
            softcap: None,
            window: None,
            chunk: None,
            sinks: None,
            alibi_slopes: None,
        };
        let reference = std::env::var("FLLM_MLA_KERNEL").is_ok_and(|v| v == "ref");
        let attn = scratch.acquire(rows * heads * mc.value * 4).map_err(err)?;
        let chunk = crate::mla_latent::rows_per_chunk(heads, mc, CHUNK_BUDGET_BYTES);
        let q_lat = scratch.acquire(chunk.min(rows) * heads * width * 2).map_err(err)?;
        let o_lat = scratch.acquire(chunk.min(rows) * heads * mc.kv_lora * 2).map_err(err)?;
        let mut r0 = 0;
        while r0 < rows {
            let n = chunk.min(rows - r0);
            let q_view = q_full.slice(r0 * heads * dk * 4, n * heads * dk * 4).map_err(err)?;
            let mut q_lat_view = q_lat.slice(0, n * heads * width * 2).map_err(err)?;
            stream.mla_latent_query(&mut q_lat_view, &q_view, w_kvb, &positions[r0..r0 + n], &shape, &rope_params).map_err(err)?;
            let mut o_view = o_lat.slice(0, n * heads * mc.kv_lora * 2).map_err(err)?;
            stream
                .mla_latent_attention(
                    &mut o_view,
                    &q_lat_view,
                    dev_kv,
                    &tables[r0 * self.stride..(r0 + n) * self.stride],
                    &lengths[r0..r0 + n],
                    layer_idx,
                    &params,
                    &shape,
                    reference,
                )
                .map_err(err)?;
            let mut out_view = attn.slice(r0 * heads * mc.value * 4, n * heads * mc.value * 4).map_err(err)?;
            stream.mla_latent_output(&mut out_view, &o_view, w_kvb, n, &shape).map_err(err)?;
            r0 += n;
        }
        for buf in [q_full, q_lat, o_lat] {
            scratch.recycle(buf);
        }
        Ok(attn)
    }
}

/// Uploads FP16 bits (the dense kv_b matrix of a layer) into a new device buffer.
pub(super) fn upload_f16(backend: &dyn crate::backend::Backend, bits: &[u16]) -> Result<DeviceBuffer, String> {
    let err = |e: crate::backend::BackendError| e.to_string();
    let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let mut buffer = backend.allocate_buffer(bytes.len().max(2)).map_err(err)?;
    let mut stream = backend.create_stream().map_err(err)?;
    stream.upload(&mut buffer, &bytes).map_err(err)?;
    stream.synchronize().map_err(err)?;
    Ok(buffer)
}

impl RustDecoder {
    /// `init_backend` of a model whose cache is the latent one: every MLA layer must run on a stream that has the kernels for its dimensions and
    /// cache format, on the device (the host reference code reads the expanded cache), without window, chunks, sinks or soft-capping.
    pub(super) fn check_latent_mla(&self, backend: &dyn crate::backend::Backend, layer_reasons: &[Option<&'static str>]) -> Result<(), String> {
        let hint = "set FLLM_MLA=expanded (or unset it) to use the expanded MLA cache";
        let m = self.cfg.mla.as_ref().ok_or("the latent MLA cache needs a multi-head latent attention model")?;
        let shape = MlaShape { heads: self.cfg.heads, nope: m.nope, rope: m.rope, value: m.value, kv_lora: m.kv_lora };
        let (key_format, _) = crate::kv_format::unpack_kind(self.kv_kind).ok_or("unknown KV cache format")?;
        let supported = backend.create_stream().map(|s| s.supports_mla_latent(&shape, key_format)).unwrap_or(false);
        if !supported {
            return Err(format!(
                "FLLM_MLA=latent: the {} backend has no latent MLA kernels for kv_lora {} / rope {} / nope {} / value {} with a {} cache; {hint}",
                backend.name(),
                m.kv_lora,
                m.rope,
                m.nope,
                m.value,
                key_format.name()
            ));
        }
        for (layer, reason) in self.layers.iter().zip(layer_reasons) {
            if !matches!(layer.attn, Attn::Mla(_)) {
                continue;
            }
            if let Some(reason) = reason {
                return Err(format!("FLLM_MLA=latent: an attention layer would run on the host ({reason}); {hint}"));
            }
            if layer.window.is_some() || layer.chunk.is_some() || layer.sinks.is_some() || self.cfg.attention_softcap.is_some() {
                return Err(format!("FLLM_MLA=latent does not cover sliding windows, chunks, sinks or soft-capping; {hint}"));
            }
        }
        Ok(())
    }
}
