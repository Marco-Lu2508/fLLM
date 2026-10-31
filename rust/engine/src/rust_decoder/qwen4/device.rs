//! Qwen3.8-Flash-Next on CUDA, HIP and Vulkan devices. The host code of the parent module is the reference.
//!
//! The layers themselves (gated delta layers, gated attention, MoE) are the ordinary device layers of the decoder. What this module adds:
//! * the `hc` residual streams in a buffer of their own. A block reads them through a gate (`q4_enter`: grouped RMS norm, low-rank gate, gated
//!   mean into the block input, injection weights) and writes its output back with `hc_combine` (`HcBlock`), like the hyper-connections of Hy4;
//! * the hashed n-gram embedding of some delta layers (`q4_ple_device`): the hash of the token ids runs on the host (it needs the token history of the
//!   sequence), the table lookup, projections, gate, dilated convolution and the add into the streams run on the device. The convolution history lives
//!   in the device state buffer of the sequence, behind the DeltaNet states;
//! * the token indexer of the attention layers (`q4_select_device`): the pooled block keys are computed by `q4_pool_keys` and handed to `dsa_select`
//!   with pools of `ratio` tokens that keep the unfinished last block; `dsa_attention` then attends to the chosen tokens.
use super::*;
use crate::backend::{Backend, BufferPool, DsaParams, DeltaRun, DeviceBuffer, DeviceWeight, QuantType, RopeParams, Stream};
use std::sync::Arc;

/// Device copy of a gate (`blk.N.hc_{attn,ffn}_*`, `output_hc_*`).
pub(in crate::rust_decoder) struct DeviceQ4Gate {
    norm: DeviceBuffer,
    down: DeviceWeight,
    up: DeviceWeight,
    inject: Option<DeviceWeight>,
}

/// Device copy of the hashed n-gram embedding of a delta layer.
pub(in crate::rust_decoder) struct DeviceQ4Ple {
    /// The n-gram table; `None`: it stays on the host and the rows of a step are gathered there.
    table: Option<DeviceWeight>,
    key: DeviceWeight,
    value: DeviceWeight,
    norm_key: DeviceBuffer,
    norm_query: DeviceBuffer,
    norm_conv: DeviceBuffer,
    conv: DeviceBuffer,
    /// Index among the layers with an n-gram embedding (selects the slice of a sequence's state buffer).
    ordinal: usize,
}

/// Device copy of the token indexer of an attention layer: the query and key projections, the norms and the per-slot key store.
pub(in crate::rust_decoder) struct DeviceQ4Index {
    q: DeviceWeight,
    k: DeviceWeight,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
    store: DeviceBuffer,
}

pub(in crate::rust_decoder) struct DeviceQ4Layer {
    pub(in crate::rust_decoder) attn: DeviceQ4Gate,
    pub(in crate::rust_decoder) ffn: DeviceQ4Gate,
    pub(in crate::rust_decoder) ple: Option<DeviceQ4Ple>,
    pub(in crate::rust_decoder) index: Option<DeviceQ4Index>,
}

pub(in crate::rust_decoder) struct DeviceQwen4 {
    pub(in crate::rust_decoder) layers: Vec<DeviceQ4Layer>,
    head: DeviceQ4Gate,
}

/// What the indexer layers of one pass share: the inputs of `dsa_select` that depend on the rows only.
pub(in crate::rust_decoder) struct Q4Step {
    /// `rows x heads` ones: the indexer heads are summed without weights.
    ones: DeviceBuffer,
    /// `rows x dim` gate scores of the pooling: 0 for the last token of a block, a large negative value for the others, so that the softmax pooling
    /// of `dsa_select` hands the key of the last token (which `q4_pool_keys` made the pooled key of its block) through unchanged.
    gate: DeviceBuffer,
    /// `ratio x dim` zeros: no position bias.
    ape: DeviceBuffer,
    /// Input of `q4_pool_keys`.
    meta: Vec<i32>,
}

impl Q4Step {
    pub(in crate::rust_decoder) fn recycle(self, scratch: &mut BufferPool) {
        for buf in [self.ones, self.gate, self.ape] {
            scratch.recycle(buf);
        }
    }
}

/// Gate score of the tokens that do not close a block: `exp` of it is exactly 0 next to the 0 of the closing token.
const NOT_LAST: f32 = -1.0e4;

impl Qwen4Model {
    /// Uploads the gates, n-gram tables and indexer weights. `upload_linear` / `upload_vec` are the decoder's closures for projections and vectors.
    pub(in crate::rust_decoder) fn upload_device(
        &self,
        rd: &RustDecoder,
        backend: &dyn Backend,
        upload_linear: &dyn Fn(&str, &Linear) -> Result<DeviceWeight, String>,
        upload_vec: &dyn Fn(&[f32]) -> Result<DeviceBuffer, String>,
    ) -> Result<DeviceQwen4, String> {
        let gate = |g: &Gate, name: &str| -> Result<DeviceQ4Gate, String> {
            Ok(DeviceQ4Gate {
                norm: upload_vec(&g.norm)?,
                down: upload_linear(&format!("{name}_down"), &g.down)?,
                up: upload_linear(&format!("{name}_up"), &g.up)?,
                inject: g.inject.as_ref().map(|l| upload_linear(&format!("{name}_inject"), l)).transpose()?,
            })
        };
        let dp = rd.q4_dsa_params();
        let table_on_device = rd.q4_table_on_device(self);
        if !table_on_device && self.layers.iter().any(|l| l.ple.is_some()) {
            eprintln!("[fllm-rs] the n-gram tables stay on the host (over {} MiB, or FLLM_QWEN4_TABLE=host): their rows are gathered there for every step", TABLE_BUDGET >> 20);
        }
        let mut layers = Vec::with_capacity(self.layers.len());
        let mut ordinal = 0;
        for (i, ql) in self.layers.iter().enumerate() {
            let ple = match &ql.ple {
                Some(ple) => {
                    let name = format!("blk.{i}.ple");
                    let table = if table_on_device { Some(upload_table(backend, &format!("blk.{i}.ple_embd"), &ple.embedding)?) } else { None };
                    let dev = DeviceQ4Ple {
                        table,
                        key: upload_linear(&format!("{name}_key"), &ple.key)?,
                        value: upload_linear(&format!("{name}_value"), &ple.value)?,
                        norm_key: upload_vec(&ple.norm_key)?,
                        norm_query: upload_vec(&ple.norm_query)?,
                        norm_conv: upload_vec(&ple.norm_conv)?,
                        conv: upload_vec(&ple.conv)?,
                        ordinal,
                    };
                    ordinal += 1;
                    Some(dev)
                }
                None => None,
            };
            let index = match (&ql.indexer, &dp) {
                (Some(ix), Some(dp)) if !rd.dsa_dense => Some(DeviceQ4Index {
                    q: upload_linear(&format!("blk.{i}.indexer.q"), &ix.split.0)?,
                    k: upload_linear(&format!("blk.{i}.indexer.k"), &ix.split.1)?,
                    q_norm: upload_vec(&ix.q_norm)?,
                    k_norm: upload_vec(&ix.k_norm)?,
                    store: backend.allocate_buffer(rd.pool_blocks * rd.block_size * dp.store_width() * 4).map_err(|e| e.to_string())?,
                }),
                _ => None,
            };
            layers.push(DeviceQ4Layer { attn: gate(&ql.attn, &format!("blk.{i}.hc_attn"))?, ffn: gate(&ql.ffn, &format!("blk.{i}.hc_ffn"))?, ple, index });
        }
        Ok(DeviceQwen4 { layers, head: gate(&self.head, "output_hc")? })
    }
}

/// The n-gram table as a device weight: the stored format when the backend reads it, else F16.
fn upload_table(backend: &dyn Backend, name: &str, table: &Embedding) -> Result<DeviceWeight, String> {
    let quant = QuantType::from_gguf_type_id(table.kind);
    let raw = matches!(table.kind, 0 | 1 | 30) || backend.capabilities().supports_quant(quant);
    if raw {
        return backend.upload_weight(name, &table.data, quant, table.rows, table.cols, None).map_err(|e| e.to_string());
    }
    let values = compute::dequantize(&table.data, table.kind, table.rows * table.cols)?;
    let half: Vec<u8> = values.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect();
    backend.upload_weight(name, &half, QuantType::F16, table.rows, table.cols, None).map_err(|e| e.to_string())
}

/// Bytes the n-gram tables may take on the device before the rows of a step are gathered on the host instead.
const TABLE_BUDGET: usize = 2 << 30;

impl RustDecoder {
    /// Parameters of the token indexer as the pooled `dsa_select`: pools of `ratio` tokens, `budget / ratio` of them kept, the unfinished last pool too.
    pub(in crate::rust_decoder) fn q4_dsa_params(&self) -> Option<DsaParams> {
        let (heads, dim, budget, ratio) = self.cfg.qwen4.as_ref()?.indexer?;
        Some(DsaParams {
            heads,
            dim,
            topk: budget,
            stride: self.stride,
            block_size: self.block_size,
            scale: (dim as f32).powf(-0.5),
            head_scale: 1.0,
            kpool: ratio.max(1),
            always_tail: true,
        })
    }

    /// Do the n-gram tables go to the device (`FLLM_QWEN4_TABLE=host|device` forces a side)?
    fn q4_table_on_device(&self, model: &Qwen4Model) -> bool {
        match std::env::var("FLLM_QWEN4_TABLE").ok().as_deref() {
            Some("host") => false,
            Some("device") => true,
            _ => model.layers.iter().filter_map(|l| l.ple.as_ref()).map(|p| p.embedding.data.len()).sum::<usize>() <= TABLE_BUDGET,
        }
    }

    /// `(layers with an n-gram embedding, state bytes of one)`: the convolution history of every channel behind the DeltaNet states.
    pub(in crate::rust_decoder) fn q4_state_extra(&self) -> (usize, usize) {
        match self.cfg.qwen4.as_ref() {
            Some(qc) if qc.ple.iter().any(|&f| f) => {
                let bytes = (qc.hc * self.cfg.hidden * (qc.ple_kernel - 1) * qc.ngram * 4).next_multiple_of(256);
                (qc.ple.iter().filter(|&&f| f).count(), bytes)
            }
            _ => (0, 0),
        }
    }

    /// Can the device run the whole model? The streams exist on one side only, so a missing piece sends every layer to the host (and is logged).
    pub(in crate::rust_decoder) fn qwen4_device_possible(&self, backend: &dyn Backend) -> bool {
        let (Some(qc), Some(_)) = (self.cfg.qwen4.as_ref(), self.qwen4.as_ref()) else { return false };
        if std::env::var("FLLM_QWEN4").is_ok_and(|v| v.eq_ignore_ascii_case("host")) || self.par.is_some() || self.cfg.residual_scale != 1.0 {
            return false;
        }
        let Ok(stream) = backend.create_stream() else { return false };
        let (kernel, dilation) = if qc.ple.iter().any(|&f| f) { (qc.ple_kernel, qc.ngram) } else { (1, 1) };
        let mut missing: Vec<&str> = Vec::new();
        if !stream.supports_qwen4(qc.hc, kernel, dilation) {
            missing.push("the stream, n-gram embedding and indexer kernels");
        }
        if !stream.supports_hc(qc.hc) {
            missing.push("the residual stream update");
        }
        if !backend.capabilities().supports_grouped_rms_norm {
            missing.push("the grouped RMS norm");
        }
        if !self.delta_shape().is_some_and(|s| stream.supports_delta(&s)) {
            missing.push("the DeltaNet layers");
        }
        if let (Some((_, dim, _, _)), Some(dp)) = (qc.indexer, self.q4_dsa_params()) {
            if self.kv_kind != 1 || !stream.supports_dsa(&dp, self.cfg.head_dim, self.kv_kind) || dim > 256 {
                missing.push("the token indexer for this shape or KV cache format");
            }
        }
        if !self.layers.iter().all(|l| matches!(l.attn, Attn::Linear(_) | Attn::Std { .. })) {
            missing.push("a layer that is neither DeltaNet nor attention");
        }
        if !missing.is_empty() {
            eprintln!("[fllm-rs] Qwen3.8-Flash-Next needs {} on the {} backend", missing.join(", "), backend.name());
        }
        missing.is_empty()
    }

    /// The mixing of a gate: writes the block input to `input` (`rows x hidden`); returns the injection weights (`rows x hc`) when the gate has them.
    fn q4_mix(&self, stream: &mut dyn Stream, scratch: &mut BufferPool, gate: &DeviceQ4Gate, streams: &DeviceBuffer, input: &mut DeviceBuffer, rows: usize) -> Result<Option<DeviceBuffer>, String> {
        let err = |e: crate::backend::BackendError| e.to_string();
        let qc = self.cfg.qwen4.as_ref().ok_or("Qwen3.8-Flash-Next configuration missing")?;
        let (hc, hidden) = (qc.hc, self.cfg.hidden);
        let width = hc * hidden;
        let mut normed = scratch.acquire(rows * width * 4).map_err(err)?;
        stream.rms_norm_grouped(&mut normed, streams, &gate.norm, self.cfg.eps, hc).map_err(err)?;
        let mut low = scratch.acquire(rows * qc.lowrank * 4).map_err(err)?;
        stream.quantized_linear(&mut low, &normed, &gate.down, gate.down.bias.as_ref()).map_err(err)?;
        stream.q4_act(&mut low, 0, 1.0 / hc as f32, 1.0).map_err(err)?;
        let mut gates = scratch.acquire(rows * width * 4).map_err(err)?;
        stream.quantized_linear(&mut gates, &low, &gate.up, gate.up.bias.as_ref()).map_err(err)?;
        stream.q4_gate_mix(input, &normed, &gates, rows, hc, hidden).map_err(err)?;
        let weights = match &gate.inject {
            Some(w) => {
                let mut post = scratch.acquire(rows * hc * 4).map_err(err)?;
                stream.quantized_linear(&mut post, &normed, w, w.bias.as_ref()).map_err(err)?;
                stream.q4_act(&mut post, 1, 1.0 / hc as f32, 2.0).map_err(err)?;
                Some(post)
            }
            None => None,
        };
        for buf in [normed, low, gates] {
            scratch.recycle(buf);
        }
        Ok(weights)
    }

    /// Start of a block: the block input goes to `x_buf` (the buffer the layer adds its output to), the weights and the input are kept for `hc_leave`.
    pub(in crate::rust_decoder) fn q4_enter(&self, stream: &mut dyn Stream, scratch: &mut BufferPool, gate: &DeviceQ4Gate, streams: &DeviceBuffer, x_buf: &mut DeviceBuffer, rows: usize) -> Result<HcBlock, String> {
        let post = self.q4_mix(stream, scratch, gate, streams, x_buf, rows)?.ok_or("a block gate without injection weights")?;
        let mut base = scratch.acquire(rows * self.cfg.hidden * 4).map_err(|e| e.to_string())?;
        stream.copy_buffer(&mut base, x_buf).map_err(|e| e.to_string())?;
        Ok(HcBlock { post, comb: None, base })
    }

    /// The collapse of the streams before the head: `x_buf` receives the gated mean of the normalised streams.
    pub(in crate::rust_decoder) fn q4_head(&self, stream: &mut dyn Stream, scratch: &mut BufferPool, model: &DeviceQwen4, streams: &DeviceBuffer, x_buf: &mut DeviceBuffer, rows: usize) -> Result<(), String> {
        self.q4_mix(stream, scratch, &model.head, streams, x_buf, rows)?;
        Ok(())
    }

    /// The n-gram feature of delta layer `i` is added to every stream. `plan`: the state runs of the pass (`state_plan`).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::rust_decoder) fn q4_ple_device(
        &self,
        stream: &mut dyn Stream,
        scratch: &mut BufferPool,
        (i, dev): (usize, &DeviceQ4Ple),
        plan: &[DeltaPlanRun],
        streams: &mut DeviceBuffer,
        ids: &[i64],
        tables: &[i64],
        lengths: &[i64],
    ) -> Result<(), String> {
        let err = |e: crate::backend::BackendError| e.to_string();
        let model = self.qwen4.as_deref().ok_or("Qwen3.8-Flash-Next weights missing")?;
        let qc = self.cfg.qwen4.as_ref().ok_or("Qwen3.8-Flash-Next configuration missing")?;
        let ple = model.layers[i].ple.as_ref().ok_or("n-gram embedding missing")?;
        let (rows, hidden, hc) = (lengths.len(), self.cfg.hidden, qc.hc);
        let width = hc * hidden;
        let heads = (qc.ngram - 1) * qc.heads_per_ngram;
        let head_dim = qc.ple_embed / heads;
        // the hash of the token history runs on the host
        let row_ids = self.q4_ple_ids(model, qc, i, ple, ids, tables, lengths, false)?;
        let mut embeddings = scratch.acquire(rows * qc.ple_embed * 4).map_err(err)?;
        match &dev.table {
            Some(table) => stream.embedding_lookup(&mut embeddings, &row_ids, table).map_err(err)?,
            None => {
                let mut host = Vec::with_capacity(rows * qc.ple_embed);
                for &id in &row_ids {
                    host.extend_from_slice(&ple.embedding.row(id as usize)?);
                }
                debug_assert_eq!(host.len(), row_ids.len() * head_dim);
                stream.upload(&mut embeddings, cast_f32_to_u8(&host)).map_err(err)?;
            }
        }
        let mut key = scratch.acquire(rows * width * 4).map_err(err)?;
        stream.quantized_linear(&mut key, &embeddings, &dev.key, dev.key.bias.as_ref()).map_err(err)?;
        let mut value = scratch.acquire(rows * hidden * 4).map_err(err)?;
        stream.quantized_linear(&mut value, &embeddings, &dev.value, dev.value.bias.as_ref()).map_err(err)?;
        let mut key_normed = scratch.acquire(rows * width * 4).map_err(err)?;
        stream.rms_norm_grouped(&mut key_normed, &key, &dev.norm_key, self.cfg.eps, hc).map_err(err)?;
        let mut query = scratch.acquire(rows * width * 4).map_err(err)?;
        stream.rms_norm_grouped(&mut query, streams, &dev.norm_query, self.cfg.eps, hc).map_err(err)?;
        let mut gated = scratch.acquire(rows * width * 4).map_err(err)?;
        stream.q4_ple_gate(&mut gated, &key_normed, &query, &value, rows, hc, hidden).map_err(err)?;
        let mut normed = scratch.acquire(rows * width * 4).map_err(err)?;
        stream.rms_norm_grouped(&mut normed, &gated, &dev.norm_conv, self.cfg.eps, hc).map_err(err)?;
        // the convolution history sits behind the DeltaNet states of the sequence's buffer
        let delta_layers = self.layers.iter().filter(|l| matches!(l.attn, Attn::Linear(_))).count();
        let delta_bytes = self.delta_shape().map_or(0, |s| s.layer_state_bytes());
        let (_, ple_bytes) = self.q4_state_extra();
        let resets = self.delta_claim(delta_layers + dev.ordinal, plan)?;
        let runs: Vec<DeltaRun<'_>> = plan.iter().zip(&resets).map(|(p, &reset)| DeltaRun { state: &p.buf, first_row: p.start, rows: p.n, reset }).collect();
        let mut out = scratch.acquire(rows * width * 4).map_err(err)?;
        stream
            .q4_ple_conv(&mut out, &gated, &normed, &dev.conv, &runs, delta_layers * delta_bytes + dev.ordinal * ple_bytes, width, qc.ple_kernel, qc.ngram)
            .map_err(err)?;
        dump_stream_add("ple", i, &mut *stream, &out, rows * width)?;
        stream.residual_add(streams, &out, 1.0).map_err(err)?;
        for buf in [embeddings, key, value, key_normed, query, gated, normed, out] {
            scratch.recycle(buf);
        }
        Ok(())
    }

    /// The inputs of `dsa_select` and `q4_pool_keys` that all indexer layers of a pass share.
    pub(in crate::rust_decoder) fn q4_step(&self, stream: &mut dyn Stream, scratch: &mut BufferPool, positions: &[usize], tables: &[i64], lengths: &[i64]) -> Result<Q4Step, String> {
        let err = |e: crate::backend::BackendError| e.to_string();
        let (heads, dim, _, ratio) = self.cfg.qwen4.as_ref().and_then(|q| q.indexer).ok_or("token indexer configuration missing")?;
        let rows = lengths.len();
        let mut meta = vec![-1i32; rows * (ratio + 1)];
        let mut gate = vec![NOT_LAST; rows * dim];
        for (start, end, _, first_position) in self.q4_runs(tables, lengths) {
            for r in start..end {
                let position = first_position + (r - start);
                debug_assert_eq!(position, positions[r]);
                if (position + 1) % ratio != 0 {
                    continue;
                }
                gate[r * dim..(r + 1) * dim].fill(0.0);
                let block = position + 1 - ratio;
                meta[r * (ratio + 1)] = block as i32;
                for j in 0..ratio {
                    let p = block + j;
                    meta[r * (ratio + 1) + 1 + j] = if p >= first_position {
                        (start + (p - first_position)) as i32
                    } else {
                        let slot = tables[r * self.stride + p / self.block_size] as usize * self.block_size + p % self.block_size;
                        -1 - slot as i32
                    };
                }
            }
        }
        let upload = |stream: &mut dyn Stream, scratch: &mut BufferPool, values: &[f32]| -> Result<DeviceBuffer, String> {
            let mut buf = scratch.acquire(values.len() * 4).map_err(err)?;
            stream.upload(&mut buf, cast_f32_to_u8(values)).map_err(err)?;
            Ok(buf)
        };
        Ok(Q4Step {
            ones: upload(stream, scratch, &vec![1.0; rows * heads])?,
            gate: upload(stream, scratch, &gate)?,
            ape: upload(stream, scratch, &vec![0.0; ratio.max(1) * dim])?,
            meta,
        })
    }

    /// The tokens every row of an indexer layer attends to (`rows x sel_width` positions, ascending, -1 padded); the caller runs `dsa_attention` on them.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::rust_decoder) fn q4_select_device(
        &self,
        stream: &mut dyn Stream,
        scratch: &mut BufferPool,
        ix: &DeviceQ4Index,
        step: &Q4Step,
        (cos, sin): &(Arc<Vec<f32>>, Arc<Vec<f32>>),
        normed: &DeviceBuffer,
        (i, positions, slots, tables, lengths): (usize, &[usize], &[usize], &[i64], &[i64]),
    ) -> Result<DeviceBuffer, String> {
        let err = |e: crate::backend::BackendError| e.to_string();
        let cfg = &self.cfg;
        let (heads, dim, _, ratio) = cfg.qwen4.as_ref().and_then(|q| q.indexer).ok_or("token indexer configuration missing")?;
        let dp = self.q4_dsa_params().ok_or("token indexer configuration missing")?;
        let rows = lengths.len();
        let rope = |heads: usize| RopeParams { heads, kv_heads: 1, dim, rope_dim: cfg.layer_rope_dims(i), neox: cfg.rope.neox, cos: Arc::clone(cos), sin: Arc::clone(sin) };
        let mut queries = scratch.acquire(rows * heads * dim * 4).map_err(err)?;
        stream.quantized_linear(&mut queries, normed, &ix.q, ix.q.bias.as_ref()).map_err(err)?;
        let mut raw = scratch.acquire(rows * dim * 4).map_err(err)?;
        stream.quantized_linear(&mut raw, normed, &ix.k, ix.k.bias.as_ref()).map_err(err)?;
        device_rms_inplace(stream, scratch, &mut queries, Some(&ix.q_norm), cfg.eps, 1)?;
        // only the queries are rotated here: the keys are pooled first, then rotated at the position of their block
        let mut unused = scratch.acquire(rows * dim * 4).map_err(err)?;
        stream.rope(&mut queries, &mut unused, positions, &rope(heads)).map_err(err)?;
        scratch.recycle(unused);
        let mut keys = scratch.acquire(rows * dim * 4).map_err(err)?;
        stream.q4_pool_keys(&mut keys, &raw, &ix.store, dp.store_width(), &ix.k_norm, &rope(1), &step.meta, ratio, cfg.eps).map_err(err)?;
        let mut sel = scratch.acquire(rows * dp.sel_width() * 4).map_err(err)?;
        let mut store = ix.store.clone();
        let pooled = dp.kpool > 1;
        stream
            .dsa_select(&mut sel, &mut store, &queries, &keys, pooled.then_some(&step.gate), pooled.then_some(&step.ape), &step.ones, slots, tables, lengths, &dp)
            .map_err(err)?;
        for buf in [queries, raw, keys] {
            scratch.recycle(buf);
        }
        Ok(sel)
    }
}

/// `FLLM_DUMP_LAYERS`: the values `buf` holds (a stream-shaped array) go to `dump_layer`.
pub(in crate::rust_decoder) fn dump_stream_add(tag: &str, layer: usize, stream: &mut dyn Stream, buf: &DeviceBuffer, count: usize) -> Result<(), String> {
    if std::env::var_os("FLLM_DUMP_LAYERS").is_none() {
        return Ok(());
    }
    let mut host = vec![0.0f32; count];
    stream.download(cast_f32_to_u8_mut(&mut host), buf).map_err(|e| e.to_string())?;
    dump_layer(tag, layer, &host);
    Ok(())
}
