//! The parallel parts of the decoder: cutting the weights of a layer for one rank, and the collectives that join the pieces again.
//! This file is a child module of `rust_decoder` (so it reaches the private layer types); the decoder calls into it at a few places
//! (`load_layers`, `forward_with_hidden`, `host_layers`, `moe_forward`, `ffn_forward`).
use super::*;
use crate::parallel::config::EpDispatch;
use crate::parallel::ctx::{ParCtx, StepDesc, COMM_ERROR};
use crate::parallel::plan::{attention_split, vocab_chunk, vocab_range};
use crate::parallel::shard::{alignment, even_split, shard_linear, shard_linear_rows_multi, Align, ShardDim, ShardSpec};
use std::ops::Range;

fn comm<T>(result: Result<T, String>) -> Result<T, String> {
    result.map_err(|e| format!("{COMM_ERROR}{e}"))
}

pub(super) fn dummy_linear() -> Linear {
    Linear::from_parts(Tensor { shape: vec![1, 1], data: vec![0.0] }, 1, 1, None).expect("a 1 x 1 projection is valid")
}

pub(super) fn placeholder_expert() -> Expert {
    Expert { gate: None, up: dummy_linear(), down: dummy_linear() }
}

/// A layer that another pipeline stage runs: nothing loaded, never executed here.
pub(super) fn placeholder_layer() -> Layer {
    Layer {
        attn: Attn::None,
        indexer: None,
        shortcut: None,
        takes_shortcut: false,
        dsa_shared: false,
        output_gate: false,
        chunk: None,
        attn_gate: None,
        value_moe: None,
        xielu: None,
        sinks: None,
        attn_sub_norm: None,
        ffn_sub_norm: None,
        out: dummy_linear(),
        ffn: Ffn::None,
        attn_norm: None,
        ffn_norm: None,
        post_attn_norm: None,
        post_ffn_norm: None,
        q_norm: None,
        k_norm: None,
        q_norm_bias: None,
        k_norm_bias: None,
        window: None,
        gate_per_head: false,
        rope_table: 0,
        kv_shared: false,
        k_eq_v: false,
        gemma: None,
        parallel: None,
        zamba: None,
    }
}

fn align_of(l: &Linear, gpu: bool) -> Result<Align, String> {
    match l.packed_storage() {
        Some((kind, _)) => alignment(kind, gpu),
        None => Ok(Align { k: 1, n: 1 }),
    }
}

/// Output rows `range` of `l` (column-parallel).
fn rows_part(l: &Linear, what: &str, range: Range<usize>, gpu: bool) -> Result<Linear, String> {
    let align = align_of(l, gpu).map_err(|e| format!("{what}: {e}"))?;
    if !range.len().is_multiple_of(align.n) {
        return Err(format!("{what}: a piece of {} output rows is not a multiple of {} (tile of the GPU kernels)", range.len(), align.n));
    }
    shard_linear(l, ShardSpec::from_range(ShardDim::Rows, range), true, what)
}

/// Input columns `range` of `l` (row-parallel); only the first tensor rank keeps the bias.
fn cols_part(l: &Linear, what: &str, range: Range<usize>, par: &ParCtx) -> Result<Linear, String> {
    let align = align_of(l, par.gpu).map_err(|e| format!("{what}: {e}"))?;
    let expected = even_split(what, l.cols, align.k, par.tp_size(), par.tp_rank)?;
    if expected != range {
        return Err(format!("{what}: internal error, column range {range:?} differs from an even split {expected:?}"));
    }
    shard_linear(l, ShardSpec::from_range(ShardDim::Cols, range), par.tp_rank == 0, what)
}

/// gate / up by rows, down by columns: one gated or plain feed-forward (a layer, an expert, or the shared expert).
fn shard_triplet(
    par: &ParCtx,
    gate: Option<&Linear>,
    up: &Linear,
    down: &Linear,
    what: &str,
) -> Result<(Option<Linear>, Linear, Linear), String> {
    let tp = par.tp_size();
    let align = align_of(down, par.gpu).map_err(|e| format!("{what}.down: {e}"))?;
    let range = even_split(&format!("{what}_down.weight"), down.cols, align.k, tp, par.tp_rank)?;
    if up.rows != down.cols {
        return Err(format!("{what}: up has {} rows but down has {} columns", up.rows, down.cols));
    }
    let g = match gate {
        Some(g) => Some(rows_part(g, &format!("{what}_gate.weight"), range.clone(), par.gpu)?),
        None => None,
    };
    let u = rows_part(up, &format!("{what}_up.weight"), range.clone(), par.gpu)?;
    let d = cols_part(down, &format!("{what}_down.weight"), range, par)?;
    Ok((g, u, d))
}

/// Cut layer `index` for this rank (tensor parallelism and expert parallelism). Called right after the layer was loaded, so that
/// only one layer's unsplit data is ever alive.
pub(super) fn shard_layer(par: &ParCtx, layer: &mut Layer, index: usize, cfg: &ModelConfig) -> Result<(), String> {
    let tp = par.tp_size();
    if tp == 1 {
        return Ok(());
    }
    let r = par.tp_rank;
    let p = format!("blk.{index}");
    if layer.attn_sub_norm.is_some() || layer.ffn_sub_norm.is_some() {
        return Err(format!("{p}: sub-norms over the attention output and the feed-forward activation cannot be split across tensor-parallel ranks"));
    }
    match &mut layer.attn {
        Attn::None => {}
        Attn::Std { q, k, v } => {
            if layer.value_moe.is_some() {
                return Err(format!("{p}: value experts cannot be split across tensor-parallel ranks"));
            }
            if cfg.qk_norm == QkNorm::Full {
                return Err(format!("{p}: a query/key norm over all heads cannot be split across tensor-parallel ranks"));
            }
            let split = attention_split(cfg.heads, cfg.kv_heads, tp)?;
            let (qh, kh) = (split.q_range(r), split.kv_range(r));
            let (dk, dv) = (cfg.head_dim, cfg.value_dim);
            let q_name = format!("{p}.attn_q.weight");
            let new_q = if layer.output_gate {
                if cfg.arch.starts_with("qwen35") {
                    // [query_h, gate_h] stored side by side for every head
                    rows_part(q, &q_name, qh.start * 2 * dk..qh.end * 2 * dk, par.gpu)?
                } else {
                    let width = cfg.heads * dk;
                    shard_linear_rows_multi(q, &[qh.start * dk..qh.end * dk, width + qh.start * dk..width + qh.end * dk], &q_name)?
                }
            } else {
                rows_part(q, &q_name, qh.start * dk..qh.end * dk, par.gpu)?
            };
            let new_k = rows_part(k, &format!("{p}.attn_k.weight"), kh.start * dk..kh.end * dk, par.gpu)?;
            let new_v = rows_part(v, &format!("{p}.attn_v.weight"), kh.start * dv..kh.end * dv, par.gpu)?;
            *q = new_q;
            *k = new_k;
            *v = new_v;
            layer.out = cols_part(&layer.out, &format!("{p}.attn_output.weight"), qh.start * dv..qh.end * dv, par)?;
            if let Some(g) = &layer.attn_gate {
                layer.attn_gate = Some(rows_part(g, &format!("{p}.attn_gate.weight"), qh.start * dv..qh.end * dv, par.gpu)?);
            }
            if let Some(s) = &layer.sinks {
                layer.sinks = Some(s[qh.clone()].to_vec());
            }
        }
        _ => {
            return Err(format!(
                "{p}: this layer type (state-space, short convolution, latent attention) cannot be split across tensor-parallel ranks; use --pipeline-parallel-size"
            ));
        }
    }
    let ep = par.cfg.expert_parallel;
    match &mut layer.ffn {
        Ffn::None => {}
        Ffn::Gated { gate, up, down } => {
            let (g, u, d) = shard_triplet(par, Some(gate), up, down, &format!("{p}.ffn"))?;
            *gate = g.ok_or("gate lost")?;
            *up = u;
            *down = d;
        }
        Ffn::Plain { up, down } => {
            let (_, u, d) = shard_triplet(par, None, up, down, &format!("{p}.ffn"))?;
            *up = u;
            *down = d;
        }
        Ffn::Moe(moe) => {
            if moe.latent.is_some() {
                return Err(format!("{p}: latent mixture-of-experts layers cannot be split across tensor-parallel ranks"));
            }
            let experts = moe.experts.len();
            if ep {
                let mine = ParCtx::expert_range(experts, tp, r);
                for (e, expert) in moe.experts.iter_mut().enumerate() {
                    if !mine.contains(&e) {
                        *expert = placeholder_expert();
                    }
                }
            } else {
                for (e, expert) in moe.experts.iter_mut().enumerate() {
                    let (g, u, d) = shard_triplet(par, expert.gate.as_ref(), &expert.up, &expert.down, &format!("{p}.ffn_exps.{e}"))?;
                    *expert = Expert { gate: g, up: u, down: d };
                }
            }
            let replicate_shared = ep && par.cfg.ep_dispatch == EpDispatch::AllToAll;
            if !replicate_shared {
                if let Some((shared, gate_vector)) = moe.shared.take() {
                    let fits = shared.down.cols.is_multiple_of(tp)
                        && align_of(&shared.down, par.gpu).is_ok_and(|a| (shared.down.cols / tp).is_multiple_of(a.k) && (shared.down.cols / tp).is_multiple_of(a.n));
                    if fits {
                        let (g, u, d) = shard_triplet(par, shared.gate.as_ref(), &shared.up, &shared.down, &format!("{p}.ffn_shexp"))?;
                        moe.shared = Some((Expert { gate: g, up: u, down: d }, gate_vector));
                    } else if r == 0 {
                        // does not divide: rank 0 computes the whole shared expert, the sum over ranks adds it once
                        moe.shared = Some((shared, gate_vector));
                    }
                }
            }
        }
    }
    Ok(())
}

/// The configuration seen by one rank's decoder: its share of the heads and of the feed-forward width.
pub(super) fn localize_cfg(par: &ParCtx, cfg: &ModelConfig) -> Result<ModelConfig, String> {
    let mut out = cfg.clone();
    let tp = par.tp_size();
    if tp == 1 {
        return Ok(out);
    }
    if !cfg.uniform_geometry() {
        return Err("tensor parallelism is not supported for models with per-layer attention geometry".into());
    }
    let split = attention_split(cfg.heads, cfg.kv_heads, tp)?;
    out.heads = split.heads_local;
    out.kv_heads = split.kv_local;
    out.ff = cfg.ff / tp;
    if let Some(slopes) = &cfg.alibi_slopes {
        out.alibi_slopes = Some(slopes[split.q_range(par.tp_rank)].to_vec());
    }
    if let Some(moe) = &mut out.moe {
        if !par.cfg.expert_parallel {
            moe.expert_ff /= tp;
        }
        moe.shared_ff /= tp;
    }
    Ok(out)
}

/// Pipeline stages keep a KV cache only for their own layers: in the returned configuration the other layers get no KV heads, so the
/// KV pool layout (host buffers; the device pools are cut from it) is as big as the stage's share of the layers. Layers that read the cache of an earlier layer have to find
/// it on the same stage.
pub(super) fn stage_local_kv(par: &ParCtx, cfg: &ModelConfig) -> Result<ModelConfig, String> {
    let mut cfg = cfg.clone();
    if par.pp_size() == 1 {
        return Ok(cfg);
    }
    let range = par.layer_range(cfg.layers);
    for layer in range.clone() {
        if let Some(source) = cfg.layer_kv_source(layer) {
            if !range.contains(&source) {
                return Err(format!(
                    "layer {layer} reads the KV cache of layer {source}, which another pipeline stage holds; this model cannot be split at that point (use fewer devices)"
                ));
            }
        }
    }
    let mut geoms = cfg.layer_geoms.take().unwrap_or_else(|| {
        (0..cfg.layers)
            .map(|_| crate::config::LayerGeom {
                heads: cfg.heads,
                kv_heads: cfg.kv_heads,
                head_dim: cfg.head_dim,
                value_dim: cfg.value_dim,
                kv_source: None,
                ff: cfg.ff,
            })
            .collect()
    });
    for (layer, geom) in geoms.iter_mut().enumerate() {
        if !range.contains(&layer) {
            geom.kv_heads = 0;
            geom.kv_source = None;
        }
    }
    cfg.layer_geoms = Some(geoms);
    Ok(cfg)
}

/// The output head: this rank's slice of the vocabulary.
pub(super) fn shard_head(par: &ParCtx, head: Linear, vocab: usize) -> Result<Linear, String> {
    let tp = par.tp_size();
    if tp == 1 {
        return Ok(head);
    }
    let n = align_of(&head, par.gpu)?.n;
    let range = vocab_range(vocab, tp, n, par.tp_rank);
    if range.is_empty() {
        return Err(format!("the vocabulary of {vocab} rows is too small for {tp} tensor-parallel ranks"));
    }
    shard_linear(&head, ShardSpec::from_range(ShardDim::Rows, range), true, "output.weight")
}

/// Vocabulary-parallel embedding: this rank keeps the rows `first..first+held`; the other rows read as zeros and the lookups are summed.
pub(super) fn shard_embedding(par: &ParCtx, embedding: &mut Embedding) -> Result<(), String> {
    let tp = par.tp_size();
    if tp == 1 || !par.cfg.vocab_parallel_embedding {
        return Ok(());
    }
    let chunk = embedding.rows.div_ceil(tp);
    let first = (par.tp_rank * chunk).min(embedding.rows);
    let held = chunk.min(embedding.rows - first);
    embedding.data = embedding.data.slice(first * embedding.row_bytes, held * embedding.row_bytes)?;
    embedding.first_row = first;
    embedding.held = held;
    Ok(())
}

/// Row groups of a batch that belong to one sequence (consecutive rows with the same block table), then packed into at most
/// `want` contiguous micro-batches of similar size.
pub(super) fn microbatch_ranges(tables: &[i64], stride: usize, rows: usize, want: usize) -> Vec<Range<usize>> {
    if rows == 0 || want <= 1 || stride == 0 {
        return std::iter::once(0..rows).collect();
    }
    let mut groups: Vec<Range<usize>> = Vec::new();
    let mut start = 0;
    for r in 1..=rows {
        if r == rows || tables[r * stride..(r + 1) * stride] != tables[(r - 1) * stride..r * stride] {
            groups.push(start..r);
            start = r;
        }
    }
    if groups.len() < 2 {
        return std::iter::once(0..rows).collect();
    }
    let target = rows.div_ceil(want.min(groups.len()));
    let mut out: Vec<Range<usize>> = Vec::new();
    let mut current = groups[0].clone();
    for g in &groups[1..] {
        if current.len() >= target {
            out.push(current);
            current = g.clone();
        } else {
            current.end = g.end;
        }
    }
    out.push(current);
    out
}

impl RustDecoder {
    /// Rank 0 keeps the handles of the other ranks' threads so that dropping the decoder ends them.
    pub(crate) fn attach_group_threads(&self, handles: Vec<std::thread::JoinHandle<()>>) {
        if let Some(par) = &self.par {
            par.attach_workers(handles);
        }
    }

    pub(super) fn par_tp_reduce(&self, data: &mut [f32]) -> Result<(), String> {
        match &self.par {
            Some(par) => comm(par.tp_reduce(data)),
            None => Ok(()),
        }
    }

    /// The embedding rows are split by vocabulary: sum the lookups of the tensor-parallel group.
    pub(super) fn par_embed_reduce(&self, x: &mut [f32]) -> Result<(), String> {
        match &self.par {
            Some(par) if par.cfg.vocab_parallel_embedding && par.tp_size() > 1 => comm(par.tp_reduce(x)),
            _ => Ok(()),
        }
    }

    /// Expert parallelism: drop the routes to experts that another rank holds.
    pub(super) fn par_filter_routes(&self, routed: &mut [Vec<(usize, f32)>]) {
        if let Some(par) = &self.par {
            if par.cfg.expert_parallel && par.tp_size() > 1 {
                let mine = ParCtx::expert_range(routed.len(), par.tp_size(), par.tp_rank);
                for (e, route) in routed.iter_mut().enumerate() {
                    if !mine.contains(&e) {
                        route.clear();
                    }
                }
            }
        }
    }

    /// The feed-forward of a layer including the sum over the tensor-parallel group.
    pub(super) fn ffn_forward(
        &self,
        layer: &Layer,
        input: &[f32],
        workspace: &mut compute::ProjectionGroupWorkspace,
    ) -> Result<Vec<f32>, String> {
        let Some(par) = &self.par else {
            return self.ffn_forward_local(layer, input, workspace);
        };
        if par.tp_size() == 1 {
            return self.ffn_forward_local(layer, input, workspace);
        }
        if let (Ffn::Moe(moe), true, EpDispatch::AllToAll) = (&layer.ffn, par.cfg.expert_parallel, par.cfg.ep_dispatch) {
            return self.moe_alltoall(par, moe, input, workspace);
        }
        let mut out = self.ffn_forward_local(layer, input, workspace)?;
        comm(par.tp_reduce(&mut out))?;
        Ok(out)
    }

    /// Mixture of experts with token dispatch and combine by all-to-all (`--expert-parallel --ep-dispatch alltoall`). Every rank
    /// owns a contiguous block of the tokens: it sends each of its tokens to the ranks that hold the chosen experts, receives the
    /// tokens that chose its experts, computes them, and sends the results back; the block is then gathered on all ranks. The
    /// per-token sum runs over the experts in ascending order, like the single-device code.
    fn moe_alltoall(
        &self,
        par: &ParCtx,
        moe: &Moe,
        input: &[f32],
        workspace: &mut compute::ProjectionGroupWorkspace,
    ) -> Result<Vec<f32>, String> {
        let cfg = self.cfg.moe.as_ref().ok_or("MoE layer without MoE configuration")?;
        let (tp, rank) = (par.tp_size(), par.tp_rank);
        let hidden = self.cfg.hidden;
        let tokens = input.len() / hidden;
        if tokens >= 1 << 24 {
            return Err("too many tokens for the all-to-all dispatch".into());
        }
        let routed = self.moe_route(moe, input)?;
        let experts = routed.len();
        let block = tokens.div_ceil(tp);
        let mine = (rank * block).min(tokens)..((rank + 1) * block).min(tokens);
        let owner = |e: usize| (0..tp).find(|&d| ParCtx::expert_range(experts, tp, d).contains(&e)).unwrap_or(tp - 1);
        let entry = 3 + hidden;
        let mut parts: Vec<Vec<f32>> = vec![Vec::new(); tp];
        for (e, route) in routed.iter().enumerate() {
            for &(t, w) in route {
                if !mine.contains(&t) {
                    continue;
                }
                let part = &mut parts[owner(e)];
                let x = &input[t * hidden..(t + 1) * hidden];
                if cfg.input_scaled {
                    part.extend([t as f32, e as f32, 1.0]);
                    part.extend(x.iter().map(|v| v * w));
                } else {
                    part.extend([t as f32, e as f32, w]);
                    part.extend_from_slice(x);
                }
            }
        }
        let send_counts: Vec<usize> = parts.iter().map(Vec::len).collect();
        let send: Vec<f32> = parts.concat();
        let recv_counts = comm(par.tp.exchange_counts(&send_counts))?;
        let mut recv = vec![0.0f32; recv_counts.iter().sum()];
        comm(par.tp.all_to_all(&send, &send_counts, &mut recv, &recv_counts))?;
        // compute the received tokens expert by expert
        let total_entries = recv.len() / entry;
        let mut by_expert: Vec<Vec<usize>> = vec![Vec::new(); experts];
        for i in 0..total_entries {
            by_expert[recv[i * entry + 1] as usize].push(i);
        }
        let mut results = vec![0.0f32; total_entries * hidden];
        for (e, list) in by_expert.iter().enumerate() {
            if list.is_empty() {
                continue;
            }
            let mut gathered = Vec::with_capacity(list.len() * hidden);
            for &i in list {
                gathered.extend_from_slice(&recv[i * entry + 3..(i + 1) * entry]);
            }
            let out = self.expert_forward(&moe.experts[e], &gathered, workspace)?;
            for (slot, &i) in list.iter().enumerate() {
                results[i * hidden..(i + 1) * hidden].copy_from_slice(&out[slot * hidden..(slot + 1) * hidden]);
            }
        }
        // send [token, weight, result...] back to the rank the entry came from, in the order it was received
        let reply_width = 2 + hidden;
        let mut reply = Vec::with_capacity(total_entries * reply_width);
        let mut reply_counts = Vec::with_capacity(tp);
        for &count in &recv_counts {
            reply_counts.push(count / entry * reply_width);
        }
        for i in 0..total_entries {
            reply.extend([recv[i * entry], recv[i * entry + 2]]);
            reply.extend_from_slice(&results[i * hidden..(i + 1) * hidden]);
        }
        let back_counts: Vec<usize> = send_counts.iter().map(|c| c / entry * reply_width).collect();
        let mut back = vec![0.0f32; back_counts.iter().sum()];
        comm(par.tp.all_to_all(&reply, &reply_counts, &mut back, &back_counts))?;
        let mut own = vec![0.0f32; block * hidden];
        for i in 0..back.len() / reply_width {
            let row = &back[i * reply_width..(i + 1) * reply_width];
            let t = row[0] as usize - mine.start;
            let w = row[1];
            for (d, v) in own[t * hidden..(t + 1) * hidden].iter_mut().zip(&row[2..]) {
                *d += w * v;
            }
        }
        if let Some((shared, gate)) = &moe.shared {
            if !mine.is_empty() {
                let x = &input[mine.start * hidden..mine.end * hidden];
                let result = self.expert_forward(shared, x, workspace)?;
                for (local, t) in mine.clone().enumerate() {
                    let xt = &input[t * hidden..(t + 1) * hidden];
                    let scale = gate.as_ref().map_or(1.0, |g| {
                        let dot: f32 = g.iter().zip(xt).map(|(a, b)| a * b).sum();
                        1.0 / (1.0 + (-dot).exp())
                    });
                    for (d, r) in own[local * hidden..(local + 1) * hidden].iter_mut().zip(&result[local * hidden..(local + 1) * hidden]) {
                        *d += scale * r;
                    }
                }
            }
        }
        let mut gathered = vec![0.0f32; tp * block * hidden];
        comm(par.tp.all_gather(&own, &mut gathered))?;
        gathered.truncate(tokens * hidden);
        Ok(gathered)
    }

    /// Logits of this rank's vocabulary slice (`rows x local`) to the full `rows x vocab` on every rank of the group.
    pub(super) fn par_head_gather(&self, local: Vec<f32>, rows: usize) -> Result<Vec<f32>, String> {
        let Some(par) = &self.par else { return Ok(local) };
        let tp = par.tp_size();
        if tp == 1 {
            return Ok(local);
        }
        let vocab = self.cfg.vocab;
        let align = align_of(&self.head, par.gpu)?.n;
        let chunk = vocab_chunk(vocab, tp, align);
        let mine = vocab_range(vocab, tp, align, par.tp_rank);
        if local.len() != rows * mine.len() {
            return Err(format!("{} local logits for {rows} rows of {} vocabulary entries", local.len(), mine.len()));
        }
        let mut send = vec![0.0f32; rows * chunk];
        for row in 0..rows {
            send[row * chunk..row * chunk + mine.len()].copy_from_slice(&local[row * mine.len()..(row + 1) * mine.len()]);
        }
        let mut recv = vec![0.0f32; tp * rows * chunk];
        comm(par.tp.all_gather(&send, &mut recv))?;
        let mut full = vec![0.0f32; rows * vocab];
        for r in 0..tp {
            let range = vocab_range(vocab, tp, align, r);
            for row in 0..rows {
                full[row * vocab + range.start..row * vocab + range.end]
                    .copy_from_slice(&recv[(r * rows + row) * chunk..(r * rows + row) * chunk + range.len()]);
            }
        }
        Ok(full)
    }

    /// Greedy ids from the vocabulary slices without moving the logits: every rank finds the largest entry of its slice, the
    /// `(value, index)` pairs are gathered and the largest wins (the lowest index on a tie, like the single-device argmax).
    #[allow(dead_code)] // used by the tests; the device path will call it once logits are sampled on the GPU
    pub(crate) fn par_argmax(&self, local: &[f32], rows: usize) -> Result<Vec<i32>, String> {
        let Some(par) = &self.par else { return Err("not a parallel decoder".into()) };
        let tp = par.tp_size();
        let vocab = self.cfg.vocab;
        let align = align_of(&self.head, par.gpu)?.n;
        let mine = vocab_range(vocab, tp, align, par.tp_rank);
        let mut pairs = vec![0.0f32; rows * 2];
        for row in 0..rows {
            let slice = &local[row * mine.len()..(row + 1) * mine.len()];
            let mut best = 0;
            for (i, v) in slice.iter().enumerate() {
                if *v > slice[best] {
                    best = i;
                }
            }
            pairs[row * 2] = slice[best];
            pairs[row * 2 + 1] = f32::from_bits((mine.start + best) as u32);
        }
        let mut all = vec![0.0f32; tp * rows * 2];
        if tp > 1 {
            comm(par.tp.all_gather(&pairs, &mut all))?;
        } else {
            all.copy_from_slice(&pairs);
        }
        let mut ids = Vec::with_capacity(rows);
        for row in 0..rows {
            let mut best_value = f32::NEG_INFINITY;
            let mut best_index = u32::MAX;
            for r in 0..tp {
                let value = all[(r * rows + row) * 2];
                let index = all[(r * rows + row) * 2 + 1].to_bits();
                if value > best_value || (value == best_value && index < best_index) {
                    best_value = value;
                    best_index = index;
                }
            }
            ids.push(best_index as i32);
        }
        Ok(ids)
    }

    /// Sizes of the key and value buffers one rank needs for the host path (after its heads were localised).
    pub(crate) fn kv_buffer_lens(&self) -> (usize, usize) {
        (self.kvl.key_total, self.kvl.value_total)
    }

    /// Forward step of a parallel decoder: rank 0 announces the step, every rank runs its stage on micro-batches, the logits end
    /// up on rank 0.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn par_forward(
        &self,
        par: &Arc<ParCtx>,
        ids: &[i64],
        lengths: &[i64],
        tables: &[i64],
        key: &mut [u16],
        value: &mut [u16],
        logit_rows: &[usize],
        embeddings: bool,
        overrides: &[(usize, Vec<f32>)],
        capture: Option<&mut Vec<f32>>,
    ) -> Result<Vec<f32>, String> {
        if embeddings || capture.is_some() {
            return Err("embedding output and MTP hidden-state capture are not available with tensor / pipeline parallelism".into());
        }
        if par.is_leader() {
            let desc = StepDesc {
                ids: ids.to_vec(),
                lengths: lengths.to_vec(),
                tables: tables.to_vec(),
                logit_rows: logit_rows.to_vec(),
                embeddings,
                overrides: overrides.to_vec(),
            };
            let mut bytes = desc.encode();
            comm(par.control.broadcast_bytes(&mut bytes, 0))?;
        }
        let rows = ids.len();
        let vocab = self.cfg.vocab;
        let wanted = if par.pp_size() > 1 {
            if par.cfg.microbatches > 0 {
                par.cfg.microbatches
            } else {
                par.pp_size()
            }
        } else {
            1
        };
        let ranges = microbatch_ranges(tables, self.stride, rows, wanted);
        let mut logits = vec![0.0f32; if par.is_last_stage() { logit_rows.len() * vocab } else { 0 }];
        for range in &ranges {
            let mut positions = Vec::new();
            let mut local_rows = Vec::new();
            for (position, &row) in logit_rows.iter().enumerate() {
                if range.contains(&row) {
                    positions.push(position);
                    local_rows.push(row - range.start);
                }
            }
            let sub_overrides: Vec<(usize, Vec<f32>)> =
                overrides.iter().filter(|(row, _)| range.contains(row)).map(|(row, x)| (*row - range.start, x.clone())).collect();
            let out = self.forward_batch(
                &ids[range.clone()],
                &lengths[range.clone()],
                &tables[range.start * self.stride..range.end * self.stride],
                key,
                value,
                &local_rows,
                false,
                &sub_overrides,
                None,
            )?;
            if par.is_last_stage() {
                if out.len() != positions.len() * vocab {
                    return Err(format!("the last stage produced {} logits for {} rows", out.len(), positions.len()));
                }
                for (slot, &position) in positions.iter().enumerate() {
                    logits[position * vocab..(position + 1) * vocab].copy_from_slice(&out[slot * vocab..(slot + 1) * vocab]);
                }
            }
        }
        let output = par.output_rank();
        if par.is_leader() {
            if par.rank == output {
                return Ok(logits);
            }
            let mut received = vec![0.0f32; logit_rows.len() * vocab];
            comm(par.control.recv(&mut received, output))?;
            return Ok(received);
        }
        if par.rank == output {
            comm(par.control.send(&logits, 0))?;
        }
        Ok(Vec::new())
    }

    /// Stage input of a micro-batch: the activations of the previous stage.
    pub(super) fn par_recv_activations(&self, rows: usize) -> Result<Vec<f32>, String> {
        let Some(par) = &self.par else { return Err("not a pipeline stage".into()) };
        let mut x = vec![0.0f32; rows * self.cfg.hidden];
        comm(par.pp.recv(&mut x, par.stage - 1))?;
        Ok(x)
    }

    pub(super) fn par_send_activations(&self, x: &[f32]) -> Result<(), String> {
        let Some(par) = &self.par else { return Err("not a pipeline stage".into()) };
        comm(par.pp.send(x, par.stage + 1))
    }
}
