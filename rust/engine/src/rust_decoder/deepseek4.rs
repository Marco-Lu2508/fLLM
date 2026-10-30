//! DeepSeek-V4 on the host: manifold-constrained hyper-connections, shared-KV attention with a sliding window and compressed entries.
//!
//! The residual stream of a token is `hc_mult` vectors. Around every attention and feed-forward block, `Hc::compute` derives from the
//! flattened streams a collapse weight (the block input is their weighted sum), an expansion weight and a doubly-stochastic mixing
//! matrix (Sinkhorn-Knopp); the block output is added to the mixed streams. Attention has one key/value head (K = V) of width
//! `head_dim`, rotary embeddings on its trailing `rope` values (undone on the output), a per-head sink and a grouped low-rank output
//! projection. Besides the last `window` tokens a layer attends to compressed entries: every `ratio` tokens a softmax-weighted pool
//! of their projected values (a compressed sparse layer, ratio 4, pools windows of `2 * ratio` tokens that overlap by one window
//! and keeps the `index_topk` entries a lightning indexer scores best; a heavily compressed layer pools windows of `ratio` tokens and
//! attends to all entries).
//!
//! Everything a sequence keeps lives in the per-layer state (`LayerState`), because nothing here uses the KV pools:
//! * `conv`: `[ring_count, ring_head, index_entries, 0]`, the ring of the last `window` key vectors, then the entries of the indexer;
//! * `rec`: `[buffered, entries, index_buffered, has_overlap, has_index_overlap, 0, 0, 0]`, the not yet pooled projections of the
//!   compressor and the indexer, the Ca slice of the last pooled window of each (the overlap), then the compressed entries.
use super::*;

/// Three vectors: the mixing weights of a hyper-connection `(post, comb, collapsed)` or a pooled entry with the slice kept as overlap.
type VecTriple = (Vec<f32>, Vec<f32>, Vec<f32>);

pub(super) struct Hc {
    pub(super) mix: Linear,
    pub(super) base: Vec<f32>,
    pub(super) scale: Vec<f32>,
}

struct Comp {
    kv: Linear,
    gate: Linear,
    /// `ratio x (coff * dim)` position bias added to the pooling logits.
    pos: Vec<f32>,
    norm: Vec<f32>,
}

struct Indexer {
    comp: Comp,
    q_b: Linear,
    proj: Linear,
}

pub(super) struct V4Layer {
    q_a: Linear,
    q_a_norm: Vec<f32>,
    q_b: Linear,
    kv: Linear,
    kv_norm: Vec<f32>,
    o_a: Vec<Linear>,
    o_b: Linear,
    hc_attn: Hc,
    hc_ffn: Hc,
    comp: Option<Comp>,
    indexer: Option<Indexer>,
}

pub(super) struct V4Model {
    layers: Vec<V4Layer>,
    head: Hc,
    freq_main: Vec<f32>,
    freq_compress: Vec<f32>,
}

/// Sizes and offsets of the state of one layer.
struct Lay {
    hd: usize,
    window: usize,
    ratio: usize,
    /// 2 for compressed sparse layers (two series with overlapping windows), 1 otherwise.
    coff: usize,
    idim: usize,
    indexed: bool,
}

const CONV_HEADER: usize = 4;
const REC_HEADER: usize = 8;

impl Lay {
    fn width(&self) -> usize {
        self.coff * self.hd
    }
    fn iwidth(&self) -> usize {
        if self.indexed { 2 * self.idim } else { 0 }
    }
    fn idx_entries(&self) -> usize {
        CONV_HEADER + self.window * self.hd
    }
    fn buf_kv(&self) -> usize {
        REC_HEADER
    }
    fn buf_gate(&self) -> usize {
        self.buf_kv() + self.ratio * self.width()
    }
    fn ov_kv(&self) -> usize {
        self.buf_gate() + self.ratio * self.width()
    }
    fn ov_gate(&self) -> usize {
        self.ov_kv() + self.ratio * self.hd * (self.coff - 1)
    }
    fn ibuf_kv(&self) -> usize {
        self.ov_gate() + self.ratio * self.hd * (self.coff - 1)
    }
    fn ibuf_gate(&self) -> usize {
        self.ibuf_kv() + self.ratio * self.iwidth()
    }
    fn iov_kv(&self) -> usize {
        self.ibuf_gate() + self.ratio * self.iwidth()
    }
    fn iov_gate(&self) -> usize {
        self.iov_kv() + if self.indexed { self.ratio * self.idim } else { 0 }
    }
    fn entries(&self) -> usize {
        self.iov_gate() + if self.indexed { self.ratio * self.idim } else { 0 }
    }
    fn blank(&self) -> LayerState {
        LayerState { len: 0, conv: vec![0.0; self.idx_entries()], rec: vec![0.0; self.entries()] }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `x * rsqrt(mean(x^2) + eps)` over every `width` values.
fn rms_plain(x: &mut [f32], width: usize, eps: f32) {
    for row in x.chunks_exact_mut(width) {
        let inv = (row.iter().map(|v| v * v).sum::<f32>() / width as f32 + eps).sqrt().recip();
        row.iter_mut().for_each(|v| *v *= inv);
    }
}

/// Interleaved rotary embedding of the trailing `rope` values of `v` at `position`; `sign` -1 undoes it.
fn rope_tail(v: &mut [f32], rope: usize, position: usize, freq: &[f32], sign: f32) {
    let at = v.len() - rope;
    for (i, &f) in freq.iter().enumerate().take(rope / 2) {
        let angle = position as f32 * f;
        let (c, s) = (angle.cos(), angle.sin() * sign);
        let (a, b) = (v[at + 2 * i], v[at + 2 * i + 1]);
        v[at + 2 * i] = a * c - b * s;
        v[at + 2 * i + 1] = b * c + a * s;
    }
}

/// `streams[k] = post[k] * block + sum_j comb[j][k] * old[j]` per row; without `comb` the old stream of `k` is kept as it is.
pub(super) fn combine_streams(streams: &mut [f32], post: &[f32], comb: Option<&[f32]>, block: &[f32], rows: usize, hc: usize, hidden: usize) {
    let old = streams.to_vec();
    for r in 0..rows {
        for k in 0..hc {
            let out = &mut streams[(r * hc + k) * hidden..(r * hc + k + 1) * hidden];
            let b = &block[r * hidden..(r + 1) * hidden];
            let pk = post[r * hc + k];
            for (d, o) in out.iter_mut().enumerate() {
                let mixed = match comb {
                    Some(comb) => (0..hc).map(|j| comb[(r * hc + j) * hc + k] * old[(r * hc + j) * hidden + d]).sum::<f32>(),
                    None => old[(r * hc + k) * hidden + d],
                };
                *o = pk * b[d] + mixed;
            }
        }
    }
}

impl Hc {
    /// Independent hyper-connection (Hy4): `(post [rows x hc], collapsed [rows x hidden])`, `post = magnitude * sigmoid(..) + eps`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn compute_independent(&self, x: &[f32], rows: usize, hc: usize, hidden: usize, norm_eps: f32, eps: f32, magnitude: f32) -> Result<(Vec<f32>, Vec<f32>), String> {
        let mut flat = x.to_vec();
        rms_plain(&mut flat, hc * hidden, norm_eps);
        let mixes = self.mix.forward(&flat)?;
        let mut post = vec![0.0f32; rows * hc];
        let mut collapsed = vec![0.0f32; rows * hidden];
        for r in 0..rows {
            let m = &mixes[r * 2 * hc..(r + 1) * 2 * hc];
            for n in 0..hc {
                let pre = sigmoid(m[n] * self.scale[0] + self.base[n]) + eps;
                post[r * hc + n] = magnitude * sigmoid(m[hc + n] * self.scale[1] + self.base[hc + n]) + eps;
                let stream = &x[(r * hc + n) * hidden..(r * hc + n + 1) * hidden];
                collapsed[r * hidden..(r + 1) * hidden].iter_mut().zip(stream).for_each(|(o, s)| *o += pre * s);
            }
        }
        Ok((post, collapsed))
    }

    /// The final collapse of the streams: weights `sigmoid(scale * fn(rms(streams)) + base) + eps`.
    pub(super) fn collapse_head(&self, streams: &[f32], rows: usize, hc: usize, hidden: usize, norm_eps: f32, eps: f32) -> Result<Vec<f32>, String> {
        let mut flat = streams.to_vec();
        rms_plain(&mut flat, hc * hidden, norm_eps);
        let mixes = self.mix.forward(&flat)?;
        let mut out = vec![0.0f32; rows * hidden];
        for r in 0..rows {
            for n in 0..hc {
                let pre = sigmoid(mixes[r * hc + n] * self.scale[0] + self.base[n]) + eps;
                let stream = &streams[(r * hc + n) * hidden..(r * hc + n + 1) * hidden];
                out[r * hidden..(r + 1) * hidden].iter_mut().zip(stream).for_each(|(o, s)| *o += pre * s);
            }
        }
        Ok(out)
    }

    pub(super) fn load(mapping: &MappedGguf, p: &str, hc: usize, hidden: usize, outputs: usize, scales: usize) -> Result<Self, String> {
        let mut mix = Linear::load(mapping, &format!("{p}_fn"), outputs, hc * hidden)?;
        mix.enable_fast_cpu();
        Ok(Self { mix, base: vector(mapping, &format!("{p}_base.weight"), outputs)?, scale: vector(mapping, &format!("{p}_scale.weight"), scales)? })
    }

    /// Mixing weights of a batch: `(post [rows x hc], comb [rows x hc x hc], collapsed [rows x hidden])`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn compute(&self, x: &[f32], rows: usize, hc: usize, hidden: usize, norm_eps: f32, eps: f32, iters: usize) -> Result<VecTriple, String> {
        let mut flat = x.to_vec();
        rms_plain(&mut flat, hc * hidden, norm_eps);
        let mixes = self.mix.forward(&flat)?;
        let width = (2 + hc) * hc;
        let mut post = vec![0.0f32; rows * hc];
        let mut comb = vec![0.0f32; rows * hc * hc];
        let mut collapsed = vec![0.0f32; rows * hidden];
        for r in 0..rows {
            let m = &mixes[r * width..(r + 1) * width];
            let pre: Vec<f32> = (0..hc).map(|n| sigmoid(m[n] * self.scale[0] + self.base[n]) + eps).collect();
            for n in 0..hc {
                post[r * hc + n] = 2.0 * sigmoid(m[hc + n] * self.scale[1] + self.base[hc + n]);
            }
            let c = &mut comb[r * hc * hc..(r + 1) * hc * hc];
            for j in 0..hc {
                let row = &mut c[j * hc..(j + 1) * hc];
                for (k, v) in row.iter_mut().enumerate() {
                    *v = m[2 * hc + j * hc + k] * self.scale[2] + self.base[2 * hc + j * hc + k];
                }
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for v in row.iter_mut() {
                    *v = (*v - max).exp();
                    sum += *v;
                }
                row.iter_mut().for_each(|v| *v = *v / sum + eps);
            }
            let column_normalize = |c: &mut [f32]| {
                for k in 0..hc {
                    let s: f32 = (0..hc).map(|j| c[j * hc + k]).sum();
                    for j in 0..hc {
                        c[j * hc + k] /= s + eps;
                    }
                }
            };
            column_normalize(c);
            for _ in 1..iters {
                for j in 0..hc {
                    let s: f32 = c[j * hc..(j + 1) * hc].iter().sum();
                    c[j * hc..(j + 1) * hc].iter_mut().for_each(|v| *v /= s + eps);
                }
                column_normalize(c);
            }
            let out = &mut collapsed[r * hidden..(r + 1) * hidden];
            for n in 0..hc {
                let stream = &x[(r * hc + n) * hidden..(r * hc + n + 1) * hidden];
                out.iter_mut().zip(stream).for_each(|(o, s)| *o += pre[n] * s);
            }
        }
        Ok((post, comb, collapsed))
    }
}

impl Comp {
    fn load(mapping: &MappedGguf, p: &str, hidden: usize, dim: usize, coff: usize, ratio: usize) -> Result<Self, String> {
        let mut kv = Linear::load(mapping, &format!("{p}_kv"), coff * dim, hidden)?;
        let mut gate = Linear::load(mapping, &format!("{p}_gate"), coff * dim, hidden)?;
        kv.enable_fast_cpu();
        gate.enable_fast_cpu();
        Ok(Self { kv, gate, pos: vector(mapping, &format!("{p}_pos.weight"), ratio * coff * dim)?, norm: vector(mapping, &format!("{p}_norm.weight"), dim)? })
    }
}

impl V4Model {
    pub(super) fn load(model: &GgufModel, cfg: &ModelConfig) -> Result<Self, String> {
        let v = cfg.v4.as_ref().ok_or("not a DeepSeek-V4 model")?;
        let mapping = model.mapping();
        let (hidden, hd) = (cfg.hidden, v.head_dim);
        let rope = cfg.rope.dims;
        let freqs = |name: &str| vector(mapping, name, rope / 2);
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("blk.{i}");
            let ratio = v.ratios[i];
            let load = |name: &str, rows: usize, cols: usize| -> Result<Linear, String> {
                let mut l = Linear::load(mapping, &format!("{p}.{name}"), rows, cols)?;
                l.replace_weights(model.adapted_projection(&format!("{p}.{name}.weight")))?;
                l.enable_fast_cpu();
                Ok(l)
            };
            let group_in = cfg.heads * hd / v.o_groups;
            let o_a = (0..v.o_groups)
                .map(|g| {
                    let mut l = Linear::load_rows(mapping, &format!("{p}.attn_output_a.weight"), None, g * v.o_lora, v.o_lora, group_in)?;
                    l.enable_fast_cpu();
                    Ok(l)
                })
                .collect::<Result<Vec<_>, String>>()?;
            let indexed = ratio > 0 && has_tensor(mapping, &format!("{p}.indexer.comp_kv.weight"));
            let coff = if indexed { 2 } else { 1 };
            let comp = if ratio > 0 { Some(Comp::load(mapping, &format!("{p}.attn_comp"), hidden, hd, coff, ratio)?) } else { None };
            let indexer = if indexed {
                Some(Indexer {
                    comp: Comp::load(mapping, &format!("{p}.indexer.comp"), hidden, v.index_dim, 2, ratio)?,
                    q_b: load("indexer.attn_q_b", v.index_heads * v.index_dim, v.q_lora)?,
                    proj: load("indexer.proj", v.index_heads, hidden)?,
                })
            } else {
                None
            };
            layers.push(V4Layer {
                q_a: load("attn_q_a", v.q_lora, hidden)?,
                q_a_norm: vector(mapping, &format!("{p}.attn_q_a_norm.weight"), v.q_lora)?,
                q_b: load("attn_q_b", cfg.heads * hd, v.q_lora)?,
                kv: load("attn_kv", hd, hidden)?,
                kv_norm: vector(mapping, &format!("{p}.attn_kv_norm.weight"), hd)?,
                o_a,
                o_b: load("attn_output_b", hidden, v.o_groups * v.o_lora)?,
                hc_attn: Hc::load(mapping, &format!("{p}.hc_attn"), v.hc_mult, hidden, (2 + v.hc_mult) * v.hc_mult, 3)?,
                hc_ffn: Hc::load(mapping, &format!("{p}.hc_ffn"), v.hc_mult, hidden, (2 + v.hc_mult) * v.hc_mult, 3)?,
                comp,
                indexer,
            });
        }
        Ok(Self {
            layers,
            head: Hc::load(mapping, "output_hc", v.hc_mult, hidden, v.hc_mult, 1)?,
            freq_main: freqs("rope_freqs_main.weight")?,
            freq_compress: freqs("rope_freqs_compress.weight")?,
        })
    }
}

/// Pools `ratio` projected rows into one compressed entry (before the rotary embedding). `prev` is the previous window's first-series
/// slice (kv, gate with bias) of a two-series layer, `None` for the first window.
#[allow(clippy::too_many_arguments)]
fn pool_window(kv: &[f32], gate: &[f32], comp: &Comp, ratio: usize, dim: usize, coff: usize, prev: Option<(&[f32], &[f32])>, eps: f32) -> VecTriple {
    let w = coff * dim;
    let mut entry = vec![0.0f32; dim];
    let mut keep_kv = Vec::new();
    let mut keep_gate = Vec::new();
    for c in 0..dim {
        let mut logits: Vec<(f32, f32)> = Vec::with_capacity(2 * ratio);
        if coff == 2 {
            for j in 0..ratio {
                logits.push(match prev {
                    Some((pk, pg)) => (pg[j * dim + c], pk[j * dim + c]),
                    None => (f32::NEG_INFINITY, 0.0),
                });
            }
        }
        let offset = if coff == 2 { dim } else { 0 };
        for j in 0..ratio {
            logits.push((gate[j * w + offset + c] + comp.pos[j * w + offset + c], kv[j * w + offset + c]));
        }
        let max = logits.iter().map(|l| l.0).fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|l| (l.0 - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        entry[c] = exps.iter().zip(&logits).map(|(e, l)| e / sum * l.1).sum();
    }
    if coff == 2 {
        for j in 0..ratio {
            for c in 0..dim {
                keep_kv.push(kv[j * w + c]);
                keep_gate.push(gate[j * w + c] + comp.pos[j * w + c]);
            }
        }
    }
    rms(&mut entry, &comp.norm, eps);
    (entry, keep_kv, keep_gate)
}

impl RustDecoder {
    /// All layers of a DeepSeek-V4 model for a batch: `x` holds the embeddings (rows x hidden); the result is the collapsed stream.
    pub(super) fn v4_layers(&self, x: &[f32], ids: &[i64], positions: &[usize], tables: &[i64], lengths: &[i64]) -> Result<Vec<f32>, String> {
        let (cfg, model) = (&self.cfg, self.v4.as_deref().ok_or("DeepSeek-V4 weights missing")?);
        let v = cfg.v4.as_ref().ok_or("DeepSeek-V4 configuration missing")?;
        let (rows, hidden, hc) = (positions.len(), cfg.hidden, v.hc_mult);
        *self.row_ids.lock().map_err(|_| "row id lock poisoned")? = ids.to_vec();
        let mut streams = Vec::with_capacity(rows * hc * hidden);
        for r in 0..rows {
            for _ in 0..hc {
                streams.extend_from_slice(&x[r * hidden..(r + 1) * hidden]);
            }
        }
        let mut workspace = compute::ProjectionGroupWorkspace::default();
        for (i, (layer, vl)) in self.layers.iter().zip(&model.layers).enumerate() {
            let (post, comb, mut normed) = vl.hc_attn.compute(&streams, rows, hc, hidden, cfg.eps, v.hc_eps, v.hc_iters)?;
            if let Some(norm) = &layer.attn_norm {
                self.norm(&mut normed, norm);
            }
            let attn = self.v4_attention(i, vl, &normed, positions, tables, lengths)?;
            combine_streams(&mut streams, &post, Some(&comb), &attn, rows, hc, hidden);
            let (post, comb, mut normed) = vl.hc_ffn.compute(&streams, rows, hc, hidden, cfg.eps, v.hc_eps, v.hc_iters)?;
            if let Some(norm) = &layer.ffn_norm {
                self.norm(&mut normed, norm);
            }
            let mlp = self.ffn_forward(layer, &normed, &mut workspace)?;
            combine_streams(&mut streams, &post, Some(&comb), &mlp, rows, hc, hidden);
        }
        model.head.collapse_head(&streams, rows, hc, hidden, cfg.eps, v.hc_eps)
    }

    /// One attention block for a batch; returns the projected output (rows x hidden).
    fn v4_attention(&self, li: usize, vl: &V4Layer, normed: &[f32], positions: &[usize], tables: &[i64], lengths: &[i64]) -> Result<Vec<f32>, String> {
        let (cfg, model) = (&self.cfg, self.v4.as_deref().ok_or("DeepSeek-V4 weights missing")?);
        let v = cfg.v4.as_ref().ok_or("DeepSeek-V4 configuration missing")?;
        let (rows, hd, heads, rope, eps) = (positions.len(), v.head_dim, cfg.heads, cfg.rope.dims, cfg.eps);
        let ratio = v.ratios[li];
        let freq: &[f32] = if ratio == 0 { &model.freq_main } else { &model.freq_compress };
        let mut q_resid = vl.q_a.forward(normed)?;
        rms(&mut q_resid, &vl.q_a_norm, eps);
        let mut q = vl.q_b.forward(&q_resid)?;
        rms_plain(&mut q, hd, eps);
        for (r, &position) in positions.iter().enumerate() {
            for h in 0..heads {
                rope_tail(&mut q[(r * heads + h) * hd..(r * heads + h + 1) * hd], rope, position, freq, 1.0);
            }
        }
        let mut kv = vl.kv.forward(normed)?;
        rms(&mut kv, &vl.kv_norm, eps);
        for (r, &position) in positions.iter().enumerate() {
            rope_tail(&mut kv[r * hd..(r + 1) * hd], rope, position, freq, 1.0);
        }
        let lay = Lay { hd, window: v.window, ratio, coff: if vl.indexer.is_some() { 2 } else { 1 }, idim: v.index_dim, indexed: vl.indexer.is_some() };
        let (ckv, cgate) = match &vl.comp {
            Some(c) => (c.kv.forward(normed)?, c.gate.forward(normed)?),
            None => (Vec::new(), Vec::new()),
        };
        let (ikv, igate, iq, iweights) = match &vl.indexer {
            Some(ix) => (ix.comp.kv.forward(normed)?, ix.comp.gate.forward(normed)?, ix.q_b.forward(&q_resid)?, ix.proj.forward(normed)?),
            None => (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
        };
        let sinks = layer_sinks(&self.layers[li], heads)?;
        let mut attn = vec![0.0f32; rows * heads * hd];
        let scale = (hd as f32).powf(-0.5);
        self.for_each_run(li, tables, lengths, &|| lay.blank(), |start, end, state| {
            for t in start..end {
                let position = positions[t];
                // sliding window ring
                {
                    let n = state.conv[0] as usize;
                    let head = state.conv[1] as usize;
                    let slot = if n < lay.window { (head + n) % lay.window } else { head };
                    let at = CONV_HEADER + slot * hd;
                    state.conv[at..at + hd].copy_from_slice(&kv[t * hd..(t + 1) * hd]);
                    if n < lay.window {
                        state.conv[0] = (n + 1) as f32;
                    } else {
                        state.conv[1] = ((head + 1) % lay.window) as f32;
                    }
                }
                if let Some(c) = &vl.comp {
                    let w = lay.width();
                    let n = state.rec[0] as usize;
                    state.rec[lay.buf_kv() + n * w..lay.buf_kv() + (n + 1) * w].copy_from_slice(&ckv[t * w..(t + 1) * w]);
                    state.rec[lay.buf_gate() + n * w..lay.buf_gate() + (n + 1) * w].copy_from_slice(&cgate[t * w..(t + 1) * w]);
                    state.rec[0] = (n + 1) as f32;
                    if n + 1 == ratio {
                        let entries = state.rec[1] as usize;
                        let has_prev = state.rec[3] > 0.0 && lay.coff == 2;
                        let (entry, keep_kv, keep_gate) = {
                            let kvb = &state.rec[lay.buf_kv()..lay.buf_kv() + ratio * w];
                            let gb = &state.rec[lay.buf_gate()..lay.buf_gate() + ratio * w];
                            let prev = has_prev.then(|| (&state.rec[lay.ov_kv()..lay.ov_kv() + ratio * hd], &state.rec[lay.ov_gate()..lay.ov_gate() + ratio * hd]));
                            pool_window(kvb, gb, c, ratio, hd, lay.coff, prev, eps)
                        };
                        let mut entry = entry;
                        rope_tail(&mut entry, rope, entries * ratio, freq, 1.0);
                        if lay.coff == 2 {
                            state.rec[lay.ov_kv()..lay.ov_kv() + ratio * hd].copy_from_slice(&keep_kv);
                            state.rec[lay.ov_gate()..lay.ov_gate() + ratio * hd].copy_from_slice(&keep_gate);
                            state.rec[3] = 1.0;
                        }
                        state.rec.extend_from_slice(&entry);
                        state.rec[1] = (entries + 1) as f32;
                        state.rec[0] = 0.0;
                    }
                }
                if let Some(ix) = &vl.indexer {
                    let (idim, w) = (lay.idim, lay.iwidth());
                    let n = state.rec[2] as usize;
                    state.rec[lay.ibuf_kv() + n * w..lay.ibuf_kv() + (n + 1) * w].copy_from_slice(&ikv[t * w..(t + 1) * w]);
                    state.rec[lay.ibuf_gate() + n * w..lay.ibuf_gate() + (n + 1) * w].copy_from_slice(&igate[t * w..(t + 1) * w]);
                    state.rec[2] = (n + 1) as f32;
                    if n + 1 == ratio {
                        let entries = state.conv[2] as usize;
                        let has_prev = state.rec[4] > 0.0;
                        let (entry, keep_kv, keep_gate) = {
                            let kvb = &state.rec[lay.ibuf_kv()..lay.ibuf_kv() + ratio * w];
                            let gb = &state.rec[lay.ibuf_gate()..lay.ibuf_gate() + ratio * w];
                            let prev = has_prev.then(|| (&state.rec[lay.iov_kv()..lay.iov_kv() + ratio * idim], &state.rec[lay.iov_gate()..lay.iov_gate() + ratio * idim]));
                            pool_window(kvb, gb, &ix.comp, ratio, idim, 2, prev, eps)
                        };
                        let mut entry = entry;
                        rope_tail(&mut entry, rope, entries * ratio, freq, 1.0);
                        state.rec[lay.iov_kv()..lay.iov_kv() + ratio * idim].copy_from_slice(&keep_kv);
                        state.rec[lay.iov_gate()..lay.iov_gate() + ratio * idim].copy_from_slice(&keep_gate);
                        state.rec[4] = 1.0;
                        state.conv.extend_from_slice(&entry);
                        state.conv[2] = (entries + 1) as f32;
                        state.rec[2] = 0.0;
                    }
                }
                // keys: the window, then the compressed entries this token attends to
                let mut keys: Vec<&[f32]> = Vec::new();
                let ring = state.conv[0] as usize;
                let head = state.conv[1] as usize;
                for i in 0..ring {
                    let at = CONV_HEADER + ((head + i) % lay.window) * hd;
                    keys.push(&state.conv[at..at + hd]);
                }
                let comp_count = state.rec[1] as usize;
                let picked: Vec<usize> = match &vl.indexer {
                    Some(ix) if comp_count > 0 => {
                        let (idim, ih) = (lay.idim, v.index_heads);
                        let mut qi = iq[t * ih * idim..(t + 1) * ih * idim].to_vec();
                        for h in 0..ih {
                            rope_tail(&mut qi[h * idim..(h + 1) * idim], rope, position, freq, 1.0);
                        }
                        let _ = ix;
                        let (isc, hsc) = ((idim as f32).powf(-0.5), (ih as f32).powf(-0.5));
                        let scored: Vec<(f32, u32)> = (0..comp_count)
                            .map(|s| {
                                let at = lay.idx_entries() + s * idim;
                                let k = &state.conv[at..at + idim];
                                let mut score = 0.0f32;
                                for h in 0..ih {
                                    let dot = compute::dot(&qi[h * idim..(h + 1) * idim], k);
                                    if dot > 0.0 {
                                        score += dot * isc * (iweights[t * ih + h] * hsc);
                                    }
                                }
                                (score, s as u32)
                            })
                            .collect();
                        best_positions(scored, v.index_topk).into_iter().map(|s| s as usize).collect()
                    }
                    _ => (0..comp_count).collect(),
                };
                let base = lay.entries();
                for s in picked {
                    keys.push(&state.rec[base + s * hd..base + (s + 1) * hd]);
                }
                let head_out = &mut attn[t * heads * hd..(t + 1) * heads * hd];
                head_out.par_chunks_mut(hd).enumerate().for_each(|(h, out)| {
                    let query = &q[(t * heads + h) * hd..(t * heads + h + 1) * hd];
                    let mut scores: Vec<f32> = keys.iter().map(|k| compute::dot(query, k) * scale).collect();
                    let sink = sinks[h];
                    let max = scores.iter().copied().fold(sink, f32::max);
                    let mut sum = (sink - max).exp();
                    for s in &mut scores {
                        *s = (*s - max).exp();
                        sum += *s;
                    }
                    for (s, k) in scores.iter().zip(&keys) {
                        let w = s / sum;
                        out.iter_mut().zip(k.iter()).for_each(|(o, x)| *o += w * x);
                    }
                    rope_tail(out, rope, position, freq, -1.0);
                });
            }
        })?;
        // grouped output projection
        let group_in = heads * hd / v.o_groups;
        let mut grouped = vec![0.0f32; rows * v.o_groups * v.o_lora];
        for (g, o_a) in vl.o_a.iter().enumerate() {
            let mut slice = Vec::with_capacity(rows * group_in);
            for r in 0..rows {
                slice.extend_from_slice(&attn[r * heads * hd + g * group_in..r * heads * hd + (g + 1) * group_in]);
            }
            let projected = o_a.forward(&slice)?;
            for r in 0..rows {
                grouped[(r * v.o_groups + g) * v.o_lora..(r * v.o_groups + g + 1) * v.o_lora].copy_from_slice(&projected[r * v.o_lora..(r + 1) * v.o_lora]);
            }
        }
        vl.o_b.forward(&grouped)
    }
}

fn layer_sinks(layer: &Layer, heads: usize) -> Result<&[f32], String> {
    layer.sinks.as_deref().filter(|s| s.len() == heads).ok_or_else(|| "DeepSeek-V4 layer without attention sinks".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_tail_is_undone_by_the_negative_sign_and_leaves_the_head_alone() {
        let freq = [1.0f32, 0.1];
        let mut v: Vec<f32> = (0..8).map(|i| i as f32 * 0.3 - 1.0).collect();
        let original = v.clone();
        rope_tail(&mut v, 4, 7, &freq, 1.0);
        assert_eq!(&v[..4], &original[..4]);
        assert!(v[4..].iter().zip(&original[4..]).any(|(a, b)| (a - b).abs() > 1e-3));
        rope_tail(&mut v, 4, 7, &freq, -1.0);
        assert!(v.iter().zip(&original).all(|(a, b)| (a - b).abs() < 1e-5));
    }

    #[test]
    fn equal_logits_pool_to_the_mean_and_the_overlap_slice_is_kept() {
        // two series (coff 2), dim 2, ratio 2: kv rows [Ca | Cb]
        let comp = Comp {
            kv: Linear::from_parts(compute::Tensor { shape: vec![1, 1], data: vec![0.0] }, 1, 1, None).unwrap(),
            gate: Linear::from_parts(compute::Tensor { shape: vec![1, 1], data: vec![0.0] }, 1, 1, None).unwrap(),
            pos: vec![0.0; 2 * 4],
            norm: vec![1.0; 2],
        };
        let kv = [1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0];
        let gate = [0.0; 8];
        let (entry, keep_kv, keep_gate) = pool_window(&kv, &gate, &comp, 2, 2, 2, None, 1e-6);
        // the first window has no previous series: the entry is the mean of the two Cb rows, normalised
        let mean = [20.0f32, 30.0];
        let inv = ((mean[0] * mean[0] + mean[1] * mean[1]) / 2.0 + 1e-6).sqrt().recip();
        assert!((entry[0] - mean[0] * inv).abs() < 1e-5 && (entry[1] - mean[1] * inv).abs() < 1e-5);
        assert_eq!(keep_kv, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(keep_gate, vec![0.0; 4]);
    }

    #[test]
    fn state_regions_do_not_overlap() {
        let lay = Lay { hd: 4, window: 3, ratio: 2, coff: 2, idim: 2, indexed: true };
        assert_eq!(lay.idx_entries(), CONV_HEADER + 12);
        let offsets = [lay.buf_kv(), lay.buf_gate(), lay.ov_kv(), lay.ov_gate(), lay.ibuf_kv(), lay.ibuf_gate(), lay.iov_kv(), lay.iov_gate(), lay.entries()];
        assert!(offsets.windows(2).all(|w| w[0] < w[1]));
        let state = lay.blank();
        assert_eq!(state.rec.len(), lay.entries());
        let plain = Lay { hd: 4, window: 3, ratio: 4, coff: 1, idim: 2, indexed: false };
        assert_eq!(plain.ov_kv(), plain.ov_gate());
        assert_eq!(plain.ibuf_kv(), plain.iov_gate());
    }
}
