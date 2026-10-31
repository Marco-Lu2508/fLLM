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
    pub(super) hc_attn: Hc,
    pub(super) hc_ffn: Hc,
    comp: Option<Comp>,
    indexer: Option<Indexer>,
}

pub(super) struct V4Model {
    pub(super) layers: Vec<V4Layer>,
    pub(super) head: Hc,
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
            dump_layer("branch", i, &attn);
            combine_streams(&mut streams, &post, Some(&comb), &attn, rows, hc, hidden);
            let (post, comb, mut normed) = vl.hc_ffn.compute(&streams, rows, hc, hidden, cfg.eps, v.hc_eps, v.hc_iters)?;
            if let Some(norm) = &layer.ffn_norm {
                self.norm(&mut normed, norm);
            }
            let mlp = self.ffn_forward(layer, &normed, &mut workspace)?;
            combine_streams(&mut streams, &post, Some(&comb), &mlp, rows, hc, hidden);
            dump_layer("ffn", i, &streams);
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

/// The compressor of a layer on the device.
pub(super) struct DeviceV4Comp {
    kv: crate::backend::DeviceWeight,
    gate: crate::backend::DeviceWeight,
    pos: crate::backend::DeviceBuffer,
    norm: crate::backend::DeviceBuffer,
}

/// The lightning indexer of a layer on the device.
pub(super) struct DeviceV4Index {
    comp: DeviceV4Comp,
    q_b: crate::backend::DeviceWeight,
    proj: crate::backend::DeviceWeight,
}

/// The attention of a DeepSeek-V4 layer on the device.
pub(super) struct DeviceV4Layer {
    q_a: crate::backend::DeviceWeight,
    q_a_norm: crate::backend::DeviceBuffer,
    q_b: crate::backend::DeviceWeight,
    kv: crate::backend::DeviceWeight,
    kv_norm: crate::backend::DeviceBuffer,
    o_a: Vec<crate::backend::DeviceWeight>,
    o_b: crate::backend::DeviceWeight,
    comp: Option<DeviceV4Comp>,
    index: Option<DeviceV4Index>,
    sinks: crate::backend::DeviceBuffer,
    shape: crate::backend::V4Shape,
    /// Byte offset of the layer's state in the sequence's state buffer.
    state_offset: usize,
}

/// Rotation tables of a pass (cosine, sine per row and pair): at the positions of the rows with the frequencies of the layers without and with
/// compression, and for the window a row completes, per compression ratio.
struct V4Rotations {
    main: crate::backend::DeviceBuffer,
    compress: crate::backend::DeviceBuffer,
    entry: Vec<(usize, crate::backend::DeviceBuffer)>,
}

impl Comp {
    fn upload(&self, name: &str, linear: &dyn Fn(&str, &Linear) -> Result<crate::backend::DeviceWeight, String>, vec: &dyn Fn(&[f32]) -> Result<crate::backend::DeviceBuffer, String>) -> Result<DeviceV4Comp, String> {
        Ok(DeviceV4Comp { kv: linear(&format!("{name}_kv"), &self.kv)?, gate: linear(&format!("{name}_gate"), &self.gate)?, pos: vec(&self.pos)?, norm: vec(&self.norm)? })
    }
}

impl V4Layer {
    pub(super) fn upload(
        &self,
        name: &str,
        shape: crate::backend::V4Shape,
        state_offset: usize,
        sinks: &[f32],
        linear: &dyn Fn(&str, &Linear) -> Result<crate::backend::DeviceWeight, String>,
        vec: &dyn Fn(&[f32]) -> Result<crate::backend::DeviceBuffer, String>,
    ) -> Result<DeviceV4Layer, String> {
        Ok(DeviceV4Layer {
            q_a: linear(&format!("{name}.attn_q_a"), &self.q_a)?,
            q_a_norm: vec(&self.q_a_norm)?,
            q_b: linear(&format!("{name}.attn_q_b"), &self.q_b)?,
            kv: linear(&format!("{name}.attn_kv"), &self.kv)?,
            kv_norm: vec(&self.kv_norm)?,
            o_a: self.o_a.iter().enumerate().map(|(g, l)| linear(&format!("{name}.attn_output_a.{g}"), l)).collect::<Result<Vec<_>, _>>()?,
            o_b: linear(&format!("{name}.attn_output_b"), &self.o_b)?,
            comp: self.comp.as_ref().map(|c| c.upload(&format!("{name}.attn_comp"), linear, vec)).transpose()?,
            index: self
                .indexer
                .as_ref()
                .map(|ix| {
                    Ok::<_, String>(DeviceV4Index {
                        comp: ix.comp.upload(&format!("{name}.indexer.comp"), linear, vec)?,
                        q_b: linear(&format!("{name}.indexer.attn_q_b"), &ix.q_b)?,
                        proj: linear(&format!("{name}.indexer.proj"), &ix.proj)?,
                    })
                })
                .transpose()?,
            sinks: vec(sinks)?,
            shape,
            state_offset,
        })
    }
}

/// `[cos, sin]` per pair of the rotary embedding at each position (as `rope_tail` computes them).
fn rotation_table(freq: &[f32], rope: usize, positions: impl Iterator<Item = usize>) -> Vec<f32> {
    let mut out = Vec::new();
    for position in positions {
        for &f in freq.iter().take(rope / 2) {
            let angle = position as f32 * f;
            out.push(angle.cos());
            out.push(angle.sin());
        }
    }
    out
}

impl RustDecoder {
    /// Shape of the attention of layer `li` for the device operations.
    pub(super) fn v4_shape(&self, li: usize) -> Option<crate::backend::V4Shape> {
        let v = self.cfg.v4.as_ref()?;
        let model = self.v4.as_deref()?;
        let ratio = v.ratios[li];
        let context = self.stride * self.block_size;
        Some(crate::backend::V4Shape {
            heads: self.cfg.heads,
            head_dim: v.head_dim,
            rope: self.cfg.rope.dims,
            window: v.window,
            ratio,
            index: model.layers[li].indexer.as_ref().map(|_| crate::backend::V4Index { heads: v.index_heads, dim: v.index_dim, topk: v.index_topk }),
            max_entries: context.checked_div(ratio).map_or(0, |n| n + 1),
            o_groups: v.o_groups,
        })
    }

    /// Can the accelerator run the whole DeepSeek-V4 model (streams, attention, states, experts)? `FLLM_V4=host` keeps it on the CPU.
    pub(super) fn v4_device_possible(&self, backend: &dyn crate::backend::Backend, clamp_ok: bool) -> bool {
        let (Some(v), Some(_)) = (self.cfg.v4.as_ref(), self.v4.as_deref()) else { return false };
        if self.par.is_some() || std::env::var("FLLM_V4").is_ok_and(|x| x.eq_ignore_ascii_case("host")) {
            return false;
        }
        let Ok(stream) = backend.create_stream() else { return false };
        let shapes: Vec<_> = (0..self.cfg.layers).filter_map(|i| self.v4_shape(i)).collect();
        let state_bytes: usize = shapes.iter().map(|s| s.layout().total * 4).sum();
        let ok = clamp_ok
            && shapes.len() == self.cfg.layers
            && stream.supports_hc(v.hc_mult)
            && stream.supports_copy_columns()
            && shapes.iter().all(|s| stream.supports_v4(s))
            && self.device_glu_mode().is_some();
        if ok && state_bytes > (16usize << 30) {
            eprintln!("[fllm-rs] DeepSeek-V4 stays on the host: the per-sequence state for a context of {} tokens would take {} MiB of device memory", self.stride * self.block_size, state_bytes >> 20);
            return false;
        }
        if ok {
            eprintln!(
                "[fllm-rs] DeepSeek-V4 runs completely on the device ({} MiB of attention state per sequence); the routers and hash tables of its mixture-of-experts layers are evaluated on the host",
                state_bytes >> 20
            );
        }
        ok
    }

    /// One DeepSeek-V4 attention block on the device: `x` receives the output added to it (the hyper-connection around the block takes the difference).
    #[allow(clippy::too_many_arguments)]
    fn v4_attention_device(
        &self,
        stream: &mut dyn crate::backend::Stream,
        scratch: &mut crate::backend::BufferPool,
        li: usize,
        dv: &DeviceV4Layer,
        ones_hd: &crate::backend::DeviceBuffer,
        rot: &V4Rotations,
        plan: &[DeltaPlanRun],
        normed: &crate::backend::DeviceBuffer,
        x: &mut crate::backend::DeviceBuffer,
        rows: usize,
        positions: &[usize],
    ) -> Result<(), String> {
        use crate::backend::{V4Pool, V4Run};
        let err = |e: crate::backend::BackendError| e.to_string();
        let (cfg, eps) = (&self.cfg, self.cfg.eps);
        let v = cfg.v4.as_ref().ok_or("DeepSeek-V4 configuration missing")?;
        let shape = &dv.shape;
        let (heads, hd, rope) = (shape.heads, shape.head_dim, shape.rope);
        let ratio = shape.ratio;
        let cs = if ratio == 0 { &rot.main } else { &rot.compress };
        let cs_entry = rot.entry.iter().find(|(r, _)| *r == ratio).map(|(_, b)| b);
        let project = |stream: &mut dyn crate::backend::Stream, scratch: &mut crate::backend::BufferPool, w: &crate::backend::DeviceWeight, input: &crate::backend::DeviceBuffer| {
            let mut out = scratch.acquire(rows * w.rows * 4).map_err(err)?;
            stream.quantized_linear(&mut out, input, w, w.bias.as_ref()).map_err(err)?;
            Ok::<_, String>(out)
        };
        let normalise = |stream: &mut dyn crate::backend::Stream, scratch: &mut crate::backend::BufferPool, input: crate::backend::DeviceBuffer, weight: &crate::backend::DeviceBuffer| {
            let mut out = scratch.acquire(input.size).map_err(err)?;
            stream.rms_norm(&mut out, &input, weight, eps).map_err(err)?;
            scratch.recycle(input);
            Ok::<_, String>(out)
        };
        let q_resid = project(stream, scratch, &dv.q_a, normed)?;
        let q_resid = normalise(stream, scratch, q_resid, &dv.q_a_norm)?;
        let q = project(stream, scratch, &dv.q_b, &q_resid)?;
        let mut q = normalise(stream, scratch, q, ones_hd)?;
        stream.v4_rope(&mut q, cs, rows, heads, hd, rope, 1.0).map_err(err)?;
        let kv = project(stream, scratch, &dv.kv, normed)?;
        let mut kv = normalise(stream, scratch, kv, &dv.kv_norm)?;
        stream.v4_rope(&mut kv, cs, rows, 1, hd, rope, 1.0).map_err(err)?;
        let runs: Vec<V4Run<'_>> = plan.iter().map(|p| V4Run { state: &p.buf, first_row: p.start, rows: p.n, position: p.position }).collect();
        let layout = shape.layout();
        if let (Some(comp), Some(entry)) = (&dv.comp, cs_entry) {
            let ckv = project(stream, scratch, &comp.kv, normed)?;
            let cgate = project(stream, scratch, &comp.gate, normed)?;
            stream.v4_compress(V4Pool::Attention, shape, &runs, dv.state_offset, &ckv, &cgate, &comp.pos, &comp.norm, entry, eps).map_err(err)?;
            scratch.recycle(ckv);
            scratch.recycle(cgate);
        }
        let mut sel: Option<crate::backend::DeviceBuffer> = None;
        if let (Some(ix), Some(index), Some(entry)) = (&dv.index, shape.index, cs_entry) {
            let ikv = project(stream, scratch, &ix.comp.kv, normed)?;
            let igate = project(stream, scratch, &ix.comp.gate, normed)?;
            stream.v4_compress(V4Pool::Index, shape, &runs, dv.state_offset, &ikv, &igate, &ix.comp.pos, &ix.comp.norm, entry, eps).map_err(err)?;
            scratch.recycle(ikv);
            scratch.recycle(igate);
            // a batch in which no row sees more entries than the indexer keeps attends to all of them: no selection needed
            if positions.iter().any(|&p| (p + 1) / ratio > index.topk) {
                let mut iq = project(stream, scratch, &ix.q_b, &q_resid)?;
                stream.v4_rope(&mut iq, cs, rows, index.heads, index.dim, rope, 1.0).map_err(err)?;
                let iw = project(stream, scratch, &ix.proj, normed)?;
                let picked = scratch.acquire(rows * index.topk * 4).map_err(err)?;
                let params = crate::backend::DsaParams {
                    heads: index.heads,
                    dim: index.dim,
                    topk: index.topk,
                    stride: 1,
                    block_size: shape.entry_cap(),
                    scale: (index.dim as f32).powf(-0.5),
                    head_scale: (index.heads as f32).powf(-0.5),
                    kpool: 1,
                    always_tail: false,
                };
                for (run, p) in runs.iter().zip(plan) {
                    // rows that see no entry yet score a single (unused) key: the selection needs a length of at least one
                    let lengths: Vec<i64> = (0..run.rows).map(|i| ((p.position + i + 1) / ratio).max(1) as i64).collect();
                    let store = run.state.slice(dv.state_offset + layout.index_entries * 4, shape.entry_cap() * index.dim * 4).map_err(err)?;
                    let q_run = iq.slice(run.first_row * index.heads * index.dim * 4, run.rows * index.heads * index.dim * 4).map_err(err)?;
                    let w_run = iw.slice(run.first_row * index.heads * 4, run.rows * index.heads * 4).map_err(err)?;
                    let mut sel_run = picked.slice(run.first_row * index.topk * 4, run.rows * index.topk * 4).map_err(err)?;
                    stream.dsa_score_select(&mut sel_run, &store, &q_run, &w_run, &vec![0i64; run.rows], &lengths, &params).map_err(err)?;
                }
                scratch.recycle(iq);
                scratch.recycle(iw);
                sel = Some(picked);
            }
        }
        scratch.recycle(q_resid);
        let mut attn = scratch.acquire(rows * heads * hd * 4).map_err(err)?;
        stream.v4_attention(&mut attn, &q, &kv, shape, &runs, dv.state_offset, &dv.sinks, sel.as_ref(), cs).map_err(err)?;
        stream.v4_ring_write(shape, &runs, dv.state_offset, &kv).map_err(err)?;
        for buf in [q, kv].into_iter().chain(sel) {
            scratch.recycle(buf);
        }
        // grouped output projection: the attention output is group-major, one block of rows x group_in values per group
        let group_in = heads * hd / v.o_groups;
        let mut grouped = scratch.acquire(rows * v.o_groups * v.o_lora * 4).map_err(err)?;
        for (g, o_a) in dv.o_a.iter().enumerate() {
            let block = attn.slice(g * rows * group_in * 4, rows * group_in * 4).map_err(err)?;
            let projected = project(stream, scratch, o_a, &block)?;
            stream.copy_columns(&mut grouped, v.o_groups * v.o_lora, g * v.o_lora, &projected, rows, v.o_lora).map_err(err)?;
            scratch.recycle(projected);
        }
        scratch.recycle(attn);
        let branch = project(stream, scratch, &dv.o_b, &grouped)?;
        scratch.recycle(grouped);
        stream.residual_add(x, &branch, 1.0).map_err(err)?;
        scratch.recycle(branch);
        let _ = li;
        Ok(())
    }

    /// All layers of a DeepSeek-V4 model on the device for a batch: `streams` (rows x hc x hidden) holds the residual streams, `x_buf` is the block
    /// input / residual buffer of the hyper-connections. The collapse of the streams at the end is left to the caller.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn v4_device_layers(
        &self,
        stream: &mut dyn crate::backend::Stream,
        scratch: &mut crate::backend::BufferPool,
        dev_weights: &DeviceModelWeights,
        streams: &mut crate::backend::DeviceBuffer,
        x_buf: &mut crate::backend::DeviceBuffer,
        ids: &[i64],
        positions: &[usize],
        tables: &[i64],
        lengths: &[i64],
    ) -> Result<(), String> {
        let err = |e: crate::backend::BackendError| e.to_string();
        let (cfg, model) = (&self.cfg, self.v4.as_deref().ok_or("DeepSeek-V4 weights missing")?);
        let v = cfg.v4.as_ref().ok_or("DeepSeek-V4 configuration missing")?;
        let hcd = dev_weights.hc.as_ref().ok_or("DeepSeek-V4 hyper-connection weights missing")?;
        let ones_hd = dev_weights.v4_ones.as_ref().ok_or("DeepSeek-V4 weights missing on the device")?;
        let (rows, hidden, rope) = (positions.len(), cfg.hidden, cfg.rope.dims);
        // hash-routed layers read the token ids of the rows
        *self.row_ids.lock().map_err(|_| "row id lock poisoned")? = ids.to_vec();
        let state_bytes = dev_weights.layers.iter().filter_map(|l| l.v4.as_ref()).map(|d| d.shape.layout().total * 4).sum::<usize>();
        let plan = self.state_plan(1, state_bytes, tables, lengths)?;
        self.delta_claim(0, &plan)?;
        let upload = |stream: &mut dyn crate::backend::Stream, scratch: &mut crate::backend::BufferPool, table: &[f32]| -> Result<crate::backend::DeviceBuffer, String> {
            let mut buf = scratch.acquire(table.len().max(1) * 4).map_err(err)?;
            stream.upload(&mut buf, cast_f32_to_u8(table)).map_err(err)?;
            Ok(buf)
        };
        let main = upload(stream, scratch, &rotation_table(&model.freq_main, rope, positions.iter().copied()))?;
        let compress = upload(stream, scratch, &rotation_table(&model.freq_compress, rope, positions.iter().copied()))?;
        let mut entry = Vec::new();
        for &ratio in v.ratios.iter().filter(|&&r| r > 0) {
            if !entry.iter().any(|(r, _)| *r == ratio) {
                let table = rotation_table(&model.freq_compress, rope, positions.iter().map(|&p| (p + 1).saturating_sub(ratio)));
                entry.push((ratio, upload(stream, scratch, &table)?));
            }
        }
        let rot = V4Rotations { main, compress, entry };
        for (i, layer) in self.layers.iter().enumerate() {
            let dev_layer = &dev_weights.layers[i];
            let dv = dev_layer.v4.as_ref().ok_or("DeepSeek-V4 layer without device weights")?;
            // attention
            let block = self.hc_enter(stream, scratch, &hcd.attn[i], &hcd.ones, streams, x_buf, rows)?;
            let mut normed = scratch.acquire(rows * hidden * 4).map_err(err)?;
            match &dev_layer.attn_norm {
                Some(norm) => device_norm(stream, cfg.norm, &mut normed, x_buf, norm, dev_layer.attn_norm_b.as_ref(), cfg.eps, self.groups)?,
                None => stream.copy_buffer(&mut normed, x_buf).map_err(err)?,
            }
            self.v4_attention_device(stream, scratch, i, dv, ones_hd, &rot, &plan, &normed, x_buf, rows, positions)?;
            scratch.recycle(normed);
            self.hc_leave(stream, scratch, streams, x_buf, block, rows, i, false)?;
            // feed-forward
            let block = self.hc_enter(stream, scratch, &hcd.ffn[i], &hcd.ones, streams, x_buf, rows)?;
            let mut ffn_normed = scratch.acquire(rows * hidden * 4).map_err(err)?;
            match &dev_layer.ffn_norm {
                Some(norm) => device_norm(stream, cfg.norm, &mut ffn_normed, x_buf, norm, dev_layer.ffn_norm_b.as_ref(), cfg.eps, self.groups)?,
                None => stream.copy_buffer(&mut ffn_normed, x_buf).map_err(err)?,
            }
            let mut out = None;
            if let (Ffn::Moe(_), Some(dm)) = (&layer.ffn, dev_layer.moe.as_ref()) {
                out = self.moe_fast(stream, scratch, dm, &ffn_normed, rows)?;
            }
            if out.is_none() && !dev_layer.host_ffn {
                if let (Ffn::Moe(moe), Some(_)) = (&layer.ffn, dev_layer.moe.as_ref()) {
                    let mut host_in = vec![0.0f32; rows * hidden];
                    stream.download(cast_f32_to_u8_mut(&mut host_in), &ffn_normed).map_err(err)?;
                    out = self.moe_on_device(stream, scratch, moe, dev_layer.moe.as_ref(), &ffn_normed, &host_in, rows)?;
                }
            }
            let out = match out {
                Some(buf) => buf,
                None => {
                    let mut host_in = vec![0.0f32; rows * hidden];
                    stream.download(cast_f32_to_u8_mut(&mut host_in), &ffn_normed).map_err(err)?;
                    let mut workspace = compute::ProjectionGroupWorkspace::default();
                    let host_out = self.ffn_forward(layer, &host_in, &mut workspace)?;
                    let mut buf = scratch.acquire(rows * hidden * 4).map_err(err)?;
                    stream.upload(&mut buf, cast_f32_to_u8(&host_out)).map_err(err)?;
                    buf
                }
            };
            stream.residual_add(x_buf, &out, 1.0).map_err(err)?;
            scratch.recycle(out);
            scratch.recycle(ffn_normed);
            self.hc_leave(stream, scratch, streams, x_buf, block, rows, i, true)?;
        }
        scratch.recycle(rot.main);
        scratch.recycle(rot.compress);
        for (_, buf) in rot.entry {
            scratch.recycle(buf);
        }
        Ok(())
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
