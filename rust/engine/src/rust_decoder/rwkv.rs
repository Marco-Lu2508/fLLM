//! RWKV-6 ("Finch") and RWKV-7 ("Goose") layers in the decoder (host path).
//!
//! Every layer is a time-mix block (token shift, low-rank adapters, a matrix-valued state per head) followed by a
//! channel-mix block (token shift, squared ReLU), each with its own LayerNorm (`attn_norm`, `attn_norm_2`) and residual add; the
//! layer is computed as a whole by [`RustDecoder::rwkv_layer`]. The norm of layer 0 on the embeddings is `token_embd_norm`.
//!
//! GGUF tensors of a layer `blk.N` follow llama.cpp: `time_mix_{receptance,key,value,output}` (+ `time_mix_gate` for RWKV-6),
//! `time_mix_ln`, `channel_mix_{key,value}` (+ `channel_mix_receptance` for RWKV-6); RWKV-7 adds `time_mix_{w,a,v,g}{0,1,2}` (the
//! low-rank adapters of the decay, the in-context learning rate, the value residual and the output gate), `time_mix_k_k`,
//! `time_mix_k_a`, `time_mix_r_k` and the mixing vectors `time_mix_lerp_{r,w,k,v,a,g}` (or the six stacked in
//! `time_mix_lerp_fused`); RWKV-6 adds `time_mix_lerp_{x,w,k,v,r,g}`, `time_mix_w1` / `time_mix_w2` (adapters of the five mixing
//! vectors), `time_mix_decay` with `time_mix_decay_w1` / `_w2`, and `time_mix_first` (the bonus `u`).
//!
//! The state of a sequence layer (`LayerState`): `conv` holds the previous normed inputs of the time-mix and the channel-mix block
//! (`2 * hidden`), `rec` the WKV matrices (`heads x head_size x head_size`; RWKV-7 `[value][key]`, RWKV-6 `[key][value]`).
use super::*;

/// A small dense matrix `[rows][cols]` (an adapter of a few dozen rows), applied with plain loops.
struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Mat {
    /// Loads a 2-D tensor with GGUF dimensions `[cols, rows]`; `cols` is checked when given.
    fn load(mapping: &MappedGguf, name: &str, cols: usize) -> Result<Self, String> {
        let t = Tensor::load(mapping, name)?;
        match t.shape[..] {
            [c, r] if c == cols => Ok(Self { rows: r, cols: c, data: t.data }),
            _ => Err(format!("invalid dimensions for {name}: {:?}, expected [{cols}, rows]", t.shape)),
        }
    }

    fn mul(&self, x: &[f32], out: &mut [f32]) {
        for (o, row) in out.iter_mut().zip(self.data.chunks_exact(self.cols)) {
            *o = row.iter().zip(x).map(|(a, b)| a * b).sum();
        }
    }
}

/// `u -> v2 (act(v1 u)) + bias`.
struct Adapter {
    down: Mat,
    up: Mat,
    bias: Option<Vec<f32>>,
}

impl Adapter {
    fn load(mapping: &MappedGguf, p: &str, tag: &str, hidden: usize, bias: bool) -> Result<Self, String> {
        let down = Mat::load(mapping, &format!("{p}.time_mix_{tag}1.weight"), hidden)?;
        let up = Mat::load(mapping, &format!("{p}.time_mix_{tag}2.weight"), down.rows)?;
        if up.rows != hidden {
            return Err(format!("invalid {p}.time_mix_{tag}2.weight: {} rows, expected {hidden}", up.rows));
        }
        Ok(Self { down, up, bias: if bias { Some(vector(mapping, &format!("{p}.time_mix_{tag}0.weight"), hidden)?) } else { None } })
    }

    fn apply(&self, u: &[f32], act: impl Fn(f32) -> f32, mid: &mut Vec<f32>, out: &mut [f32]) {
        mid.resize(self.down.rows, 0.0);
        self.down.mul(u, mid);
        mid.iter_mut().for_each(|m| *m = act(*m));
        self.up.mul(mid, out);
        if let Some(b) = &self.bias {
            out.iter_mut().zip(b).for_each(|(o, b)| *o += b);
        }
    }
}

struct Rwkv7 {
    /// Mixing vectors in the order r, w, k, v, a, g.
    lerp: [Vec<f32>; 6],
    w: Adapter,
    a: Adapter,
    /// Value residual adapter; layer 0 has none.
    v: Option<Adapter>,
    g: Adapter,
    k_k: Vec<f32>,
    k_a: Vec<f32>,
    r_k: Vec<f32>,
}

struct Rwkv6 {
    /// Mixing vectors: x (for the adapters), then w, k, v, r, g.
    lerp: [Vec<f32>; 6],
    /// Adapter of the five mixing vectors: `5 * rank` rows, then one `[hidden][rank]` matrix per vector.
    maa_w1: Mat,
    maa_w2: Vec<Mat>,
    decay: Vec<f32>,
    decay_w1: Mat,
    decay_w2: Mat,
    first: Vec<f32>,
    gate: Linear,
    channel_receptance: Linear,
    channel_lerp_r: Vec<f32>,
}

enum Version {
    V7(Box<Rwkv7>),
    V6(Box<Rwkv6>),
}

pub(super) struct Rwkv {
    version: Version,
    receptance: Linear,
    key: Linear,
    value: Linear,
    ln_x: Norm,
    ln2: Norm,
    channel_lerp_k: Vec<f32>,
    channel_key: Linear,
    channel_value: Linear,
}

impl Rwkv {
    /// `None` when the model is not an RWKV model. `linear(name, rows, cols)` loads a projection.
    pub(super) fn load(
        cfg: &ModelConfig,
        mapping: &MappedGguf,
        p: &str,
        layer: usize,
        linear: &dyn Fn(&str, usize, usize) -> Result<Linear, String>,
    ) -> Result<Option<Self>, String> {
        let Some(rc) = &cfg.rwkv else { return Ok(None) };
        let (h, ff) = (cfg.hidden, cfg.ff);
        if rc.head_size == 0 || !h.is_multiple_of(rc.head_size) {
            return Err(format!("RWKV hidden size {h} is not a multiple of the head size {}", rc.head_size));
        }
        let flat = |name: &str| vector(mapping, &format!("{p}.{name}.weight"), h);
        let ln2 = load_norm(mapping, &format!("{p}.attn_norm_2"), h)?.ok_or_else(|| format!("missing {p}.attn_norm_2.weight"))?;
        let ln_x = load_norm(mapping, &format!("{p}.time_mix_ln"), h)?.ok_or_else(|| format!("missing {p}.time_mix_ln.weight"))?;
        let lerp = |tags: [&str; 6]| -> Result<[Vec<f32>; 6], String> {
            let fused = format!("{p}.time_mix_lerp_fused.weight");
            if has_tensor(mapping, &fused) {
                // llama.cpp stacks the six vectors of RWKV-7 in the order r, w, k, v, a, g
                let data = vector(mapping, &fused, 6 * h)?;
                let mut parts = data.chunks_exact(h).map(<[f32]>::to_vec);
                let mut next = || parts.next().ok_or("fused lerp tensor");
                return Ok([next()?, next()?, next()?, next()?, next()?, next()?]);
            }
            let mut out: [Vec<f32>; 6] = Default::default();
            for (slot, tag) in out.iter_mut().zip(tags) {
                *slot = flat(&format!("time_mix_lerp_{tag}"))?;
            }
            Ok(out)
        };
        let version = if rc.v7 {
            Version::V7(Box::new(Rwkv7 {
                lerp: lerp(["r", "w", "k", "v", "a", "g"])?,
                w: Adapter::load(mapping, p, "w", h, true)?,
                a: Adapter::load(mapping, p, "a", h, true)?,
                v: if layer > 0 && has_tensor(mapping, &format!("{p}.time_mix_v1.weight")) { Some(Adapter::load(mapping, p, "v", h, true)?) } else { None },
                g: Adapter::load(mapping, p, "g", h, false)?,
                k_k: flat("time_mix_k_k")?,
                k_a: flat("time_mix_k_a")?,
                r_k: flat("time_mix_r_k")?,
            }))
        } else {
            let maa_w1 = Mat::load(mapping, &format!("{p}.time_mix_w1.weight"), h)?;
            let rank = maa_w1.rows / 5;
            let w2 = Tensor::load(mapping, &format!("{p}.time_mix_w2.weight"))?;
            // llama.cpp layout [rank, hidden, 5]: five matrices of `hidden` rows with `rank` columns
            if maa_w1.rows != 5 * rank || w2.shape != [rank, h, 5] {
                return Err(format!("invalid time-mix adapter in layer {layer}: w1 {} rows, w2 {:?}", maa_w1.rows, w2.shape));
            }
            let maa_w2 = w2.data.chunks_exact(h * rank).map(|d| Mat { rows: h, cols: rank, data: d.to_vec() }).collect();
            let heads = h / rc.head_size;
            Version::V6(Box::new(Rwkv6 {
                lerp: lerp(["x", "w", "k", "v", "r", "g"])?,
                maa_w1,
                maa_w2,
                decay: flat("time_mix_decay")?,
                decay_w1: Mat::load(mapping, &format!("{p}.time_mix_decay_w1.weight"), h)?,
                decay_w2: {
                    let rows = tensor_rows(mapping, &format!("{p}.time_mix_decay_w2.weight")).unwrap_or(0);
                    if rows != h {
                        return Err(format!("invalid {p}.time_mix_decay_w2.weight"));
                    }
                    let t = Tensor::load(mapping, &format!("{p}.time_mix_decay_w2.weight"))?;
                    Mat { rows: h, cols: t.shape[0], data: t.data }
                },
                first: vector(mapping, &format!("{p}.time_mix_first.weight"), heads * rc.head_size)?,
                gate: linear(&format!("{p}.time_mix_gate"), h, h)?,
                channel_receptance: linear(&format!("{p}.channel_mix_receptance"), h, h)?,
                channel_lerp_r: flat("channel_mix_lerp_r")?,
            }))
        };
        Ok(Some(Self {
            version,
            receptance: linear(&format!("{p}.time_mix_receptance"), h, h)?,
            key: linear(&format!("{p}.time_mix_key"), h, h)?,
            value: linear(&format!("{p}.time_mix_value"), h, h)?,
            ln_x,
            ln2,
            channel_lerp_k: flat("channel_mix_lerp_k")?,
            channel_key: linear(&format!("{p}.channel_mix_key"), ff, h)?,
            channel_value: linear(&format!("{p}.channel_mix_value"), h, ff)?,
        }))
    }

    /// The projections that a weight-quantisation or adapter pass visits.
    pub(super) fn linears_mut(&mut self) -> Vec<&mut Linear> {
        let mut all = vec![&mut self.receptance, &mut self.key, &mut self.value, &mut self.channel_key, &mut self.channel_value];
        if let Version::V6(v6) = &mut self.version {
            all.extend([&mut v6.gate, &mut v6.channel_receptance]);
        }
        all
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Normalises every group of `weight.len() / groups` values of every row (torch `GroupNorm` over one position per row).
fn group_norm(rows: &mut [f32], width: usize, groups: usize, norm: &Norm, eps: f32) {
    let size = width / groups;
    rows.par_chunks_mut(width).for_each(|row| {
        for (g, chunk) in row.chunks_mut(size).enumerate() {
            let mean = chunk.iter().sum::<f32>() / size as f32;
            let var = chunk.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / size as f32;
            let inv = (var + eps).sqrt().recip();
            for (j, v) in chunk.iter_mut().enumerate() {
                let at = g * size + j;
                *v = (*v - mean) * inv * norm.weight[at] + norm.bias.as_ref().map_or(0.0, |b| b[at]);
            }
        }
    });
}

impl RustDecoder {
    /// Walks the runs of consecutive tokens of each sequence like `for_each_run`, but for a state that the same pass already
    /// advanced (the token count is not checked and not bumped again).
    fn for_each_known_run(
        &self,
        layer: usize,
        tables: &[i64],
        lengths: &[i64],
        mut f: impl FnMut(usize, usize, &mut LayerState),
    ) -> Result<(), String> {
        let rows = lengths.len();
        let mut store = self.states.lock().map_err(|_| "recurrent state lock poisoned")?;
        let mut start = 0;
        while start < rows {
            let first = tables[start * self.stride];
            let mut end = start + 1;
            while end < rows && tables[end * self.stride] == first && lengths[end] == lengths[end - 1] + 1 {
                end += 1;
            }
            let state = store
                .map
                .get_mut(&first)
                .and_then(|s| s.layers[layer].as_mut())
                .ok_or("recurrent state of a sequence vanished during a layer")?;
            f(start, end, state);
            start = end;
        }
        Ok(())
    }

    /// One RWKV layer for a batch of tokens: `x` (rows x hidden) is the residual stream and is updated in place, `normed` is the
    /// layer norm of the time-mix block applied to `x`. `v_first` carries the values of layer 0 to the later layers (RWKV-7).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn rwkv_layer(
        &self,
        rw: &Rwkv,
        layer: &Layer,
        index: usize,
        x: &mut [f32],
        normed: &[f32],
        tables: &[i64],
        lengths: &[i64],
        v_first: &mut Vec<f32>,
    ) -> Result<(), String> {
        let rc = self.cfg.rwkv.as_ref().ok_or("missing RWKV configuration")?;
        let c = self.cfg.hidden;
        let rows = lengths.len();
        let (n, heads) = (rc.head_size, c / rc.head_size);
        let blank = || LayerState { len: 0, conv: vec![0.0; 2 * c], rec: vec![0.0; heads * n * n] };
        // token shift: the normed input of the previous token of the same sequence (zeros at the start)
        let mut shift = vec![0.0f32; rows * c];
        self.for_each_run(index, tables, lengths, &blank, |start, end, state| {
            for r in start..end {
                shift[r * c..(r + 1) * c].copy_from_slice(&state.conv[..c]);
                state.conv[..c].copy_from_slice(&normed[r * c..(r + 1) * c]);
            }
        })?;
        let mix = |lerp: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0f32; rows * c];
            out.par_chunks_mut(c).enumerate().for_each(|(r, o)| {
                for j in 0..c {
                    let h = normed[r * c + j];
                    o[j] = h + (shift[r * c + j] - h) * lerp[j];
                }
            });
            out
        };
        let mut y = vec![0.0f32; rows * c];
        // the per-row inputs of the recurrence (RWKV-7: r, decay, k, v, kk, a; RWKV-6: r, decay, k, v)
        match &rw.version {
            Version::V7(v7) => {
                let (xr, xw, xk, xv, xa, xg) = (mix(&v7.lerp[0]), mix(&v7.lerp[1]), mix(&v7.lerp[2]), mix(&v7.lerp[3]), mix(&v7.lerp[4]), mix(&v7.lerp[5]));
                let r = rw.receptance.forward(&xr)?;
                let mut k = rw.key.forward(&xk)?;
                let mut v = rw.value.forward(&xv)?;
                let mut decay = vec![0.0f32; rows * c];
                let mut a = vec![0.0f32; rows * c];
                let mut gate = vec![0.0f32; rows * c];
                let mut mid = Vec::new();
                if index == 0 || v7.v.is_none() {
                    *v_first = v.clone();
                } else if v_first.len() != v.len() {
                    return Err("the values of layer 0 are missing for the value residual".into());
                }
                let mut mix_v = vec![0.0f32; c];
                for row in 0..rows {
                    let s = row * c..(row + 1) * c;
                    v7.w.apply(&xw[s.clone()], f32::tanh, &mut mid, &mut decay[s.clone()]);
                    // decay = exp(-0.6065306597126334 * sigmoid(w0 + ...))
                    decay[s.clone()].iter_mut().for_each(|d| *d = (-0.606_530_66 * sigmoid(*d)).exp());
                    v7.a.apply(&xa[s.clone()], |t| t, &mut mid, &mut a[s.clone()]);
                    a[s.clone()].iter_mut().for_each(|t| *t = sigmoid(*t));
                    v7.g.apply(&xg[s.clone()], sigmoid, &mut mid, &mut gate[s.clone()]);
                    if let (Some(adapter), true) = (&v7.v, index > 0) {
                        adapter.apply(&xv[s.clone()], |t| t, &mut mid, &mut mix_v);
                        for (j, m) in mix_v.iter().enumerate() {
                            let at = row * c + j;
                            v[at] += (v_first[at] - v[at]) * sigmoid(*m);
                        }
                    }
                }
                // kk = normalize(k * k_k) per head; k = k * (1 + (a - 1) * k_a)
                let mut kk = vec![0.0f32; rows * c];
                for row in 0..rows {
                    for h in 0..heads {
                        let range = row * c + h * n..row * c + (h + 1) * n;
                        let mut norm = 0.0f32;
                        for j in range.clone() {
                            kk[j] = k[j] * v7.k_k[j % c];
                            norm += kk[j] * kk[j];
                        }
                        let inv = 1.0 / norm.sqrt().max(1e-12);
                        kk[range.clone()].iter_mut().for_each(|t| *t *= inv);
                        for j in range {
                            k[j] *= 1.0 + (a[j] - 1.0) * v7.k_a[j % c];
                        }
                    }
                }
                self.for_each_known_run(index, tables, lengths, |start, end, state| {
                    for row in start..end {
                        let at = row * c;
                        state.rec.par_chunks_mut(n * n).zip(y[at..at + c].par_chunks_mut(n)).enumerate().for_each(|(h, (s, out))| {
                            let o = at + h * n;
                            let (rh, wh, kh, vh, kkh, ah) = (&r[o..o + n], &decay[o..o + n], &k[o..o + n], &v[o..o + n], &kk[o..o + n], &a[o..o + n]);
                            for i in 0..n {
                                let srow = &mut s[i * n..(i + 1) * n];
                                // sa = S[i] . (-kk); S[i][j] = S[i][j] * decay[j] + sa * kk[j] * a[j] + v[i] * k[j]
                                let sa: f32 = -srow.iter().zip(kkh).map(|(s, k)| s * k).sum::<f32>();
                                let mut acc = 0.0f32;
                                for j in 0..n {
                                    srow[j] = srow[j] * wh[j] + sa * kkh[j] * ah[j] + vh[i] * kh[j];
                                    acc += srow[j] * rh[j];
                                }
                                out[i] = acc;
                            }
                        });
                    }
                })?;
                group_norm(&mut y, c, heads, &rw.ln_x, rc.group_eps);
                for row in 0..rows {
                    for h in 0..heads {
                        let o = row * c + h * n;
                        let bonus: f32 = (0..n).map(|j| r[o + j] * k[o + j] * v7.r_k[h * n + j]).sum();
                        for i in 0..n {
                            y[o + i] = (y[o + i] + bonus * v[o + i]) * gate[o + i];
                        }
                    }
                }
            }
            Version::V6(v6) => {
                // adapters of the five mixing vectors
                let rank = v6.maa_w1.rows / 5;
                let xx = mix(&v6.lerp[0]);
                let mut xs: [Vec<f32>; 5] = Default::default(); // w, k, v, r, g
                let mut low = vec![0.0f32; v6.maa_w1.rows];
                let mut delta = vec![0.0f32; c];
                for slot in &mut xs {
                    *slot = vec![0.0f32; rows * c];
                }
                for row in 0..rows {
                    v6.maa_w1.mul(&xx[row * c..(row + 1) * c], &mut low);
                    low.iter_mut().for_each(|t| *t = t.tanh());
                    for (j, slot) in xs.iter_mut().enumerate() {
                        v6.maa_w2[j].mul(&low[j * rank..(j + 1) * rank], &mut delta);
                        for (i, d) in delta.iter().enumerate() {
                            let at = row * c + i;
                            let h = normed[at];
                            slot[at] = h + (shift[at] - h) * (v6.lerp[j + 1][i] + d);
                        }
                    }
                }
                let [xw, xk, xv, xr, xg] = xs;
                let r = rw.receptance.forward(&xr)?;
                let k = rw.key.forward(&xk)?;
                let v = rw.value.forward(&xv)?;
                let mut gate = v6.gate.forward(&xg)?;
                gate.iter_mut().for_each(|g| *g = *g * sigmoid(*g));
                let mut decay = vec![0.0f32; rows * c];
                let mut mid = vec![0.0f32; v6.decay_w1.rows];
                for row in 0..rows {
                    let s = row * c..(row + 1) * c;
                    v6.decay_w1.mul(&xw[s.clone()], &mut mid);
                    mid.iter_mut().for_each(|t| *t = t.tanh());
                    v6.decay_w2.mul(&mid, &mut decay[s.clone()]);
                    for (d, base) in decay[s].iter_mut().zip(&v6.decay) {
                        *d = (-(*base + *d).exp()).exp();
                    }
                }
                self.for_each_known_run(index, tables, lengths, |start, end, state| {
                    for row in start..end {
                        let at = row * c;
                        state.rec.par_chunks_mut(n * n).zip(y[at..at + c].par_chunks_mut(n)).enumerate().for_each(|(h, (s, out))| {
                            let o = at + h * n;
                            let (rh, wh, kh, vh) = (&r[o..o + n], &decay[o..o + n], &k[o..o + n], &v[o..o + n]);
                            let u = &v6.first[h * n..(h + 1) * n];
                            out.iter_mut().for_each(|t| *t = 0.0);
                            for j in 0..n {
                                let srow = &mut s[j * n..(j + 1) * n];
                                for i in 0..n {
                                    let kv = kh[j] * vh[i];
                                    out[i] += rh[j] * (u[j] * kv + srow[i]);
                                    srow[i] = kv + wh[j] * srow[i];
                                }
                            }
                        });
                    }
                })?;
                group_norm(&mut y, c, heads, &rw.ln_x, rc.group_eps);
                y.iter_mut().zip(&gate).for_each(|(a, g)| *a *= g);
            }
        }
        let attn = layer.out.forward(&y)?;
        x.iter_mut().zip(&attn).for_each(|(x, a)| *x += a);
        // channel mix
        let mut h2 = x.to_vec();
        self.norm(&mut h2, &rw.ln2);
        let mut shift2 = vec![0.0f32; rows * c];
        self.for_each_known_run(index, tables, lengths, |start, end, state| {
            for r in start..end {
                shift2[r * c..(r + 1) * c].copy_from_slice(&state.conv[c..2 * c]);
                state.conv[c..2 * c].copy_from_slice(&h2[r * c..(r + 1) * c]);
            }
        })?;
        let mix2 = |lerp: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0f32; rows * c];
            out.par_chunks_mut(c).enumerate().for_each(|(r, o)| {
                for j in 0..c {
                    let h = h2[r * c + j];
                    o[j] = h + (shift2[r * c + j] - h) * lerp[j];
                }
            });
            out
        };
        let mut hidden = rw.channel_key.forward(&mix2(&rw.channel_lerp_k))?;
        hidden.par_iter_mut().for_each(|t| *t = t.max(0.0) * t.max(0.0));
        let mut out = rw.channel_value.forward(&hidden)?;
        if let Version::V6(v6) = &rw.version {
            let mut recv = v6.channel_receptance.forward(&mix2(&v6.channel_lerp_r))?;
            recv.iter_mut().for_each(|t| *t = sigmoid(*t));
            out.iter_mut().zip(&recv).for_each(|(o, r)| *o *= r);
        }
        x.iter_mut().zip(&out).for_each(|(x, o)| *x += o);
        if rc.rescale_every > 0 && (index + 1).is_multiple_of(rc.rescale_every) {
            x.iter_mut().for_each(|t| *t *= 0.5);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_matrix_times_vector() {
        let m = Mat { rows: 2, cols: 3, data: vec![1.0, 2.0, 3.0, -1.0, 0.0, 1.0] };
        let mut out = [0.0f32; 2];
        m.mul(&[1.0, 1.0, 2.0], &mut out);
        assert_eq!(out, [9.0, 1.0]);
    }

    #[test]
    fn adapter_applies_the_activation_between_the_matrices_and_adds_the_bias() {
        let adapter = Adapter {
            down: Mat { rows: 1, cols: 2, data: vec![1.0, 1.0] },
            up: Mat { rows: 2, cols: 1, data: vec![2.0, -1.0] },
            bias: Some(vec![0.5, 0.5]),
        };
        let (mut mid, mut out) = (Vec::new(), [0.0f32; 2]);
        adapter.apply(&[0.5, 0.5], f32::tanh, &mut mid, &mut out);
        let t = 1.0f32.tanh();
        assert!((out[0] - (2.0 * t + 0.5)).abs() < 1e-6 && (out[1] - (-t + 0.5)).abs() < 1e-6);
    }

    #[test]
    fn group_norm_normalises_every_group_of_a_row() {
        let norm = Norm { weight: vec![1.0; 4], bias: Some(vec![0.0; 4]) };
        let mut rows = vec![1.0, 3.0, 10.0, 14.0, 5.0, 5.0, 0.0, 2.0];
        group_norm(&mut rows, 4, 2, &norm, 1e-12);
        let expect = [-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, -1.0, 1.0];
        for (a, b) in rows.iter().zip(expect) {
            assert!((a - b).abs() < 1e-5, "{rows:?}");
        }
    }
}
