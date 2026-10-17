//! Kimi Delta Attention (KDA) for a run of consecutive tokens of one sequence: the gated delta rule of `cpu_delta` with a
//! decay per key channel instead of one per head (Kimi-Linear, Ling 3.0, Kimi K3 and the linear layers of GLM-5.3).
//!
//! For every head with state `S` (`k_dim x v_dim`), after the short convolution and the L2 normalisation of `q` and `k`
//! (`x / sqrt(sum x^2 + 1e-6)`, `q` additionally scaled by `1/sqrt(k_dim)`):
//!
//! ```text
//! S[i][:] *= exp(g[i])                      per key channel i, g <= 0
//! delta   = beta * (v - S^T k)
//! S      += k (x) delta
//! o       = S^T q
//! ```
//!
//! followed by `rms_norm(o) * weight * sigmoid(out_gate)` per head. The log decay is
//! `g = -exp(A_log) * softplus(f + dt_bias)` (Kimi) or `lower_bound * sigmoid(exp(A_log) * (f + dt_bias))` (GLM-5, "safe
//! gate"). Reference: `fla.ops.kda.naive_recurrent_kda`, `transformers` `KimiLinearDeltaAttention`; the vectors in
//! `testdata/kda_reference.json` come from fla (see `scripts/gen_kda_reference.py`).
//!
//! The recurrence is evaluated token by token. The chunked form of the kernels is an exact reformulation of it (the reference
//! generator checks the two against each other), and on the CPU the sequential form costs about `4 * k_dim * v_dim` flops per
//! head and token, small next to the projections of the layer.
#![allow(clippy::needless_range_loop)]
use rayon::prelude::*;

/// Static shape and weights of a KDA layer.
pub(crate) struct Layer<'a> {
    pub heads: usize,
    pub k_dim: usize,
    pub v_dim: usize,
    pub conv_kernel: usize,
    /// Epsilon of the output RMS norm.
    pub eps: f32,
    /// GLM-5 style bounded gate (`None`: Kimi's softplus gate).
    pub lower_bound: Option<f32>,
    /// `conv_dim x kernel`, rows ordered q, k, v.
    pub conv: &'a [f32],
    /// `heads * k_dim`
    pub dt_bias: &'a [f32],
    /// `exp(A_log)` per head.
    pub rate: &'a [f32],
    /// Output norm weight, `v_dim`.
    pub norm: &'a [f32],
}

impl Layer<'_> {
    pub(crate) fn conv_dim(&self) -> usize {
        2 * self.heads * self.k_dim + self.heads * self.v_dim
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Causal depthwise convolution with SiLU over `n` tokens; `history` (`conv_dim x (kernel - 1)`, oldest first) is updated.
fn convolve(l: &Layer<'_>, history: &mut [f32], qkv: &[f32], n: usize) -> Vec<f32> {
    let (dim, ksz) = (l.conv_dim(), l.conv_kernel);
    assert!(ksz >= 2, "the KDA convolution needs a kernel of at least two taps");
    let hist = ksz - 1;
    let mut mixed = vec![0.0f32; n * dim];
    let channel = |c: usize, column: &mut [f32], past: &mut [f32]| {
        let w = &l.conv[c * ksz..(c + 1) * ksz];
        // inputs of the channel: history followed by the new tokens
        let input = |i: usize| if i < hist { past[i] } else { qkv[(i - hist) * dim + c] };
        for t in 0..n {
            let mut acc = 0.0f32;
            for (k, wk) in w.iter().enumerate() {
                acc += wk * input(t + k);
            }
            column[t] = silu(acc);
        }
        let fresh: Vec<f32> = (0..hist).map(|k| input(n + k)).collect();
        past.copy_from_slice(&fresh);
    };
    // channels are independent: compute columns in parallel, then transpose into rows
    let mut columns = vec![0.0f32; dim * n];
    columns
        .par_chunks_mut(n.max(1))
        .zip(history.par_chunks_mut(hist))
        .enumerate()
        .for_each(|(c, (column, past))| channel(c, column, past));
    for c in 0..dim {
        for t in 0..n {
            mixed[t * dim + c] = columns[c * n + t];
        }
    }
    mixed
}

/// `q`, `k`, `v`, log decay and write strength of a run.
type Recurrent<'a> = (&'a [f32], &'a [f32], &'a [f32], &'a [f32], &'a [f32]);

/// The recurrence on already convolved inputs. `q`, `k` are `n x (heads * k_dim)` and `v` is `n x (heads * v_dim)` (all before the
/// L2 normalisation, taken from the same rows as the convolution output), `g` the log decay `n x (heads * k_dim)`, `beta` the write
/// strength `n x heads`. `state` (`heads x k_dim x v_dim`) is updated in place; the pre-norm head outputs go to `out`.
pub(crate) fn recurrence(
    (heads, k_dim, v_dim): (usize, usize, usize),
    (q, k, v, g, beta): Recurrent<'_>,
    state: &mut [f32],
    out: &mut [f32],
    n: usize,
) {
    let scale = (k_dim as f32).powf(-0.5);
    let unit = |x: &[f32]| (x.iter().map(|a| a * a).sum::<f32>() + 1e-6).sqrt().recip();
    let heads_out: Vec<Vec<f32>> = state
        .par_chunks_mut(k_dim * v_dim)
        .enumerate()
        .map(|(h, s)| {
            let mut o = vec![0.0f32; n * v_dim];
            let mut qn = vec![0.0f32; k_dim];
            let mut kn = vec![0.0f32; k_dim];
            let mut memory = vec![0.0f32; v_dim];
            for t in 0..n {
                let qrow = &q[t * heads * k_dim + h * k_dim..][..k_dim];
                let krow = &k[t * heads * k_dim + h * k_dim..][..k_dim];
                let (nq, nk) = (unit(qrow) * scale, unit(krow));
                for i in 0..k_dim {
                    qn[i] = qrow[i] * nq;
                    kn[i] = krow[i] * nk;
                }
                let gate = &g[t * heads * k_dim + h * k_dim..][..k_dim];
                let value = &v[t * heads * v_dim + h * v_dim..][..v_dim];
                let strength = beta[t * heads + h];
                memory.fill(0.0);
                for i in 0..k_dim {
                    let decay = gate[i].exp();
                    let row = &mut s[i * v_dim..(i + 1) * v_dim];
                    for (m, x) in memory.iter_mut().zip(row.iter_mut()) {
                        *x *= decay;
                        *m += *x * kn[i];
                    }
                }
                let delta: Vec<f32> = value.iter().zip(&memory).map(|(v, m)| (v - m) * strength).collect();
                let result = &mut o[t * v_dim..(t + 1) * v_dim];
                for i in 0..k_dim {
                    let row = &mut s[i * v_dim..(i + 1) * v_dim];
                    for ((x, d), r) in row.iter_mut().zip(&delta).zip(result.iter_mut()) {
                        *x += kn[i] * d;
                        *r += *x * qn[i];
                    }
                }
            }
            o
        })
        .collect();
    for (h, o) in heads_out.iter().enumerate() {
        for t in 0..n {
            out[t * heads * v_dim + h * v_dim..][..v_dim].copy_from_slice(&o[t * v_dim..(t + 1) * v_dim]);
        }
    }
}

/// Runs `n` consecutive tokens. `qkv` is `n x conv_dim` (the q, k and v projections side by side), `forget` `n x (heads * k_dim)`
/// (the forget-gate projection before `dt_bias`), `beta` `n x heads` (before the sigmoid), `out_gate` `n x (heads * v_dim)`
/// (the output-gate projection). `out` (`n x (heads * v_dim)`) receives the normalised, gated head outputs.
pub(crate) fn run(
    l: &Layer<'_>,
    history: &mut [f32],
    rec: &mut [f32],
    (qkv, forget, beta, out_gate): (&[f32], &[f32], &[f32], &[f32]),
    out: &mut [f32],
    n: usize,
) {
    let (heads, hk, hv) = (l.heads, l.k_dim, l.v_dim);
    let key_dim = heads * hk;
    let dim = l.conv_dim();
    let mixed = convolve(l, history, qkv, n);
    let split = |width: usize, offset: usize| -> Vec<f32> {
        let mut v = Vec::with_capacity(n * width);
        for t in 0..n {
            v.extend_from_slice(&mixed[t * dim + offset..t * dim + offset + width]);
        }
        v
    };
    let (q, k, v) = (split(key_dim, 0), split(key_dim, key_dim), split(heads * hv, 2 * key_dim));
    let mut g = vec![0.0f32; n * key_dim];
    for t in 0..n {
        for h in 0..heads {
            for i in 0..hk {
                let at = t * key_dim + h * hk + i;
                let raw = forget[at] + l.dt_bias[h * hk + i];
                g[at] = match l.lower_bound {
                    Some(bound) => bound * sigmoid(l.rate[h] * raw),
                    None => -l.rate[h] * if raw > 20.0 { raw } else { raw.exp().ln_1p() },
                };
            }
        }
    }
    let strength: Vec<f32> = beta.iter().map(|b| sigmoid(*b)).collect();
    recurrence((heads, hk, hv), (&q, &k, &v, &g, &strength), rec, out, n);
    // rms_norm(o) * weight * sigmoid(gate), per head
    out.par_chunks_mut(heads * hv).zip(out_gate.par_chunks(heads * hv)).for_each(|(row, gate)| {
        for h in 0..heads {
            let o = &mut row[h * hv..(h + 1) * hv];
            let inv = (o.iter().map(|x| x * x).sum::<f32>() / hv as f32 + l.eps).sqrt().recip();
            for (c, x) in o.iter_mut().enumerate() {
                *x *= inv * l.norm[c] * sigmoid(gate[h * hv + c]);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn floats(doc: &Value, key: &str) -> Vec<f32> {
        doc[key].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect()
    }

    fn close(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        let (mut worst, mut at) = (0f32, 0);
        for (i, (a, b)) in got.iter().zip(want).enumerate() {
            let d = (a - b).abs();
            if d > worst {
                (worst, at) = (d, i);
            }
        }
        assert!(worst < 2e-5, "{what}: largest difference {worst} at element {at} (got {}, want {})", got[at], want[at]);
    }

    #[test]
    fn recurrence_matches_the_fla_reference_vectors() {
        let doc: Value = serde_json::from_str(include_str!("../testdata/kda_reference.json")).unwrap();
        let (n, h, kd, vd) = (
            doc["tokens"].as_u64().unwrap() as usize,
            doc["heads"].as_u64().unwrap() as usize,
            doc["k_dim"].as_u64().unwrap() as usize,
            doc["v_dim"].as_u64().unwrap() as usize,
        );
        let (q, k, v, g, beta) = (floats(&doc, "q"), floats(&doc, "k"), floats(&doc, "v"), floats(&doc, "g"), floats(&doc, "beta"));
        // the reference holds state as [head][k][v]
        let mut state = vec![0.0f32; h * kd * vd];
        let mut out = vec![0.0f32; n * h * vd];
        recurrence((h, kd, vd), (&q, &k, &v, &g, &beta), &mut state, &mut out, n);
        close(&out, &floats(&doc, "out"), "outputs");
        close(&state, &floats(&doc, "state"), "final state");

        // the same tokens as two runs, the second continuing from the first one's state
        let split = doc["split"].as_u64().unwrap() as usize;
        let mut state = vec![0.0f32; h * kd * vd];
        let mut head = vec![0.0f32; split * h * vd];
        recurrence((h, kd, vd), (&q[..split * h * kd], &k[..split * h * kd], &v[..split * h * vd], &g[..split * h * kd], &beta[..split * h]), &mut state, &mut head, split);
        close(&state, &floats(&doc, "state_at_split"), "state at the split");
        let mut tail = vec![0.0f32; (n - split) * h * vd];
        recurrence((h, kd, vd), (&q[split * h * kd..], &k[split * h * kd..], &v[split * h * vd..], &g[split * h * kd..], &beta[split * h..]), &mut state, &mut tail, n - split);
        let want = floats(&doc, "out");
        close(&head, &want[..split * h * vd], "first run");
        close(&tail, &want[split * h * vd..], "second run");
        close(&state, &floats(&doc, "state"), "state after the second run");
    }

    #[test]
    fn convolution_history_carries_over_between_runs() {
        let (heads, hk, hv, ksz) = (2usize, 4usize, 4usize, 3usize);
        let dim = 2 * heads * hk + heads * hv;
        let mut seed = 99u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1 << 24) as f32 - 0.5
        };
        let conv: Vec<f32> = (0..dim * ksz).map(|_| rnd()).collect();
        let layer = Layer { heads, k_dim: hk, v_dim: hv, conv_kernel: ksz, eps: 1e-6, lower_bound: None, conv: &conv, dt_bias: &[], rate: &[], norm: &[] };
        let n = 7;
        let qkv: Vec<f32> = (0..n * dim).map(|_| rnd()).collect();
        let mut whole_history = vec![0.0f32; dim * (ksz - 1)];
        let whole = convolve(&layer, &mut whole_history, &qkv, n);
        let mut history = vec![0.0f32; dim * (ksz - 1)];
        let mut parts = convolve(&layer, &mut history, &qkv[..3 * dim], 3);
        parts.extend(convolve(&layer, &mut history, &qkv[3 * dim..], n - 3));
        close(&parts, &whole, "split convolution");
        close(&history, &whole_history, "history");
    }
}
