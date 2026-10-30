//! The recurrences of RWKV-6 and RWKV-7 for a run of consecutive tokens of one sequence (the reference of the device kernels).
//!
//! The device path of the decoder computes the projections, the low-rank adapters and the token-shift mixes with ordinary linear
//! operations and hands the per-token inputs to one operation per layer and run (`Stream::rwkv7_wkv` / `Stream::rwkv6_wkv`); this file
//! is what that operation computes, in `f64` so that it can serve as the reference of `selftest-gpu`. The host path of the decoder
//! (`rust_decoder/rwkv.rs`) has its own `f32` transcription of the same formulas.
//!
//! RWKV-7 (state `S[i][j]`, `i` = value channel, `j` = key channel, per head of `n` channels), per token:
//!
//! ```text
//! decay = exp(-0.6065306597126334 * sigmoid(w_raw))     a = sigmoid(a_raw)
//! v     = v + (v_first - v) * sigmoid(v_res)            (layers after the first that have a value adapter; otherwise v_first = v)
//! kk    = normalize(k * k_k)                            k = k * (1 + (a - 1) * k_a)
//! S[i][j] = S[i][j] * decay[j] + (-sum_j' S[i][j'] kk[j']) * kk[j] * a[j] + v[i] * k[j]
//! y[i]  = sum_j S[i][j] r[j]
//! out   = (group_norm(y) + sum_j r[j] k[j] r_k[j] * v[i]) * gate[i]
//! ```
//!
//! RWKV-6 (state `S[j][i]`, `j` = key channel, `i` = value channel), per token:
//!
//! ```text
//! w = exp(-exp(w_raw))        kv = k[j] v[i]
//! y[i] = sum_j r[j] (u[j] kv + S[j][i])               S[j][i] = kv + w[j] S[j][i]
//! out  = group_norm(y) * gate                          (gate = silu of the gate projection, computed by the caller)
//! ```
#![allow(clippy::needless_range_loop)]

/// Static shape and weights of one RWKV layer (the group norm over the heads of the time-mix output and the per-layer vectors).
pub(crate) struct Wkv<'a> {
    pub heads: usize,
    pub n: usize,
    pub eps: f32,
    pub norm_weight: &'a [f32],
    pub norm_bias: &'a [f32],
    /// RWKV-7: `k_k`, `k_a`, `r_k` (`hidden` each); RWKV-6: `first` (`u`) in `k_k` only.
    pub k_k: &'a [f32],
    pub k_a: &'a [f32],
    pub r_k: &'a [f32],
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// The per-row inputs of the RWKV-7 recurrence (`rows * hidden` values each).
pub(crate) struct Inputs7<'a> {
    pub r: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub w: &'a [f32],
    pub a: &'a [f32],
    pub v_res: Option<&'a [f32]>,
    pub gate: &'a [f32],
}

/// Advances `state` (`heads * n * n` values, `[value][key]`) over `rows` tokens and writes `out` (`rows * hidden`); `v_first` is read
/// when `inputs.v_res` is set and written (the values of the tokens) otherwise.
pub(crate) fn wkv7(p: &Wkv<'_>, state: &mut [f32], rows: usize, inputs: &Inputs7<'_>, v_first: &mut [f32], out: &mut [f32]) {
    let (n, c) = (p.n, p.heads * p.n);
    let mut s: Vec<f64> = state.iter().map(|&t| f64::from(t)).collect();
    let mut kk = vec![0.0f64; n];
    for t in 0..rows {
        for h in 0..p.heads {
            let o = t * c + h * n;
            let vec = |x: &[f32]| -> Vec<f64> { x[o..o + n].iter().map(|&t| f64::from(t)).collect() };
            let r = vec(inputs.r);
            let k_in = vec(inputs.k);
            let a: Vec<f64> = inputs.a[o..o + n].iter().map(|&t| sigmoid(f64::from(t))).collect();
            let decay: Vec<f64> = inputs.w[o..o + n].iter().map(|&t| (-0.606_530_659_712_633_4 * sigmoid(f64::from(t))).exp()).collect();
            let mut v = vec(inputs.v);
            match inputs.v_res {
                Some(res) => {
                    for i in 0..n {
                        let first = f64::from(v_first[o + i]);
                        v[i] += (first - v[i]) * sigmoid(f64::from(res[o + i]));
                    }
                }
                None => {
                    for i in 0..n {
                        v_first[o + i] = v[i] as f32;
                    }
                }
            }
            let mut norm = 0.0f64;
            for j in 0..n {
                kk[j] = k_in[j] * f64::from(p.k_k[h * n + j]);
                norm += kk[j] * kk[j];
            }
            let inv = 1.0 / norm.sqrt().max(1e-12);
            let k: Vec<f64> = (0..n).map(|j| k_in[j] * (1.0 + (a[j] - 1.0) * f64::from(p.k_a[h * n + j]))).collect();
            let mut y = vec![0.0f64; n];
            for i in 0..n {
                let row = &mut s[(h * n + i) * n..(h * n + i + 1) * n];
                let sa: f64 = -(0..n).map(|j| row[j] * kk[j] * inv).sum::<f64>();
                let mut acc = 0.0;
                for j in 0..n {
                    row[j] = row[j] * decay[j] + sa * kk[j] * inv * a[j] + v[i] * k[j];
                    acc += row[j] * r[j];
                }
                y[i] = acc;
            }
            let mean = y.iter().sum::<f64>() / n as f64;
            let var = y.iter().map(|t| (t - mean) * (t - mean)).sum::<f64>() / n as f64;
            let rstd = 1.0 / (var + f64::from(p.eps)).sqrt();
            let bonus: f64 = (0..n).map(|j| r[j] * k[j] * f64::from(p.r_k[h * n + j])).sum();
            for i in 0..n {
                let normed = (y[i] - mean) * rstd * f64::from(p.norm_weight[h * n + i]) + f64::from(p.norm_bias[h * n + i]);
                out[o + i] = ((normed + bonus * v[i]) * f64::from(inputs.gate[o + i])) as f32;
            }
        }
    }
    for (d, &t) in state.iter_mut().zip(&s) {
        *d = t as f32;
    }
}

/// The per-row inputs of the RWKV-6 recurrence (`rows * hidden` values each).
pub(crate) struct Inputs6<'a> {
    pub r: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub w: &'a [f32],
    pub gate: &'a [f32],
}

/// RWKV-6 counterpart of [`wkv7`]; `state` is `[key][value]` per head and `p.k_k` holds the bonus `u`.
pub(crate) fn wkv6(p: &Wkv<'_>, state: &mut [f32], rows: usize, inputs: &Inputs6<'_>, out: &mut [f32]) {
    let (n, c) = (p.n, p.heads * p.n);
    let mut s: Vec<f64> = state.iter().map(|&t| f64::from(t)).collect();
    for t in 0..rows {
        for h in 0..p.heads {
            let o = t * c + h * n;
            let mut y = vec![0.0f64; n];
            for j in 0..n {
                let (r, k) = (f64::from(inputs.r[o + j]), f64::from(inputs.k[o + j]));
                let w = (-f64::from(inputs.w[o + j]).exp()).exp();
                let u = f64::from(p.k_k[h * n + j]);
                let row = &mut s[(h * n + j) * n..(h * n + j + 1) * n];
                for i in 0..n {
                    let kv = k * f64::from(inputs.v[o + i]);
                    y[i] += r * (u * kv + row[i]);
                    row[i] = kv + w * row[i];
                }
            }
            let mean = y.iter().sum::<f64>() / n as f64;
            let var = y.iter().map(|t| (t - mean) * (t - mean)).sum::<f64>() / n as f64;
            let rstd = 1.0 / (var + f64::from(p.eps)).sqrt();
            for i in 0..n {
                let normed = (y[i] - mean) * rstd * f64::from(p.norm_weight[h * n + i]) + f64::from(p.norm_bias[h * n + i]);
                out[o + i] = (normed * f64::from(inputs.gate[o + i])) as f32;
            }
        }
    }
    for (d, &t) in state.iter_mut().zip(&s) {
        *d = t as f32;
    }
}

/// The token shift of one block over a run: `shift[t]` is the previous input of the sequence (`previous` for the first token of the run,
/// updated to the last input afterwards).
pub(crate) fn shift(hidden: usize, rows: usize, x: &[f32], previous: &mut [f32], out: &mut [f32]) {
    for t in 0..rows {
        let dst = &mut out[t * hidden..(t + 1) * hidden];
        if t == 0 {
            dst.copy_from_slice(&previous[..hidden]);
        } else {
            dst.copy_from_slice(&x[(t - 1) * hidden..t * hidden]);
        }
    }
    previous[..hidden].copy_from_slice(&x[(rows - 1) * hidden..rows * hidden]);
}

/// The element-wise activations of the RWKV blocks (`Stream::rwkv_act`).
pub(crate) fn activate(mode: i32, x: f32, gate: f32) -> f32 {
    match mode {
        0 => x.tanh(),
        1 => 1.0 / (1.0 + (-x).exp()),
        2 => x / (1.0 + (-x).exp()),
        3 => x.max(0.0) * x.max(0.0),
        5 => x * 0.5,
        _ => x / (1.0 + (-gate).exp()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shift_carries_the_previous_input_across_runs() {
        let mut previous = vec![7.0, 8.0];
        let mut out = vec![0.0; 6];
        shift(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &mut previous, &mut out);
        assert_eq!(out, [7.0, 8.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(previous, [5.0, 6.0]);
    }

    #[test]
    fn rwkv6_single_token_matches_the_closed_form() {
        // one head of two channels, zero state: y[i] = sum_j r[j] u[j] k[j] v[i]
        let (norm_weight, norm_bias) = ([1.0f32; 2], [0.0f32; 2]);
        let p = Wkv { heads: 1, n: 2, eps: 1e-5, norm_weight: &norm_weight, norm_bias: &norm_bias, k_k: &[0.5, 2.0], k_a: &[], r_k: &[] };
        let mut state = vec![0.0f32; 4];
        let inputs = Inputs6 { r: &[1.0, 2.0], k: &[3.0, 1.0], v: &[1.0, 3.0], w: &[0.0, 0.0], gate: &[1.0, 1.0] };
        let mut out = vec![0.0f32; 2];
        wkv6(&p, &mut state, 1, &inputs, &mut out);
        // y = [(1*0.5*3 + 2*2*1) * 1, (..) * 3] = [5.5, 16.5]; the group norm of two values gives -1, +1 (up to eps)
        assert!((out[0] + 1.0).abs() < 1e-3 && (out[1] - 1.0).abs() < 1e-3, "{out:?}");
        assert!((state[0] - 3.0).abs() < 1e-6 && (state[3] - 3.0).abs() < 1e-6, "{state:?}");
    }
}
