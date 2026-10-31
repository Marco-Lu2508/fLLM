//! Chunk-parallel form of the linear recurrences (the `f64` reference of the device kernels, proven equal to the sequential recurrence).
//!
//! Every recurrent layer of the decoder (Gated DeltaNet, Kimi Delta Attention, RWKV-7, RWKV-6, Mamba-2) is one instance of
//!
//! ```text
//! P_t = (diag(w_t) - b_t a_t^T) P_{t-1} + k_t v_t^T          P: hk x hv state of one head, w_t = exp(lw_t), lw_t <= 0
//! o_t = P_t^T q_t + c_t v_t                                  o_t: hv values
//! ```
//!
//! | layer                | w_t            | a_t        | b_t        | k_t (`kp`)   | c_t                    |
//! |----------------------|----------------|------------|------------|--------------|------------------------|
//! | Gated DeltaNet       | `g_t` (scalar) | `g_t k_t`  | `beta_t k_t` | `beta_t k_t` | 0                      |
//! | KDA                  | `g_t` (vector) | `g_t * k_t`| `beta_t k_t` | `beta_t k_t` | 0                      |
//! | RWKV-7               | `w_t` (vector) | `kk_t`     | `kk_t * a_t` | `k_t`        | 0                      |
//! | RWKV-6               | `w_t` (vector) | none       | none       | `k_t`        | `sum_i r_i u_i k_i`, readout before the token |
//! | Mamba-2              | `exp(dt A)`    | none       | none       | `dt B_t`     | `D`                    |
//!
//! The sequential form walks the tokens. The chunked form walks chunks of `C` tokens: with the cumulative log decay `G_t`
//! inside a chunk (`Gp_t = G_t - lw_t`, the one before the token), the rank-one terms are solved for the whole chunk at once
//! (the WY / UT representation: `(I + B) Y = X`, a triangular system whose entries need no state), after which the chunk is
//! four small matrix products against the incoming state `P_0` (`Qe`, `O0`, `Wr`, `R0`, `Kd`, `Bd` below) and the state
//! advances by one `hk x hv` update per chunk instead of per token. Pairwise decay factors are always `exp(G_t - G_s)` with
//! `t >= s` (never larger than 1), so nothing overflows however strong the decay is.
//!
//! Per chunk with tokens `0..L` (`B` strictly lower triangular, `Q` lower triangular including the diagonal):
//!
//! ```text
//! Bm[t][s] = sum_i a[t][i] b[s][i]  exp(Gp[t][i] - G[s][i])      s <  t          (rank-one terms among themselves)
//! Xi[t][s] = sum_i a[t][i] kp[s][i] exp(Gp[t][i] - G[s][i])      s <  t
//! Qk[t][s] = sum_i q[t][i] kp[s][i] exp(G[t][i]  - G[s][i])      s <= t
//! Qb[t][s] = sum_i q[t][i] b[s][i]  exp(G[t][i]  - G[s][i])      s <= t
//! [R0 | Wr] = (I + Bm)^-1 [Xi V | a * exp(Gp)]                   forward substitution, one column per thread on a device
//! Qe = q * exp(G) - Qb Wr        O0 = Qk V + c V - Qb R0         (no state needed)
//! Kd = kp * exp(Gend - G)        Bd = b * exp(Gend - G)
//! ```
//!
//! and then, with the incoming state `P_0`:
//!
//! ```text
//! R = R0 + Wr P_0       O = O0 + Qe P_0       P_end = diag(exp(Gend)) P_0 + Kd^T V - Bd^T R
//! ```
//!
//! Without rank-one terms (`a`, `b` absent) `R`, `Wr`, `R0` and the system vanish.
#![allow(clippy::needless_range_loop)]

/// Inputs of one head over `n` tokens. Row-major: `q`, `kp`, `a`, `b` are `n x hk`, `v` is `n x hv`, `lw` is `n x hk` when
/// `lw_vec` and `n` values otherwise, `c` is `n` values.
pub(crate) struct Scan<'a> {
    pub hk: usize,
    pub hv: usize,
    pub n: usize,
    pub q: &'a [f64],
    pub kp: &'a [f64],
    pub v: &'a [f64],
    pub a: Option<&'a [f64]>,
    pub b: Option<&'a [f64]>,
    pub lw: &'a [f64],
    pub lw_vec: bool,
    pub c: Option<&'a [f64]>,
    /// The readout sees the state before this token (RWKV-6): `o_t = P_{t-1}^T q_t + c_t v_t`. Only without rank-one terms.
    pub q_prev: bool,
}

impl Scan<'_> {
    fn lw_at(&self, t: usize, i: usize) -> f64 {
        if self.lw_vec { self.lw[t * self.hk + i] } else { self.lw[t] }
    }
}

/// The token-by-token recurrence. `state` is `hk x hv` row-major (updated in place), the result `n x hv`.
pub(crate) fn sequential(s: &Scan<'_>, state: &mut [f64]) -> Vec<f64> {
    let (hk, hv) = (s.hk, s.hv);
    let mut out = vec![0.0; s.n * hv];
    for t in 0..s.n {
        let mut early = vec![0.0; hv];
        if s.q_prev {
            for j in 0..hv {
                early[j] = (0..hk).map(|i| state[i * hv + j] * s.q[t * hk + i]).sum();
            }
        }
        // a^T P (before the decay), 1 x hv
        let mut ap = vec![0.0; hv];
        if let Some(a) = s.a {
            for i in 0..hk {
                for j in 0..hv {
                    ap[j] += a[t * hk + i] * state[i * hv + j];
                }
            }
        }
        for i in 0..hk {
            let w = s.lw_at(t, i).exp();
            for j in 0..hv {
                let mut p = state[i * hv + j] * w;
                if let Some(b) = s.b {
                    p -= b[t * hk + i] * ap[j];
                }
                p += s.kp[t * hk + i] * s.v[t * hv + j];
                state[i * hv + j] = p;
            }
        }
        for j in 0..hv {
            let mut o = s.c.map_or(0.0, |c| c[t]) * s.v[t * hv + j];
            if s.q_prev {
                o += early[j];
            } else {
                for i in 0..hk {
                    o += state[i * hv + j] * s.q[t * hk + i];
                }
            }
            out[t * hv + j] = o;
        }
    }
    out
}

/// The chunked form with chunks of `chunk` tokens (the last one may be shorter).
pub(crate) fn chunked(s: &Scan<'_>, state: &mut [f64], chunk: usize) -> Vec<f64> {
    let (hk, hv) = (s.hk, s.hv);
    let rank_one = s.a.is_some() && s.b.is_some();
    assert!(!(rank_one && s.q_prev), "the early readout has no rank-one form");
    let early = usize::from(s.q_prev);
    let mut out = vec![0.0; s.n * hv];
    let mut t0 = 0;
    while t0 < s.n {
        let l = chunk.min(s.n - t0);
        // cumulative log decay inside the chunk (inclusive and exclusive)
        let mut g = vec![0.0; l * hk];
        let mut gp = vec![0.0; l * hk];
        for i in 0..hk {
            let mut acc = 0.0;
            for t in 0..l {
                gp[t * hk + i] = acc;
                acc += s.lw_at(t0 + t, i);
                g[t * hk + i] = acc;
            }
        }
        let row = |m: &[f64], t: usize, i: usize| m[(t0 + t) * hk + i];
        let vrow = |t: usize, j: usize| s.v[(t0 + t) * hv + j];
        let (mut r0, mut wr) = (vec![0.0; l * hv], vec![0.0; l * hk]);
        let (mut qk, mut qb) = (vec![0.0; l * l], vec![0.0; l * l]);
        // the readout decay reaches the token itself (`G_t`) or stops before it (`Gp_t`, RWKV-6)
        let gq = if s.q_prev { &gp } else { &g };
        for t in 0..l {
            for u in 0..(t + 1 - early) {
                let (mut x, mut y) = (0.0, 0.0);
                for i in 0..hk {
                    let e = (gq[t * hk + i] - g[u * hk + i]).exp();
                    x += row(s.q, t, i) * row(s.kp, u, i) * e;
                    if let Some(b) = s.b {
                        y += row(s.q, t, i) * row(b, u, i) * e;
                    }
                }
                qk[t * l + u] = x;
                qb[t * l + u] = y;
            }
        }
        if rank_one {
            let (a, b) = (s.a.unwrap(), s.b.unwrap());
            let (mut bm, mut xi) = (vec![0.0; l * l], vec![0.0; l * l]);
            for t in 0..l {
                for u in 0..t {
                    let (mut x, mut y) = (0.0, 0.0);
                    for i in 0..hk {
                        let e = (gp[t * hk + i] - g[u * hk + i]).exp();
                        x += row(a, t, i) * row(b, u, i) * e;
                        y += row(a, t, i) * row(s.kp, u, i) * e;
                    }
                    bm[t * l + u] = x;
                    xi[t * l + u] = y;
                }
            }
            // forward substitution (I + Bm) Y = X, X = [Xi V | a exp(Gp)]
            for t in 0..l {
                for j in 0..hv {
                    let mut x = 0.0;
                    for u in 0..t {
                        x += xi[t * l + u] * vrow(u, j);
                    }
                    for u in 0..t {
                        x -= bm[t * l + u] * r0[u * hv + j];
                    }
                    r0[t * hv + j] = x;
                }
                for i in 0..hk {
                    let mut x = row(a, t, i) * gp[t * hk + i].exp();
                    for u in 0..t {
                        x -= bm[t * l + u] * wr[u * hk + i];
                    }
                    wr[t * hk + i] = x;
                }
            }
        }
        // outputs that need no state, and the state-dependent factors
        let (mut qe, mut o0) = (vec![0.0; l * hk], vec![0.0; l * hv]);
        for t in 0..l {
            for i in 0..hk {
                let mut x = row(s.q, t, i) * gq[t * hk + i].exp();
                for u in 0..=t {
                    x -= qb[t * l + u] * wr[u * hk + i];
                }
                qe[t * hk + i] = x;
            }
            for j in 0..hv {
                let mut x = s.c.map_or(0.0, |c| c[t0 + t]) * vrow(t, j);
                for u in 0..(t + 1 - early) {
                    x += qk[t * l + u] * vrow(u, j) - qb[t * l + u] * r0[u * hv + j];
                }
                o0[t * hv + j] = x;
            }
        }
        // the incoming state
        let mut rm = r0.clone();
        for t in 0..l {
            for j in 0..hv {
                let (mut x, mut y) = (o0[t * hv + j], rm[t * hv + j]);
                for i in 0..hk {
                    x += qe[t * hk + i] * state[i * hv + j];
                    y += wr[t * hk + i] * state[i * hv + j];
                }
                out[(t0 + t) * hv + j] = x;
                rm[t * hv + j] = y;
            }
        }
        for i in 0..hk {
            let gend = g[(l - 1) * hk + i];
            for j in 0..hv {
                let mut x = state[i * hv + j] * gend.exp();
                for u in 0..l {
                    let e = (gend - g[u * hk + i]).exp();
                    x += row(s.kp, u, i) * e * vrow(u, j);
                    if let Some(b) = s.b {
                        x -= row(b, u, i) * e * rm[u * hv + j];
                    }
                }
                state[i * hv + j] = x;
            }
        }
        t0 += l;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.next()
        }
        fn vec(&mut self, n: usize, lo: f64, hi: f64) -> Vec<f64> {
            (0..n).map(|_| self.uniform(lo, hi)).collect()
        }
        /// Unit vectors per row.
        fn unit_rows(&mut self, rows: usize, width: usize) -> Vec<f64> {
            let mut m = self.vec(rows * width, -1.0, 1.0);
            for r in m.chunks_mut(width) {
                let norm = r.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
                r.iter_mut().for_each(|x| *x /= norm);
            }
            m
        }
    }

    fn close(a: &[f64], b: &[f64], what: &str, case: &str) {
        let scale = b.iter().fold(1e-3f64, |m, x| m.max(x.abs()));
        let worst = a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
        assert!(worst <= 1e-9 * scale, "{what} differs by {worst:e} (scale {scale:e}) in {case}");
    }

    /// Runs `s` both ways with and without an initial state and compares outputs and final states.
    fn check(s: &Scan<'_>, rng: &mut Rng, case: &str) {
        for chunk in [16usize, 32, 64] {
            for with_state in [false, true] {
                let init = if with_state { rng.vec(s.hk * s.hv, -1.0, 1.0) } else { vec![0.0; s.hk * s.hv] };
                let (mut p1, mut p2) = (init.clone(), init);
                let want = sequential(s, &mut p1);
                let got = chunked(s, &mut p2, chunk);
                let case = format!("{case}, chunk {chunk}, state {with_state}");
                close(&got, &want, "output", &case);
                close(&p2, &p1, "final state", &case);
            }
        }
    }

    const LENGTHS: [usize; 9] = [1, 7, 16, 31, 33, 64, 65, 100, 193];

    #[test]
    fn gated_delta_rule_chunks_equal_the_recurrence() {
        let mut rng = Rng(1);
        for &(hk, hv) in &[(8usize, 8usize), (16, 24), (32, 16)] {
            for &n in &LENGTHS {
                let k = rng.unit_rows(n, hk);
                let q: Vec<f64> = rng.unit_rows(n, hk).iter().map(|x| x / (hk as f64).sqrt()).collect();
                let v = rng.vec(n * hv, -1.0, 1.0);
                let lw: Vec<f64> = (0..n).map(|_| -rng.uniform(0.0, 1.5)).collect();
                let beta = rng.vec(n, 0.05, 1.0);
                let a: Vec<f64> = (0..n * hk).map(|x| k[x] * lw[x / hk].exp()).collect();
                let b: Vec<f64> = (0..n * hk).map(|x| k[x] * beta[x / hk]).collect();
                let s = Scan { hk, hv, n, q: &q, kp: &b, v: &v, a: Some(&a), b: Some(&b), lw: &lw, lw_vec: false, c: None, q_prev: false };
                // the delta rule written directly: S *= g; S += k (x) beta (v - S^T k); o = S^T q
                let direct = |state: &mut [f64]| {
                    let mut out = vec![0.0; n * hv];
                    for t in 0..n {
                        let g = lw[t].exp();
                        for j in 0..hv {
                            let mut mem = 0.0;
                            for i in 0..hk {
                                state[i * hv + j] *= g;
                                mem += state[i * hv + j] * k[t * hk + i];
                            }
                            let d = beta[t] * (v[t * hv + j] - mem);
                            for i in 0..hk {
                                state[i * hv + j] += k[t * hk + i] * d;
                                out[t * hv + j] += state[i * hv + j] * q[t * hk + i];
                            }
                        }
                    }
                    out
                };
                let init = rng.vec(hk * hv, -1.0, 1.0);
                let (mut p1, mut p2) = (init.clone(), init);
                let want = direct(&mut p1);
                let got = sequential(&s, &mut p2);
                close(&got, &want, "mapping output", &format!("delta {hk}x{hv} n {n}"));
                close(&p2, &p1, "mapping state", &format!("delta {hk}x{hv} n {n}"));
                check(&s, &mut rng, &format!("delta {hk}x{hv} n {n}"));
            }
        }
    }

    #[test]
    fn kimi_delta_attention_chunks_equal_the_recurrence() {
        let mut rng = Rng(2);
        for &(hk, hv) in &[(8usize, 8usize), (16, 16), (32, 32)] {
            for &n in &LENGTHS {
                let k = rng.unit_rows(n, hk);
                let q = rng.unit_rows(n, hk);
                let v = rng.vec(n * hv, -1.0, 1.0);
                // includes channels that forget almost everything at once
                let lw: Vec<f64> = (0..n * hk).map(|x| if x % 7 == 3 { -rng.uniform(5.0, 40.0) } else { -rng.uniform(0.0, 1.0) }).collect();
                let beta = rng.vec(n, 0.05, 1.0);
                let a: Vec<f64> = (0..n * hk).map(|x| k[x] * lw[x].exp()).collect();
                let b: Vec<f64> = (0..n * hk).map(|x| k[x] * beta[x / hk]).collect();
                let s = Scan { hk, hv, n, q: &q, kp: &b, v: &v, a: Some(&a), b: Some(&b), lw: &lw, lw_vec: true, c: None, q_prev: false };
                let direct = |state: &mut [f64]| {
                    let mut out = vec![0.0; n * hv];
                    for t in 0..n {
                        for j in 0..hv {
                            let mut mem = 0.0;
                            for i in 0..hk {
                                state[i * hv + j] *= lw[t * hk + i].exp();
                                mem += state[i * hv + j] * k[t * hk + i];
                            }
                            let d = beta[t] * (v[t * hv + j] - mem);
                            for i in 0..hk {
                                state[i * hv + j] += k[t * hk + i] * d;
                                out[t * hv + j] += state[i * hv + j] * q[t * hk + i];
                            }
                        }
                    }
                    out
                };
                let init = rng.vec(hk * hv, -1.0, 1.0);
                let (mut p1, mut p2) = (init.clone(), init);
                let want = direct(&mut p1);
                let got = sequential(&s, &mut p2);
                close(&got, &want, "mapping output", &format!("kda {hk}x{hv} n {n}"));
                close(&p2, &p1, "mapping state", &format!("kda {hk}x{hv} n {n}"));
                check(&s, &mut rng, &format!("kda {hk}x{hv} n {n}"));
            }
        }
    }

    #[test]
    fn rwkv7_chunks_equal_the_recurrence() {
        let mut rng = Rng(3);
        for &nn in &[8usize, 16, 32] {
            for &n in &LENGTHS {
                let r = rng.vec(n * nn, -1.0, 1.0);
                let k = rng.vec(n * nn, -1.0, 1.0);
                let v = rng.vec(n * nn, -1.0, 1.0);
                let kk = rng.unit_rows(n, nn);
                let gate = rng.vec(n * nn, 0.0, 1.0);
                let decay: Vec<f64> = rng.vec(n * nn, 0.545, 1.0);
                let lw: Vec<f64> = decay.iter().map(|d| d.ln()).collect();
                let b: Vec<f64> = (0..n * nn).map(|x| kk[x] * gate[x]).collect();
                let s = Scan { hk: nn, hv: nn, n, q: &r, kp: &k, v: &v, a: Some(&kk), b: Some(&b), lw: &lw, lw_vec: true, c: None, q_prev: false };
                // the kernel's form: state[i][j] with i value channel, j key channel
                let direct = |state: &mut [f64]| {
                    // state arrives as [key][value]; transpose to the kernel's [value][key]
                    let mut sv = vec![0.0; nn * nn];
                    for j in 0..nn {
                        for i in 0..nn {
                            sv[i * nn + j] = state[j * nn + i];
                        }
                    }
                    let mut out = vec![0.0; n * nn];
                    for t in 0..n {
                        for i in 0..nn {
                            let mut sa = 0.0;
                            for j in 0..nn {
                                sa += sv[i * nn + j] * kk[t * nn + j];
                            }
                            sa = -sa;
                            let mut acc = 0.0;
                            for j in 0..nn {
                                sv[i * nn + j] = sv[i * nn + j] * decay[t * nn + j] + sa * kk[t * nn + j] * gate[t * nn + j] + v[t * nn + i] * k[t * nn + j];
                                acc += sv[i * nn + j] * r[t * nn + j];
                            }
                            out[t * nn + i] = acc;
                        }
                    }
                    for j in 0..nn {
                        for i in 0..nn {
                            state[j * nn + i] = sv[i * nn + j];
                        }
                    }
                    out
                };
                let init = rng.vec(nn * nn, -1.0, 1.0);
                let (mut p1, mut p2) = (init.clone(), init);
                let want = direct(&mut p1);
                let got = sequential(&s, &mut p2);
                close(&got, &want, "mapping output", &format!("rwkv7 {nn} n {n}"));
                close(&p2, &p1, "mapping state", &format!("rwkv7 {nn} n {n}"));
                check(&s, &mut rng, &format!("rwkv7 {nn} n {n}"));
            }
        }
    }

    #[test]
    fn rwkv6_chunks_equal_the_recurrence() {
        let mut rng = Rng(4);
        for &nn in &[8usize, 16, 32] {
            for &n in &LENGTHS {
                let r = rng.vec(n * nn, -1.0, 1.0);
                let k = rng.vec(n * nn, -1.0, 1.0);
                let v = rng.vec(n * nn, -1.0, 1.0);
                let u = rng.vec(nn, -1.0, 1.0);
                // w = exp(-exp(w_raw)): some channels forget at once
                let lw: Vec<f64> = (0..n * nn).map(|x| -(rng.uniform(-4.0, if x % 5 == 1 { 3.5 } else { 0.5 })).exp()).collect();
                let c: Vec<f64> = (0..n).map(|t| (0..nn).map(|i| r[t * nn + i] * u[i] * k[t * nn + i]).sum()).collect();
                let s = Scan { hk: nn, hv: nn, n, q: &r, kp: &k, v: &v, a: None, b: None, lw: &lw, lw_vec: true, c: Some(&c), q_prev: true };
                // y[i] = sum_j r[j] (u[j] k[j] v[i] + S[j][i]); S[j][i] = k[j] v[i] + w[j] S[j][i]
                let direct = |state: &mut [f64]| {
                    let mut out = vec![0.0; n * nn];
                    for t in 0..n {
                        for i in 0..nn {
                            let mut acc = 0.0;
                            for j in 0..nn {
                                let kv = k[t * nn + j] * v[t * nn + i];
                                acc += r[t * nn + j] * (u[j] * kv + state[j * nn + i]);
                                state[j * nn + i] = kv + lw[t * nn + j].exp() * state[j * nn + i];
                            }
                            out[t * nn + i] = acc;
                        }
                    }
                    out
                };
                let init = rng.vec(nn * nn, -1.0, 1.0);
                let (mut p1, mut p2) = (init.clone(), init);
                let want = direct(&mut p1);
                let got = sequential(&s, &mut p2);
                close(&got, &want, "mapping output", &format!("rwkv6 {nn} n {n}"));
                close(&p2, &p1, "mapping state", &format!("rwkv6 {nn} n {n}"));
                check(&s, &mut rng, &format!("rwkv6 {nn} n {n}"));
            }
        }
    }

    #[test]
    fn mamba2_chunks_equal_the_recurrence() {
        let mut rng = Rng(5);
        for &(state, head) in &[(8usize, 12usize), (16, 32), (32, 24)] {
            for &n in &LENGTHS {
                let bm = rng.vec(n * state, -1.0, 1.0);
                let cm = rng.vec(n * state, -1.0, 1.0);
                let x = rng.vec(n * head, -1.0, 1.0);
                let dt: Vec<f64> = rng.vec(n, 0.001, 0.5);
                let a = -rng.uniform(0.2, 4.0);
                let skip = rng.uniform(-1.0, 1.0);
                let lw: Vec<f64> = dt.iter().map(|d| d * a).collect();
                let kp: Vec<f64> = (0..n * state).map(|i| dt[i / state] * bm[i]).collect();
                let c = vec![skip; n];
                let s = Scan { hk: state, hv: head, n, q: &cm, kp: &kp, v: &x, a: None, b: None, lw: &lw, lw_vec: false, c: Some(&c), q_prev: false };
                // cell[p][n] = cell * exp(dt a) + dt x[p] B[n]; y[p] = sum_n cell[p][n] C[n] + D x[p]
                let direct = |st: &mut [f64]| {
                    let mut out = vec![0.0; n * head];
                    for t in 0..n {
                        let da = (dt[t] * a).exp();
                        for p in 0..head {
                            let mut acc = 0.0;
                            for i in 0..state {
                                let cell = &mut st[i * head + p];
                                *cell = *cell * da + dt[t] * x[t * head + p] * bm[t * state + i];
                                acc += *cell * cm[t * state + i];
                            }
                            out[t * head + p] = acc + skip * x[t * head + p];
                        }
                    }
                    out
                };
                let init = rng.vec(state * head, -1.0, 1.0);
                let (mut p1, mut p2) = (init.clone(), init);
                let want = direct(&mut p1);
                let got = sequential(&s, &mut p2);
                close(&got, &want, "mapping output", &format!("mamba2 {state}x{head} n {n}"));
                close(&p2, &p1, "mapping state", &format!("mamba2 {state}x{head} n {n}"));
                check(&s, &mut rng, &format!("mamba2 {state}x{head} n {n}"));
            }
        }
    }
}
