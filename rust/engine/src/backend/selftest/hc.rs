//! Self-test cases for the hyper-connection operations (`hc_mix`, `hc_combine`) against an f64 reference of `Hc::compute`,
//! `Hc::compute_independent`, `Hc::collapse_head` and `combine_streams` (rust_decoder/deepseek4.rs).
use super::*;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl Tester<'_> {
    pub(super) fn hc_cases(&mut self) {
        if self.device.create_stream().map(|s| s.supports_glu_clamped()).unwrap_or(false) {
            for kind in [ClampedGlu::Clamp(1.5), ClampedGlu::Clamp(0.2), ClampedGlu::Situ(3.0, 2.0), ClampedGlu::Situ(1.0, 0.0)] {
                let r = self.glu_clamped_check(kind, 3001);
                self.record("glu_clamped", &format!("{kind:?} 3001 values"), r);
            }
        } else {
            self.record_skip("glu_clamped", "clamped SwiGLU, SiTU", "supports_glu_clamped is false");
        }
        if self.device.create_stream().map(|s| s.supports_rotate_heads()).unwrap_or(false) {
            for (groups, dim, shift) in [(3usize, 16usize, 4usize), (7, 128, 64), (5, 33, 1)] {
                let r = self.rotate_check(groups, dim, shift);
                self.record("rotate_heads", &format!("{groups} groups x {dim} shift {shift}"), r);
            }
        } else {
            self.record_skip("rotate_heads", "tail rotation of the indexer heads", "supports_rotate_heads is false");
        }
        let ok = self.device.create_stream().map(|s| s.supports_hc(4)).unwrap_or(false);
        if !ok {
            self.record_skip("hc", "hyper-connection mix and update", "supports_hc is false");
            return;
        }
        let modes = [
            ("independent", HcMode::Independent { magnitude: 2.0 }),
            ("sinkhorn", HcMode::Sinkhorn { iters: 5 }),
            ("head", HcMode::Head),
            ("mean", HcMode::Mean),
        ];
        for (rows, hc, hidden) in [(1usize, 4usize, 64usize), (5, 4, 300), (3, 2, 33), (2, 8, 129)] {
            for (name, mode) in modes {
                let r = self.hc_mix_check(rows, hc, hidden, mode);
                self.record("hc_mix", &format!("{name} {rows} rows x {hc} streams x {hidden}"), r);
            }
            for comb in [false, true] {
                let r = self.hc_combine_check(rows, hc, hidden, comb);
                self.record("hc_combine", &format!("{} {rows} rows x {hc} streams x {hidden}", if comb { "with mixing matrix" } else { "post only" }), r);
            }
        }
    }

    fn glu_clamped_check(&mut self, kind: ClampedGlu, n: usize) -> Result<Outcome, BackendError> {
        let gate = self.rng.vector(n, 4.0);
        let up = self.rng.vector(n, 3.0);
        let want: Vec<f64> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| {
                let (g, u) = (f64::from(g), f64::from(u));
                match kind {
                    ClampedGlu::Clamp(limit) => {
                        let limit = f64::from(limit);
                        let g = g.min(limit);
                        g / (1.0 + (-g).exp()) * u.clamp(-limit, limit)
                    }
                    ClampedGlu::Situ(beta, linear) => {
                        let (beta, linear) = (f64::from(beta), f64::from(linear));
                        let up = if linear > 0.0 { linear * (u / linear).tanh() } else { u };
                        beta * (g / beta).tanh() / (1.0 + (-g).exp()) * up
                    }
                }
            })
            .collect();
        let mut stream = self.device.create_stream()?;
        let gb = put(self.device, stream.as_mut(), &gate)?;
        let ub = put(self.device, stream.as_mut(), &up)?;
        let mut out = put(self.device, stream.as_mut(), &vec![f32::NAN; n])?;
        let mut ms = 0.0;
        timed(&mut ms, stream.as_mut(), |s| s.glu_clamped(&mut out, &gb, &ub, kind))?;
        let have = get(stream.as_mut(), &out, n)?;
        let scale: Vec<f64> = want.iter().map(|v| v.abs().max(1.0)).collect();
        Ok(Outcome { device: worst(&have, &want, &scale), cpu: 0.0, tol: 3e-5, cpu_tol: 1.0, ms })
    }

    fn rotate_check(&mut self, groups: usize, dim: usize, shift: usize) -> Result<Outcome, BackendError> {
        let x = self.rng.vector(groups * dim, 2.0);
        let mut want = vec![0.0f64; x.len()];
        for g in 0..groups {
            for d in 0..dim {
                want[g * dim + (d + shift) % dim] = f64::from(x[g * dim + d]);
            }
        }
        let mut stream = self.device.create_stream()?;
        let xb = put(self.device, stream.as_mut(), &x)?;
        let mut out = put(self.device, stream.as_mut(), &vec![f32::NAN; x.len()])?;
        let mut ms = 0.0;
        timed(&mut ms, stream.as_mut(), |s| s.rotate_heads(&mut out, &xb, dim, shift))?;
        let have = get(stream.as_mut(), &out, x.len())?;
        Ok(Outcome { device: worst(&have, &want, &vec![1.0; want.len()]), cpu: 0.0, tol: 0.0, cpu_tol: 1.0, ms })
    }

    fn hc_mix_check(&mut self, rows: usize, hc: usize, hidden: usize, mode: HcMode) -> Result<Outcome, BackendError> {
        let eps = 1e-6f64;
        let width = mode.mix_width(hc);
        let streams = self.rng.vector(rows * hc * hidden, 1.5);
        let mixes = self.rng.vector(rows * width, 2.0);
        let base = self.rng.vector(width.max(1), 1.0);
        let scale = self.rng.vector(3, 0.8);
        let mut want_post = vec![0.0f64; rows * hc];
        let mut want_comb = vec![0.0f64; rows * hc * hc];
        let mut want_collapsed = vec![0.0f64; rows * hidden];
        for r in 0..rows {
            let m: Vec<f64> = mixes[r * width..(r + 1) * width].iter().map(|&v| f64::from(v)).collect();
            let (s0, s1, s2) = (f64::from(scale[0]), f64::from(scale[1]), f64::from(scale[2]));
            let pre: Vec<f64> = (0..hc)
                .map(|n| match mode {
                    HcMode::Mean => 1.0 / hc as f64,
                    _ => sigmoid(m[n] * s0 + f64::from(base[n])) + eps,
                })
                .collect();
            match mode {
                HcMode::Independent { magnitude } => {
                    for n in 0..hc {
                        want_post[r * hc + n] = f64::from(magnitude) * sigmoid(m[hc + n] * s1 + f64::from(base[hc + n])) + eps;
                    }
                }
                HcMode::Sinkhorn { iters } => {
                    for n in 0..hc {
                        want_post[r * hc + n] = 2.0 * sigmoid(m[hc + n] * s1 + f64::from(base[hc + n]));
                    }
                    let c = &mut want_comb[r * hc * hc..(r + 1) * hc * hc];
                    for j in 0..hc {
                        let row = &mut c[j * hc..(j + 1) * hc];
                        for (k, v) in row.iter_mut().enumerate() {
                            *v = m[2 * hc + j * hc + k] * s2 + f64::from(base[2 * hc + j * hc + k]);
                        }
                        let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let sum: f64 = row.iter().map(|v| (v - max).exp()).sum();
                        row.iter_mut().for_each(|v| *v = (*v - max).exp() / sum + eps);
                    }
                    let columns = |c: &mut [f64]| {
                        for k in 0..hc {
                            let s: f64 = (0..hc).map(|j| c[j * hc + k]).sum();
                            for j in 0..hc {
                                c[j * hc + k] /= s + eps;
                            }
                        }
                    };
                    columns(c);
                    for _ in 1..iters.max(1) {
                        for j in 0..hc {
                            let s: f64 = c[j * hc..(j + 1) * hc].iter().sum();
                            c[j * hc..(j + 1) * hc].iter_mut().for_each(|v| *v /= s + eps);
                        }
                        columns(c);
                    }
                }
                _ => {}
            }
            for d in 0..hidden {
                want_collapsed[r * hidden + d] = (0..hc).map(|n| pre[n] * f64::from(streams[(r * hc + n) * hidden + d])).sum();
            }
        }
        let mut stream = self.device.create_stream()?;
        let sb = put(self.device, stream.as_mut(), &streams)?;
        let mb = put(self.device, stream.as_mut(), &mixes)?;
        let bb = put(self.device, stream.as_mut(), &base)?;
        let cb = put(self.device, stream.as_mut(), &scale)?;
        let mut post = put(self.device, stream.as_mut(), &vec![f32::NAN; rows * hc])?;
        let mut comb = put(self.device, stream.as_mut(), &vec![f32::NAN; rows * hc * hc])?;
        let mut collapsed = put(self.device, stream.as_mut(), &vec![f32::NAN; rows * hidden])?;
        let weights = !matches!(mode, HcMode::Mean);
        let mut ms = 0.0;
        timed(&mut ms, stream.as_mut(), |s| {
            s.hc_mix(&mut collapsed, &mut post, &mut comb, &sb, weights.then_some(&mb), weights.then_some(&bb), weights.then_some(&cb), rows, hc, hidden, mode, 1e-6)
        })?;
        let one = |want: &[f64], have: Vec<f32>| worst(&have, want, &vec![1.0; want.len()]);
        let mut err = one(&want_collapsed, get(stream.as_mut(), &collapsed, rows * hidden)?);
        if matches!(mode, HcMode::Independent { .. } | HcMode::Sinkhorn { .. }) {
            err = err.max(one(&want_post, get(stream.as_mut(), &post, rows * hc)?));
        }
        if matches!(mode, HcMode::Sinkhorn { .. }) {
            err = err.max(one(&want_comb, get(stream.as_mut(), &comb, rows * hc * hc)?));
        }
        Ok(Outcome { device: err, cpu: 0.0, tol: 5e-5, cpu_tol: 1.0, ms })
    }

    fn hc_combine_check(&mut self, rows: usize, hc: usize, hidden: usize, with_comb: bool) -> Result<Outcome, BackendError> {
        let streams = self.rng.vector(rows * hc * hidden, 1.5);
        let post = self.rng.vector(rows * hc, 1.0);
        let comb = self.rng.vector(rows * hc * hc, 1.0);
        let block = self.rng.vector(rows * hidden, 1.5);
        let mut want = vec![0.0f64; rows * hc * hidden];
        for r in 0..rows {
            for k in 0..hc {
                for d in 0..hidden {
                    let mixed: f64 = if with_comb {
                        (0..hc).map(|j| f64::from(comb[(r * hc + j) * hc + k]) * f64::from(streams[(r * hc + j) * hidden + d])).sum()
                    } else {
                        f64::from(streams[(r * hc + k) * hidden + d])
                    };
                    want[(r * hc + k) * hidden + d] = f64::from(post[r * hc + k]) * f64::from(block[r * hidden + d]) + mixed;
                }
            }
        }
        let mut stream = self.device.create_stream()?;
        let mut sb = put(self.device, stream.as_mut(), &streams)?;
        let pb = put(self.device, stream.as_mut(), &post)?;
        let cb = put(self.device, stream.as_mut(), &comb)?;
        let bb = put(self.device, stream.as_mut(), &block)?;
        let mut ms = 0.0;
        timed(&mut ms, stream.as_mut(), |s| s.hc_combine(&mut sb, &pb, with_comb.then_some(&cb), &bb, rows, hc, hidden))?;
        let have = get(stream.as_mut(), &sb, want.len())?;
        let scale: Vec<f64> = want.iter().map(|v| v.abs().max(1.0)).collect();
        Ok(Outcome { device: worst(&have, &want, &scale), cpu: 0.0, tol: 5e-6, cpu_tol: 1.0, ms })
    }
}
