//! Self-test cases for the Qwen3.8-Flash-Next operations (`q4_act`, `q4_gate_mix`, `q4_ple_gate`, `q4_ple_conv`, `q4_pool_keys`) against f64 references
//! of `Gate::mix`, `Qwen4Model::q4_ple` and `Qwen4Model::q4_select` (rust_decoder/qwen4.rs).
use super::*;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}

impl Tester<'_> {
    pub(super) fn q4_cases(&mut self) {
        let device = self.device;
        let supported = |hc: usize, kernel: usize, dilation: usize| device.create_stream().map(|s| s.supports_qwen4(hc, kernel, dilation)).unwrap_or(false);
        if !supported(3, 4, 3) {
            self.record_skip("q4", "Qwen3.8-Flash-Next streams, n-gram embedding and token indexer", "supports_qwen4 is false");
            return;
        }
        for (mode, scale, factor) in [(0, 1.0f32 / 3.0, 1.0f32), (1, 0.25, 2.0)] {
            let r = self.q4_act_check(mode, scale, factor, 3001);
            self.record("q4_act", &format!("{} x{scale:.2} *{factor} 3001 values", if mode == 0 { "silu" } else { "sigmoid" }), r);
        }
        for (rows, hc, hidden) in [(1usize, 3usize, 16usize), (5, 3, 64), (3, 4, 300), (2, 8, 33)] {
            let r = self.q4_gate_mix_check(rows, hc, hidden);
            self.record("q4_gate_mix", &format!("{rows} rows x {hc} streams x {hidden}"), r);
            let r = self.q4_ple_gate_check(rows, hc, hidden);
            self.record("q4_ple_gate", &format!("{rows} rows x {hc} streams x {hidden}"), r);
        }
        for (width, kernel, dilation, long) in [(64usize, 4usize, 3usize, false), (200, 2, 1, false), (96, 3, 2, false), (130, 4, 8, false), (48, 4, 3, true)] {
            if !supported(1, kernel, dilation) {
                continue;
            }
            let r = self.q4_ple_conv_check(width, kernel, dilation, long);
            self.record("q4_ple_conv", &format!("{width} channels kernel {kernel} dilation {dilation}{}", if long { ", runs of 2600 and 1000 rows" } else { "" }), r);
        }
        for (dim, rope_dim, neox, ratio) in [(16usize, 8usize, true, 4usize), (16, 16, false, 4), (64, 32, true, 7), (128, 64, true, 16), (33, 32, true, 2), (256, 128, false, 4)] {
            let r = self.q4_pool_keys_check(dim, rope_dim, neox, ratio);
            self.record("q4_pool_keys", &format!("dim {dim} rotary {rope_dim} {} blocks of {ratio}", if neox { "neox" } else { "interleaved" }), r);
        }
    }

    fn q4_act_check(&mut self, mode: i32, scale: f32, factor: f32, n: usize) -> Result<Outcome, BackendError> {
        let x = self.rng.vector(n, 6.0);
        let want: Vec<f64> = x
            .iter()
            .map(|&v| {
                let v = f64::from(v) * f64::from(scale);
                f64::from(factor) * if mode == 0 { silu(v) } else { sigmoid(v) }
            })
            .collect();
        let mut stream = self.device.create_stream()?;
        let mut xb = put(self.device, stream.as_mut(), &x)?;
        let mut ms = 0.0;
        timed(&mut ms, stream.as_mut(), |s| s.q4_act(&mut xb, mode, scale, factor))?;
        let have = get(stream.as_mut(), &xb, n)?;
        let scale: Vec<f64> = want.iter().map(|v| v.abs().max(1.0)).collect();
        Ok(Outcome { device: worst(&have, &want, &scale), cpu: 0.0, tol: 3e-5, cpu_tol: 1.0, ms })
    }

    fn q4_gate_mix_check(&mut self, rows: usize, hc: usize, hidden: usize) -> Result<Outcome, BackendError> {
        let normed = self.rng.vector(rows * hc * hidden, 2.0);
        let gates = self.rng.vector(rows * hc * hidden, 3.0);
        let mut want = vec![0.0f64; rows * hidden];
        for r in 0..rows {
            for d in 0..hidden {
                want[r * hidden + d] = (0..hc)
                    .map(|n| {
                        let at = (r * hc + n) * hidden + d;
                        sigmoid(f64::from(gates[at])) * f64::from(normed[at]) / hc as f64
                    })
                    .sum();
            }
        }
        let mut stream = self.device.create_stream()?;
        let nb = put(self.device, stream.as_mut(), &normed)?;
        let gb = put(self.device, stream.as_mut(), &gates)?;
        let mut out = put(self.device, stream.as_mut(), &vec![f32::NAN; rows * hidden])?;
        let mut ms = 0.0;
        timed(&mut ms, stream.as_mut(), |s| s.q4_gate_mix(&mut out, &nb, &gb, rows, hc, hidden))?;
        let have = get(stream.as_mut(), &out, want.len())?;
        let scale: Vec<f64> = want.iter().map(|v| v.abs().max(1.0)).collect();
        Ok(Outcome { device: worst(&have, &want, &scale), cpu: 0.0, tol: 5e-6, cpu_tol: 1.0, ms })
    }

    fn q4_ple_gate_check(&mut self, rows: usize, hc: usize, hidden: usize) -> Result<Outcome, BackendError> {
        let mut key = self.rng.vector(rows * hc * hidden, 1.5);
        let query = self.rng.vector(rows * hc * hidden, 1.5);
        let value = self.rng.vector(rows * hidden, 2.0);
        // one stream with a zero dot product: the signed root of 0 is 0, not 1e-3
        key[..hidden].iter_mut().for_each(|v| *v = 0.0);
        let mut want = vec![0.0f64; rows * hc * hidden];
        for r in 0..rows {
            for n in 0..hc {
                let at = (r * hc + n) * hidden;
                let dot: f64 = (0..hidden).map(|d| f64::from(key[at + d]) * f64::from(query[at + d])).sum();
                let s = dot / (hidden as f64).sqrt();
                let g = s.abs().max(1e-6).sqrt() * if s == 0.0 { 0.0 } else { s.signum() };
                for d in 0..hidden {
                    want[at + d] = sigmoid(g) * f64::from(value[r * hidden + d]);
                }
            }
        }
        let mut stream = self.device.create_stream()?;
        let kb = put(self.device, stream.as_mut(), &key)?;
        let qb = put(self.device, stream.as_mut(), &query)?;
        let vb = put(self.device, stream.as_mut(), &value)?;
        let mut out = put(self.device, stream.as_mut(), &vec![f32::NAN; want.len()])?;
        let mut ms = 0.0;
        timed(&mut ms, stream.as_mut(), |s| s.q4_ple_gate(&mut out, &kb, &qb, &vb, rows, hc, hidden))?;
        let have = get(stream.as_mut(), &out, want.len())?;
        let scale: Vec<f64> = want.iter().map(|v| v.abs().max(1.0)).collect();
        Ok(Outcome { device: worst(&have, &want, &scale), cpu: 0.0, tol: 5e-5, cpu_tol: 1.0, ms })
    }

    /// The dilated convolution over several calls with three sequences (state kept between the calls, a reset at the start of each).
    fn q4_ple_conv_check(&mut self, width: usize, kernel: usize, dilation: usize, long: bool) -> Result<Outcome, BackendError> {
        let conv = self.rng.vector(width * kernel, 0.7);
        let short: [&[(usize, usize)]; 5] = [&[(0, 29), (1, 2), (2, 1)], &[(0, 1), (1, 1), (2, 1)], &[(0, 1), (1, 1), (2, 1)], &[(2, 9), (0, 1)], &[(1, 1), (0, 1), (2, 1)]];
        // runs longer than one dispatch of the Vulkan shader (it cuts them into pieces that continue from the stored history)
        let big: [&[(usize, usize)]; 3] = [&[(0, 2600), (1, 3)], &[(0, 1), (1, 1)], &[(1, 1000), (0, 1)]];
        let calls: &[&[(usize, usize)]] = if long { &big } else { &short };
        let hl = (kernel - 1) * dilation;
        let mut stream = self.device.create_stream()?;
        let conv_buf = put(self.device, stream.as_mut(), &conv)?;
        let offset = 256;
        let states: Vec<DeviceBuffer> = (0..3).map(|_| self.device.allocate_buffer(offset + width * hl * 4)).collect::<Result<_, _>>()?;
        let mut hist = vec![vec![0.0f32; width * hl]; 3];
        let mut started = [false; 3];
        let (mut worst_err, mut total_ms) = (0.0f64, 0.0f64);
        for call in calls.iter().copied() {
            let rows: usize = call.iter().map(|c| c.1).sum();
            let gated = self.rng.vector(rows * width, 1.0);
            let normed = self.rng.vector(rows * width, 1.5);
            let mut want = vec![0.0f64; rows * width];
            let mut runs = Vec::new();
            let mut first = 0;
            for &(seq, n) in call {
                for t in 0..n {
                    for ch in 0..width {
                        let h = &mut hist[seq][ch * hl..(ch + 1) * hl];
                        let w = &conv[ch * kernel..(ch + 1) * kernel];
                        let cur = normed[(first + t) * width + ch];
                        let mut value = 0.0f64;
                        for j in 0..kernel - 1 {
                            value += f64::from(w[j]) * f64::from(h[j * dilation]);
                        }
                        value += f64::from(w[kernel - 1]) * f64::from(cur);
                        want[(first + t) * width + ch] = f64::from(gated[(first + t) * width + ch]) + silu(value);
                        h.copy_within(1.., 0);
                        h[hl - 1] = cur;
                    }
                }
                runs.push(DeltaRun { state: &states[seq], first_row: first, rows: n, reset: !started[seq] });
                started[seq] = true;
                first += n;
            }
            let gb = put(self.device, stream.as_mut(), &gated)?;
            let nb = put(self.device, stream.as_mut(), &normed)?;
            let mut out = put(self.device, stream.as_mut(), &vec![f32::NAN; rows * width])?;
            let mut ms = 0.0;
            timed(&mut ms, stream.as_mut(), |s| s.q4_ple_conv(&mut out, &gb, &nb, &conv_buf, &runs, offset, width, kernel, dilation))?;
            total_ms += ms;
            let have = get(stream.as_mut(), &out, rows * width)?;
            let scale: Vec<f64> = want.iter().map(|v| v.abs().max(1.0)).collect();
            worst_err = worst_err.max(worst(&have, &want, &scale));
        }
        Ok(Outcome { device: worst_err, cpu: 0.0, tol: 2e-5, cpu_tol: 1.0, ms: total_ms })
    }

    /// Rows that complete a block (their sources are rows of the batch and slots of the store) next to rows that keep their key.
    fn q4_pool_keys_check(&mut self, dim: usize, rope_dim: usize, neox: bool, ratio: usize) -> Result<Outcome, BackendError> {
        let (slots, positions) = (64usize, 96usize);
        let width = 2 * dim;
        let store_values = self.rng.vector(slots * width, 1.5);
        let rows = 7;
        let k = self.rng.vector(rows * dim, 1.5);
        let norm = self.rng.vector(dim, 0.5).iter().map(|v| v + 1.0).collect::<Vec<f32>>();
        let half = rope_dim / 2;
        let angle = |p: usize, i: usize| (p as f64 + 1.0) * 0.05 / (1.0 + i as f64);
        let cos: Vec<f32> = (0..positions * half).map(|i| angle(i / half.max(1), i % half.max(1)).cos() as f32).collect();
        let sin: Vec<f32> = (0..positions * half).map(|i| angle(i / half.max(1), i % half.max(1)).sin() as f32).collect();
        let rope = RopeParams { heads: 1, kv_heads: 1, dim, rope_dim, neox, cos: Arc::new(cos.clone()), sin: Arc::new(sin.clone()) };
        // rows 0, 3 and 6 complete a block: row 0 reads everything from the store, row 3 the three previous rows of the batch (when ratio >= 4) and the rest from the store,
        // row 6 only rows of the batch (padded with store slots when the block is longer than the batch run)
        let mut meta = Vec::new();
        for r in 0..rows {
            let completes = matches!(r, 0 | 3 | 6);
            meta.push(if completes { (5 + 3 * r) as i32 } else { -1 });
            for j in 0..ratio {
                let back = ratio - 1 - j;
                meta.push(match r {
                    0 => -1 - (7 + j as i32),
                    _ if back <= r => (r - back) as i32,
                    _ => -1 - (20 + j as i32),
                });
            }
        }
        let mut want = vec![0.0f64; rows * dim];
        for r in 0..rows {
            let m = &meta[r * (ratio + 1)..(r + 1) * (ratio + 1)];
            if m[0] < 0 {
                for c in 0..dim {
                    want[r * dim + c] = f64::from(k[r * dim + c]);
                }
                continue;
            }
            let mean: Vec<f64> = (0..dim)
                .map(|c| {
                    m[1..]
                        .iter()
                        .map(|&s| if s >= 0 { f64::from(k[s as usize * dim + c]) } else { f64::from(store_values[(-1 - s) as usize * width + c]) })
                        .sum::<f64>()
                        / ratio as f64
                })
                .collect();
            let ss: f64 = mean.iter().map(|v| v * v).sum::<f64>() / dim as f64;
            let inv = 1.0 / (ss + 1e-6).sqrt();
            let mut v: Vec<f64> = (0..dim).map(|c| mean[c] * inv * f64::from(norm[c])).collect();
            let p = m[0] as usize;
            for i in 0..half {
                let (a_at, b_at) = if neox { (i, i + half) } else { (2 * i, 2 * i + 1) };
                let (a, b) = (v[a_at], v[b_at]);
                let (c, s) = (f64::from(cos[p * half + i]), f64::from(sin[p * half + i]));
                v[a_at] = a * c - b * s;
                v[b_at] = b * c + a * s;
            }
            want[r * dim..(r + 1) * dim].copy_from_slice(&v);
        }
        let mut stream = self.device.create_stream()?;
        let kb = put(self.device, stream.as_mut(), &k)?;
        let sb = put(self.device, stream.as_mut(), &store_values)?;
        let nb = put(self.device, stream.as_mut(), &norm)?;
        let mut out = put(self.device, stream.as_mut(), &vec![f32::NAN; rows * dim])?;
        let mut ms = 0.0;
        timed(&mut ms, stream.as_mut(), |s| s.q4_pool_keys(&mut out, &kb, &sb, width, &nb, &rope, &meta, ratio, 1e-6))?;
        let have = get(stream.as_mut(), &out, want.len())?;
        let scale: Vec<f64> = want.iter().map(|v| v.abs().max(1.0)).collect();
        Ok(Outcome { device: worst(&have, &want, &scale), cpu: 0.0, tol: 5e-5, cpu_tol: 1.0, ms })
    }
}
