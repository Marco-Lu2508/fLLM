//! Self-test cases for the Inkling operations: the convolution that adds its input back (`conv_resid`, prompts, decode steps and several sequences per call, state
//! carried from call to call) and the attention with a learned distance profile and log scaling (`rel_attention`), against f64 transcriptions of `Conv::step`
//! and of the attention loop of `inkling_layers` (rust_decoder/inkling.rs).
use super::*;

impl Tester<'_> {
    pub(super) fn inkling_cases(&mut self) {
        for (channels, taps) in [(40usize, 4usize), (130, 2), (96, 8), (17, 3)] {
            let supported = self.device.create_stream().map(|s| s.supports_conv_resid(channels, taps)).unwrap_or(false);
            let label = format!("{channels} channels, {taps} taps");
            if !supported {
                self.record_skip("conv resid", &label, "supports_conv_resid is false");
                continue;
            }
            let r = self.conv_resid_check(channels, taps);
            self.record("conv resid", &label, r);
        }
        // heads, kv heads, head width, relative channels, extent, window (0: all), log floor (0: off), lengths, rows per sequence (the last positions), page size
        type Case = (usize, usize, usize, usize, usize, usize, usize, &'static [usize], usize, usize);
        let cases: [Case; 6] = [
            (4, 2, 16, 4, 10, 0, 4, &[37, 12], 0, 16),
            (4, 1, 8, 4, 10, 6, 0, &[40, 3], 0, 5),
            (6, 3, 24, 7, 50, 0, 8, &[300], 6, 16),
            (8, 8, 64, 16, 600, 100, 0, &[700, 90], 4, 16),
            (16, 4, 128, 16, 1000, 0, 64, &[3000], 3, 16),
            (2, 2, 40, 3, 4, 0, 0, &[50, 17, 1], 0, 4),
        ];
        for (i, &(heads, kv_heads, head_dim, d_rel, extent, window, log_floor, lengths, rows, block_size)) in cases.iter().enumerate() {
            let stride = lengths.iter().map(|n| n.div_ceil(block_size)).max().unwrap_or(1) + 2;
            let params = RelAttnParams { heads, kv_heads, head_dim, d_rel, extent, window, log_floor, log_alpha: 0.3, scale: 1.0 / head_dim as f32, stride, block_size };
            let supported = self.device.create_stream().map(|s| s.supports_rel_attention(&params, 1)).unwrap_or(false);
            let label = format!("{heads}/{kv_heads} heads x {head_dim}, {d_rel} rel, extent {extent}, window {window}, log floor {log_floor}, {} tokens", lengths.iter().sum::<usize>());
            if !supported {
                self.record_skip("rel attention", &label, "supports_rel_attention is false");
                continue;
            }
            let r = self.rel_attention_check(&params, lengths, rows, i == 0);
            self.record("rel attention", &label, r);
        }
    }

    fn conv_resid_check(&mut self, channels: usize, taps: usize) -> Result<Outcome, BackendError> {
        let conv = self.rng.vector(channels * taps, 0.7);
        let calls: [&[(usize, usize)]; 5] = [&[(0, 29), (1, 2), (2, 1)], &[(0, 1), (1, 1), (2, 1)], &[(0, 1), (1, 1), (2, 1)], &[(2, 9), (0, 1)], &[(1, 1), (0, 1), (2, 1)]];
        let mut stream = self.device.create_stream()?;
        let conv_buf = put(self.device, stream.as_mut(), &conv)?;
        let offset = 256;
        let states: Vec<DeviceBuffer> = (0..3).map(|_| self.device.allocate_buffer(offset + channels * (taps - 1) * 4)).collect::<Result<_, _>>()?;
        let mut hist = vec![vec![0.0f32; channels * (taps - 1)]; 3];
        let mut started = [false; 3];
        let (mut worst_err, mut total_ms) = (0.0f64, 0.0f64);
        for call in calls {
            let rows: usize = call.iter().map(|c| c.1).sum();
            let x = self.rng.vector(rows * channels, 1.5);
            let mut want = vec![0.0f64; rows * channels];
            let mut runs = Vec::new();
            let mut first = 0;
            for &(seq, n) in call {
                for t in 0..n {
                    let row = &x[(first + t) * channels..(first + t + 1) * channels];
                    for ch in 0..channels {
                        let h = &mut hist[seq][ch * (taps - 1)..(ch + 1) * (taps - 1)];
                        let w = &conv[ch * taps..(ch + 1) * taps];
                        let mut value = 0.0f64;
                        for (wi, hi) in w.iter().zip(h.iter()) {
                            value += f64::from(*wi) * f64::from(*hi);
                        }
                        value += f64::from(w[taps - 1]) * f64::from(row[ch]);
                        want[(first + t) * channels + ch] = value + f64::from(row[ch]);
                        h.copy_within(1.., 0);
                        h[taps - 2] = row[ch];
                    }
                }
                runs.push(DeltaRun { state: &states[seq], first_row: first, rows: n, reset: !started[seq] });
                started[seq] = true;
                first += n;
            }
            let xb = put(self.device, stream.as_mut(), &x)?;
            let mut out = put(self.device, stream.as_mut(), &vec![f32::NAN; rows * channels])?;
            let mut ms = 0.0;
            timed(&mut ms, stream.as_mut(), |s| s.conv_resid(channels, taps, &conv_buf, &runs, offset, &xb, &mut out))?;
            total_ms += ms;
            let have = get(stream.as_mut(), &out, rows * channels)?;
            let scale: Vec<f64> = want.iter().map(|v| v.abs().max(1.0)).collect();
            worst_err = worst_err.max(worst(&have, &want, &scale));
        }
        // at most taps + 1 products and sums of values below about 6 in f32: a few 1e-6; 1e-5 leaves room for fused multiply-adds being contracted differently
        Ok(Outcome { device: worst_err, cpu: 0.0, tol: 1e-5, cpu_tol: 1.0, ms: total_ms })
    }

    fn rel_attention_check(&mut self, p: &RelAttnParams, lengths: &[usize], last_rows: usize, decode_after: bool) -> Result<Outcome, BackendError> {
        let (heads, kv_heads, hd) = (p.heads, p.kv_heads, p.head_dim);
        // random block tables
        let pool_blocks = lengths.iter().map(|n| n.div_ceil(p.block_size)).sum::<usize>() + 4;
        let mut free: Vec<usize> = (0..pool_blocks).collect();
        for i in (1..free.len()).rev() {
            let j = (self.rng.next() % (i as u64 + 1)) as usize;
            free.swap(i, j);
        }
        let mut tables_seq = Vec::new();
        for &n in lengths {
            let mut table = vec![-1i64; p.stride];
            for entry in table.iter_mut().take(n.div_ceil(p.block_size)) {
                *entry = free.pop().ok_or_else(|| BackendError::InvalidShape("pool too small".into()))? as i64;
            }
            tables_seq.push(table);
        }
        let slot_of = |s: usize, pos: usize| tables_seq[s][pos / p.block_size] as usize * p.block_size + pos % p.block_size;
        let mut stream = self.device.create_stream()?;
        let cache = self.device.allocate_kv_cache(1, pool_blocks, p.block_size, kv_heads, hd, hd, 1)?;
        let mut slot_list = Vec::new();
        let mut k = Vec::new();
        let mut v = Vec::new();
        for (s, &n) in lengths.iter().enumerate() {
            for pos in 0..n {
                slot_list.push(slot_of(s, pos));
            }
            k.extend(self.rng.vector(n * kv_heads * hd, 1.0));
            v.extend(self.rng.vector(n * kv_heads * hd, 1.0));
        }
        let kb = put(self.device, stream.as_mut(), &k)?;
        let vb = put(self.device, stream.as_mut(), &v)?;
        stream.kv_cache_write(&cache, 0, &slot_list, &kb, &vb)?;
        let profile = self.rng.vector(p.d_rel * p.extent, 0.5);
        let pb = put(self.device, stream.as_mut(), &profile)?;
        let offsets: Vec<usize> = lengths.iter().scan(0usize, |acc, &n| { let o = *acc; *acc += n; Some(o) }).collect();
        let rounded = |x: f32| f64::from(f16::from_f32(x).to_f32());
        // rows: the last `last_rows` positions of every sequence (0: all of them); a second call holds the last token of every sequence alone (a decode step)
        let mut calls: Vec<Vec<(usize, usize)>> = vec![lengths.iter().enumerate().flat_map(|(s, &n)| ((if last_rows == 0 { 0 } else { n.saturating_sub(last_rows) })..n).map(move |pos| (s, pos))).collect()];
        if decode_after {
            calls.push(lengths.iter().enumerate().map(|(s, &n)| (s, n - 1)).collect());
        }
        let (mut worst_err, mut total_ms) = (0.0f64, 0.0f64);
        for rows_here in calls {
            let rows = rows_here.len();
            let q = self.rng.vector(rows * heads * hd, 1.0);
            let rel = self.rng.vector(rows * heads * p.d_rel, 1.0);
            let qb = put(self.device, stream.as_mut(), &q)?;
            let rb = put(self.device, stream.as_mut(), &rel)?;
            let lens: Vec<i64> = rows_here.iter().map(|&(_, pos)| pos as i64 + 1).collect();
            let tables: Vec<i64> = rows_here.iter().flat_map(|&(s, _)| tables_seq[s].clone()).collect();
            let mut out = put(self.device, stream.as_mut(), &vec![f32::NAN; rows * heads * hd])?;
            let mut ms = 0.0;
            timed(&mut ms, stream.as_mut(), |s| s.rel_attention(&mut out, &qb, &rb, &pb, &cache, &tables, &lens, 0, p))?;
            total_ms += ms;
            let have = get(stream.as_mut(), &out, rows * heads * hd)?;
            let mut want = vec![0.0f64; rows * heads * hd];
            for (r, &(s, pos)) in rows_here.iter().enumerate() {
                let n = pos + 1;
                let tau = if p.log_floor > 0 { 1.0 + f64::from(p.log_alpha) * ((n as f64) / p.log_floor as f64).max(1.0).ln() } else { 1.0 };
                let first = if p.window > 0 { n.saturating_sub(p.window) } else { 0 };
                for h in 0..heads {
                    let kh = h / (heads / kv_heads);
                    let scores: Vec<f64> = (first..n)
                        .map(|j| {
                            let dot: f64 = (0..hd).map(|d| f64::from(q[(r * heads + h) * hd + d]) * tau * rounded(k[((offsets[s] + j) * kv_heads + kh) * hd + d])).sum();
                            let dist = pos - j;
                            let bias = if dist < p.extent { (0..p.d_rel).map(|c| f64::from(rel[(r * heads + h) * p.d_rel + c]) * f64::from(profile[c * p.extent + dist])).sum::<f64>() * tau } else { 0.0 };
                            dot * f64::from(p.scale) + bias
                        })
                        .collect();
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let exps: Vec<f64> = scores.iter().map(|x| (x - max).exp()).collect();
                    let sum: f64 = exps.iter().sum();
                    for d in 0..hd {
                        let acc: f64 = (first..n).zip(&exps).map(|(j, e)| e * rounded(v[((offsets[s] + j) * kv_heads + kh) * hd + d])).sum();
                        want[(r * heads + h) * hd + d] = acc / sum;
                    }
                }
            }
            let scale: Vec<f64> = want.iter().map(|v| v.abs().max(1.0)).collect();
            worst_err = worst_err.max(worst(&have, &want, &scale));
        }
        Ok(Outcome { device: worst_err, cpu: 0.0, tol: 2e-5, cpu_tol: 1.0, ms: total_ms })
    }
}
