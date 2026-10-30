//! Self-test cases: the RWKV-6 / RWKV-7 operations (group `rwkv`).
//!
//! Every case runs the five operations of a layer (`rwkv_shift`, `rwkv_lerp`, `rwkv_act`, `rwkv7_wkv` or `rwkv6_wkv`) over several calls with
//! prompts, single tokens and mixed batches of three sequences whose states live in device state buffers (the state of a layer sits 256
//! bytes into the buffer), and compares the outputs with the f64 recurrences of `cpu_rwkv`. The shift vector and the WKV matrices are carried
//! from call to call on both sides, so an error in the state handling shows in the later calls.
use super::*;
use crate::backend::{Rwkv7Inputs, RwkvShape, RwkvWeights};

impl Tester<'_> {
    pub(super) fn rwkv_cases(&mut self) {
        for (head_size, heads) in [(16usize, 3usize), (32, 2), (64, 2), (128, 1)] {
            for v7 in [true, false] {
                let shape = RwkvShape { hidden: head_size * heads, head_size, group_eps: 64e-5, v7 };
                let label = format!("{} {heads} heads of {head_size}", if v7 { "rwkv7" } else { "rwkv6" });
                match self.device.create_stream() {
                    Ok(s) if s.supports_rwkv(&shape) => {}
                    Ok(_) => {
                        self.record_skip("rwkv", &label, "the RWKV operations for this shape (supports_rwkv is false)");
                        continue;
                    }
                    Err(e) => {
                        self.record("rwkv", &label, Err(e));
                        continue;
                    }
                }
                let r = self.rwkv(&shape);
                self.record("rwkv", &label, r);
            }
        }
    }

    fn rwkv(&mut self, shape: &RwkvShape) -> Result<Outcome, BackendError> {
        let (c, n, heads) = (shape.hidden, shape.head_size, shape.heads());
        let norm_w: Vec<f32> = (0..c).map(|_| 1.0 + 0.25 * self.rng.unit()).collect();
        let norm_b = self.rng.vector(c, 0.2);
        let k_k: Vec<f32> = (0..c).map(|_| 0.5 + self.rng.unit().abs()).collect();
        let k_a: Vec<f32> = (0..c).map(|_| self.rng.unit().abs()).collect();
        let r_k = self.rng.vector(c, 0.3);
        let mix = self.rng.vector(c, 1.0);
        let params = crate::cpu_rwkv::Wkv {
            heads,
            n,
            eps: shape.group_eps,
            norm_weight: &norm_w,
            norm_bias: &norm_b,
            // RWKV-6 keeps its bonus in `k_k`
            k_k: &k_k,
            k_a: &k_a,
            r_k: &r_k,
        };
        let calls: [&[(usize, usize)]; 6] = [
            &[(0, 37), (1, 5), (2, 2)],
            &[(0, 1), (1, 1), (2, 1)],
            &[(0, 1), (1, 1), (2, 1)],
            &[(0, 20), (1, 1)],
            &[(2, 6), (0, 1), (1, 3)],
            &[(1, 1), (0, 1), (2, 1)],
        ];
        let mut stream = self.device.create_stream()?;
        let norm_w_buf = put(self.device, stream.as_mut(), &norm_w)?;
        let norm_b_buf = put(self.device, stream.as_mut(), &norm_b)?;
        let k_k_buf = put(self.device, stream.as_mut(), &k_k)?;
        let k_a_buf = put(self.device, stream.as_mut(), &k_a)?;
        let r_k_buf = put(self.device, stream.as_mut(), &r_k)?;
        let mix_buf = put(self.device, stream.as_mut(), &mix)?;
        let weights = RwkvWeights {
            norm_weight: &norm_w_buf,
            norm_bias: &norm_b_buf,
            k_k: &k_k_buf,
            k_a: shape.v7.then_some(&k_a_buf),
            r_k: shape.v7.then_some(&r_k_buf),
        };
        let offset = 256;
        let states: Vec<DeviceBuffer> = (0..3).map(|_| self.device.allocate_buffer(offset + shape.layer_state_bytes())).collect::<Result<_, _>>()?;
        let mut rec = vec![vec![0.0f32; heads * n * n]; 3];
        let mut shifted_state = vec![vec![0.0f32; c]; 3];
        let mut seen = [false; 3];
        let (mut worst_err, mut total_ms) = (0.0f64, 0.0f64);
        for (call_index, call) in calls.iter().enumerate() {
            let rows: usize = call.iter().map(|&(_, len)| len).sum();
            let r = self.rng.vector(rows * c, 1.0);
            let k = self.rng.vector(rows * c, 1.0);
            let v = self.rng.vector(rows * c, 1.0);
            let w = self.rng.vector(rows * c, 1.0);
            let a = self.rng.vector(rows * c, 1.0);
            let gate = self.rng.vector(rows * c, 1.0);
            let x = self.rng.vector(rows * c, 1.0);
            let delta = self.rng.vector(rows * c, 0.5);
            // odd calls blend the values with the stored ones (RWKV-7 layers after the first)
            let use_res = shape.v7 && call_index % 2 == 1;
            let v_res = self.rng.vector(rows * c, 1.0);
            let v_first_in = self.rng.vector(rows * c, 1.0);
            let mut want_shift = vec![0.0f32; rows * c];
            let mut want_out = vec![0.0f32; rows * c];
            let mut want_first = v_first_in.clone();
            let mut runs = Vec::with_capacity(call.len());
            let mut first = 0;
            for &(seq, len) in *call {
                let reset = !seen[seq];
                seen[seq] = true;
                if reset {
                    rec[seq].fill(0.0);
                    shifted_state[seq].fill(0.0);
                }
                let span = first * c..(first + len) * c;
                crate::cpu_rwkv::shift(c, len, &x[span.clone()], &mut shifted_state[seq], &mut want_shift[span.clone()]);
                if shape.v7 {
                    let inputs = crate::cpu_rwkv::Inputs7 {
                        r: &r[span.clone()],
                        k: &k[span.clone()],
                        v: &v[span.clone()],
                        w: &w[span.clone()],
                        a: &a[span.clone()],
                        v_res: use_res.then(|| &v_res[span.clone()]),
                        gate: &gate[span.clone()],
                    };
                    crate::cpu_rwkv::wkv7(&params, &mut rec[seq], len, &inputs, &mut want_first[span.clone()], &mut want_out[span.clone()]);
                } else {
                    let inputs = crate::cpu_rwkv::Inputs6 { r: &r[span.clone()], k: &k[span.clone()], v: &v[span.clone()], w: &w[span.clone()], gate: &gate[span.clone()] };
                    crate::cpu_rwkv::wkv6(&params, &mut rec[seq], len, &inputs, &mut want_out[span.clone()]);
                }
                runs.push(DeltaRun { state: &states[seq], first_row: first, rows: len, reset });
                first += len;
            }
            let (r_buf, k_buf, v_buf, w_buf, a_buf, gate_buf, x_buf) = (
                put(self.device, stream.as_mut(), &r)?,
                put(self.device, stream.as_mut(), &k)?,
                put(self.device, stream.as_mut(), &v)?,
                put(self.device, stream.as_mut(), &w)?,
                put(self.device, stream.as_mut(), &a)?,
                put(self.device, stream.as_mut(), &gate)?,
                put(self.device, stream.as_mut(), &x)?,
            );
            let v_res_buf = put(self.device, stream.as_mut(), &v_res)?;
            let delta_buf = put(self.device, stream.as_mut(), &delta)?;
            let mut v_first = put(self.device, stream.as_mut(), &v_first_in)?;
            // poisoned outputs: every element must be written
            let mut out = put(self.device, stream.as_mut(), &vec![f32::NAN; rows * c])?;
            let mut shifted = put(self.device, stream.as_mut(), &vec![f32::NAN; rows * c])?;
            let mut lerped = put(self.device, stream.as_mut(), &vec![f32::NAN; rows * c])?;
            let mut ms = 0.0;
            timed(&mut ms, stream.as_mut(), |s| {
                s.rwkv_shift(c, &runs, offset + shape.shift_offset(), &x_buf, &mut shifted)?;
                s.rwkv_lerp(c, &x_buf, &shifted, &mix_buf, (call_index % 2 == 0).then_some(&delta_buf), &mut lerped)?;
                if shape.v7 {
                    let inputs = Rwkv7Inputs { r: &r_buf, k: &k_buf, v: &v_buf, w: &w_buf, a: &a_buf, v_res: use_res.then_some(&v_res_buf), gate: &gate_buf };
                    s.rwkv7_wkv(shape, &weights, &runs, offset, &inputs, &mut v_first, &mut out)
                } else {
                    s.rwkv6_wkv(shape, &weights, &runs, offset, &r_buf, &k_buf, &v_buf, &w_buf, &gate_buf, &mut out)
                }
            })?;
            total_ms += ms;
            let unit = vec![1.0; rows * c];
            let to64 = |t: &[f32]| t.iter().map(|&v| f64::from(v)).collect::<Vec<f64>>();
            worst_err = worst_err.max(worst(&get(stream.as_mut(), &out, rows * c)?, &to64(&want_out), &unit));
            worst_err = worst_err.max(worst(&get(stream.as_mut(), &shifted, rows * c)?, &to64(&want_shift), &unit));
            let want_lerp: Vec<f64> = (0..rows * c)
                .map(|i| {
                    let m = f64::from(mix[i % c]) + if call_index % 2 == 0 { f64::from(delta[i]) } else { 0.0 };
                    f64::from(x[i]) + (f64::from(want_shift[i]) - f64::from(x[i])) * m
                })
                .collect();
            worst_err = worst_err.max(worst(&get(stream.as_mut(), &lerped, rows * c)?, &want_lerp, &unit));
            if shape.v7 {
                // the values of the run: stored when there is no blend, untouched otherwise
                worst_err = worst_err.max(worst(&get(stream.as_mut(), &v_first, rows * c)?, &to64(&want_first), &unit));
            }
        }
        // the activations, in place, on one batch
        let count = 5 * c + 3;
        let base = self.rng.vector(count, 3.0);
        let gate_vals = self.rng.vector(count, 3.0);
        let gate_buf = put(self.device, stream.as_mut(), &gate_vals)?;
        for mode in 0..=5 {
            let mut buf = put(self.device, stream.as_mut(), &base)?;
            let mut ms = 0.0;
            timed(&mut ms, stream.as_mut(), |s| s.rwkv_act(&mut buf, (mode == 4).then_some(&gate_buf), mode))?;
            total_ms += ms;
            let want: Vec<f64> = base
                .iter()
                .zip(&gate_vals)
                .map(|(&x, &g)| {
                    let (x, g) = (f64::from(x), f64::from(g));
                    match mode {
                        0 => x.tanh(),
                        1 => 1.0 / (1.0 + (-x).exp()),
                        2 => x / (1.0 + (-x).exp()),
                        3 => x.max(0.0) * x.max(0.0),
                        4 => x / (1.0 + (-g).exp()),
                        _ => x * 0.5,
                    }
                })
                .collect();
            worst_err = worst_err.max(worst(&get(stream.as_mut(), &buf, count)?, &want, &vec![1.0; count]));
        }
        Ok(Outcome { device: worst_err, cpu: 0.0, tol: 2e-3, cpu_tol: 1.0, ms: total_ms })
    }
}
