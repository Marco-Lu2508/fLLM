//! Self-test group `actdtype`: the narrow-activation pipeline (`FLLM_ACT_DTYPE=f16|bf16`, docs/activation-dtype.md) against the FP32 pipeline of the same
//! device, kernel by kernel and for a small chain of decoder blocks.
//!
//! Every case feeds both pipelines the same values: the inputs are rounded to the narrow type on the host first, so the FP32 stream sees exactly what the
//! narrow stream stores. The remaining difference is the rounding of the results (and of the intermediate values of a chain) to the narrow type, which is
//! bounded by the unit roundoff of the type, u = 2^-11 for FP16 and 2^-8 for BF16, times the error scale of the element (the sum of the magnitudes of the
//! terms of the result for the rotation and the residual add, the value itself for products). FP16 subnormals have an absolute step of 2^-24: the error
//! scale of an element is floored there (the floor is `floor / u` in the units of the scale). A check passes when `max |narrow - fp32| / scale <= tol`;
//! `tol` is a small multiple of u (stated per case). KV cache writes must match bit for bit.
use super::*;
use crate::backend::act_dtype::ActDtype;

/// The FP32 stream and the narrow stream of one device.
type StreamPair = (Box<dyn Stream>, Box<dyn Stream>);
/// Rotated q, and the keys and values read back from the f16 cache.
type RopeKvOut = (Vec<f32>, Vec<f32>, Vec<f32>);

const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const DIM: usize = 64;
const BLOCK: usize = 16;
const STRIDE: usize = 16;

fn unit_roundoff(dt: ActDtype) -> f64 {
    match dt {
        ActDtype::F16 => 4.8828125e-4, // 2^-11
        ActDtype::Bf16 => 3.90625e-3,  // 2^-8
        ActDtype::F32 => 5.960_464_5e-8,
    }
}

/// Absolute floor of the error scale: the subnormal step of FP16 expressed in units of the scale (so that an error of one subnormal step is within u).
fn scale_floor(dt: ActDtype) -> f64 {
    match dt {
        ActDtype::F16 => 5.960_464_5e-8 / unit_roundoff(dt), // 2^-24 / 2^-11
        _ => 1e-30,
    }
}

fn round_to(dt: ActDtype, v: f32) -> f32 {
    match dt {
        ActDtype::F16 => f16::from_f32(v).to_f32(),
        ActDtype::Bf16 => bf16::from_f32(v).to_f32(),
        ActDtype::F32 => v,
    }
}

fn round_vec(dt: ActDtype, values: &[f32]) -> Vec<f32> {
    values.iter().map(|&v| round_to(dt, v)).collect()
}

fn outcome(device: f64, tol: f64, ms: f64) -> Outcome {
    Outcome { device, cpu: 0.0, tol, cpu_tol: 1.0, ms }
}

/// A buffer holding `values` in the activation type of `stream`.
fn put_act(backend: &dyn Backend, stream: &mut dyn Stream, values: &[f32]) -> Result<DeviceBuffer, BackendError> {
    let mut buf = backend.allocate_buffer(values.len().max(1) * 4)?;
    stream.upload_activations(&mut buf, values)?;
    Ok(buf)
}

/// The first `count` values of an activation buffer as f32 values.
fn get_act(backend: &dyn Backend, stream: &mut dyn Stream, buf: &DeviceBuffer, count: usize) -> Result<Vec<f32>, BackendError> {
    if stream.act_dtype().is_narrow() {
        let mut wide = backend.allocate_buffer(buf.size)?;
        stream.widen_activations(&mut wide, buf)?;
        get(stream, &wide, count)
    } else {
        get(stream, buf, count)
    }
}

fn as_f64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

/// `scale = |ref| + floor` per element
fn magnitude_scale(dt: ActDtype, reference: &[f32]) -> Vec<f64> {
    reference.iter().map(|&v| f64::from(v).abs() + scale_floor(dt)).collect()
}

impl Tester<'_> {
    pub(super) fn actdtype_cases(&mut self) {
        for dt in [ActDtype::F16, ActDtype::Bf16] {
            let tag = dt.name();
            let supported = self.device.set_activation_dtype(dt);
            self.device.set_activation_dtype(ActDtype::F32);
            if !supported {
                self.record_skip(&format!("actdtype {tag}"), "all", "the device has no kernels for this activation type");
                continue;
            }
            for (rows, cols) in [(1usize, 512usize), (3, 4096), (40, 2560), (2, 100), (5, 8192), (4, 5120), (1, 11008)] {
                let r = self.act_norm(dt, rows, cols, false);
                self.record(&format!("actdtype norm {tag}"), &format!("{rows}x{cols}"), r);
            }
            for (rows, cols) in [(1usize, 4096usize), (7, 2048), (40, 512), (3, 100)] {
                let r = self.act_norm(dt, rows, cols, true);
                self.record(&format!("actdtype add+norm {tag}"), &format!("{rows}x{cols}"), r);
            }
            for count in [4096usize, 11008, 8195] {
                let r = self.act_residual(dt, count);
                self.record(&format!("actdtype residual {tag}"), &format!("{count} values"), r);
            }
            for (count, geglu) in [(11008usize, false), (4096, true), (1005, false)] {
                let r = self.act_glu(dt, count, geglu);
                self.record(&format!("actdtype glu {tag}"), &format!("{count} values {}", if geglu { "gelu" } else { "silu" }), r);
            }
            for (rope_dim, neox) in [(64usize, true), (64, false), (32, true), (32, false)] {
                let r = self.act_rope(dt, rope_dim, neox);
                self.record(&format!("actdtype rope {tag}"), &format!("rows 5, head 64, rotary {rope_dim}, {}", if neox { "neox" } else { "pairs" }), r);
            }
            for rows in [1usize, 6, 33] {
                let r = self.act_kv_write(dt, rows);
                self.record(&format!("actdtype kv write {tag}"), &format!("{rows} rows"), r);
            }
            for (rows, rope_dim, neox) in [(1usize, 64usize, true), (6, 64, false), (33, 32, true)] {
                let r = self.act_rope_kv(dt, rows, rope_dim, neox, false);
                self.record(&format!("actdtype rope+kv {tag}"), &format!("{rows} rows, rotary {rope_dim}, {}", if neox { "neox" } else { "pairs" }), r);
            }
            for (rows, rope_dim, neox) in [(1usize, 64usize, true), (6, 64, false), (33, 32, true)] {
                let r = self.act_rope_kv(dt, rows, rope_dim, neox, true);
                self.record(&format!("actdtype qknorm+rope+kv {tag}"), &format!("{rows} rows, rotary {rope_dim}, {}", if neox { "neox" } else { "pairs" }), r);
            }
            for kind in self.kinds(&[0x8000_0010u32, 0x8000_0011, 0x8000_0012]) {
                for batch in [1usize, 2, 5, 40] {
                    let r = self.act_linear(dt, kind, 384, 1024, batch, batch == 5);
                    self.record(&format!("actdtype marlin {tag} {}", format_name(kind)), &format!("384x1024 batch {batch}{}", if batch == 5 { " +bias" } else { "" }), r);
                }
                let r = self.act_linear_add(dt, kind, 256, 512, 3);
                self.record(&format!("actdtype marlin add {tag} {}", format_name(kind)), "256x512 batch 3, residual scale 0.5", r);
            }
            // formats without a narrow kernel: K-quant GEMVs read the narrow activations through the int8 quantization of the integer dot-product kernels, the others
            // run on FP32 copies (the batch sizes cover the GEMV, small-tile and tensor-core kernels)
            for kind in self.kinds(&[12u32, 14, 8, 2]) {
                for batch in [1usize, 2, 4, 40] {
                    let r = self.act_linear(dt, kind, 259, 1024, batch, batch == 2);
                    self.record(&format!("actdtype linear {tag} {}", format_name(kind)), &format!("259x1024 batch {batch}{}", if batch == 2 { " +bias" } else { "" }), r);
                }
            }
            for rows in [1usize, 4, 40] {
                let r = self.act_ffn_chain(dt, rows);
                self.record(&format!("actdtype chain {tag}"), &format!("2 FFN blocks, hidden 512, ff 1024, {rows} rows"), r);
            }
            for runs in [&[48usize][..], &[3][..]] {
                let r = self.act_attention(dt, runs);
                self.record(&format!("actdtype attention {tag}"), &format!("{runs:?} rows, f16 cache"), r);
            }
            let r = self.act_check(dt);
            self.record(&format!("actdtype check {tag}"), "max |x| and NaN/Inf count", r);
            for rows in [1usize, 3, 40] {
                let r = self.act_merged(dt, rows, false);
                self.record(&format!("actdtype merged qkv {tag}"), &format!("q 256 + k 128 + v 128 from 512, {rows} rows"), r);
                let r = self.act_merged(dt, rows, true);
                self.record(&format!("actdtype merged qkv+rope {tag}"), &format!("{rows} rows"), r);
                let r = self.act_merged_gate_up(dt, rows);
                self.record(&format!("actdtype merged gate/up {tag}"), &format!("2 x 1024 from 512, {rows} rows"), r);
            }
            for rows in [8usize, 200] {
                let r = self.act_pdl(dt, rows);
                self.record(&format!("actdtype pdl {tag}"), &format!("norm, glu, residual, {rows} rows: FLLM_PDL on = off"), r);
            }
        }
    }

    /// Both streams; the FP32 one first (the backend is switched back to FP32 afterwards).
    fn act_streams(&self, dt: ActDtype) -> Result<StreamPair, BackendError> {
        self.device.set_activation_dtype(ActDtype::F32);
        let wide = self.device.create_stream()?;
        if !self.device.set_activation_dtype(dt) {
            return Err(BackendError::UnsupportedOperation(format!("{dt} activations")));
        }
        let narrow = self.device.create_stream();
        self.device.set_activation_dtype(ActDtype::F32);
        Ok((wide, narrow?))
    }

    /// RMS norm, optionally behind the residual add (`x += 0.75 * branch`): result and updated stream against the FP32 pipeline.
    fn act_norm(&mut self, dt: ActDtype, rows: usize, cols: usize, add: bool) -> Result<Outcome, BackendError> {
        let x = round_vec(dt, &self.rng.vector(rows * cols, 3.0));
        let branch = round_vec(dt, &self.rng.vector(rows * cols, 2.0));
        let w: Vec<f32> = self.rng.vector(cols, 0.5).iter().map(|v| v + 1.0).collect();
        let (mut wide, mut narrow) = self.act_streams(dt)?;
        let device = self.device;
        let run = |stream: &mut dyn Stream| -> Result<(Vec<f32>, Vec<f32>), BackendError> {
            let mut xb = put_act(device, stream, &x)?;
            let nw = put(device, stream, &w)?;
            let mut out = device.allocate_buffer(rows * cols * 4)?;
            if add {
                let br = put_act(device, stream, &branch)?;
                if !stream.add_rms_norm(&mut xb, &br, 0.75, &nw, 1e-6, &mut out, None)? {
                    stream.residual_add(&mut xb, &br, 0.75)?;
                    stream.rms_norm(&mut out, &xb, &nw, 1e-6)?;
                }
            } else {
                stream.rms_norm(&mut out, &xb, &nw, 1e-6)?;
            }
            Ok((get_act(device, stream, &out, rows * cols)?, get_act(device, stream, &xb, rows * cols)?))
        };
        let (want_out, want_res) = run(wide.as_mut())?;
        let start = Instant::now();
        let (got_out, got_res) = run(narrow.as_mut())?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        let u = unit_roundoff(dt);
        // out: one rounding of the result (add: also the rounded residual, which the FP32 pipeline does not have: two roundings)
        let err_out = worst(&got_out, &as_f64(&want_out), &magnitude_scale(dt, &want_out));
        let mut err = err_out / (if add { 2.0 } else { 1.0 } * u);
        if add {
            // the residual itself: one rounding of x + 0.75 * branch, scale |x| + |0.75 branch|
            let scale: Vec<f64> = x.iter().zip(&branch).map(|(&a, &b)| f64::from(a).abs() + 0.75 * f64::from(b).abs() + scale_floor(dt)).collect();
            err = err.max(worst(&got_res, &as_f64(&want_res), &scale) / u);
        }
        Ok(outcome(err, 1.5, ms))
    }

    /// `x += scale * res` on both streams.
    fn act_residual(&mut self, dt: ActDtype, count: usize) -> Result<Outcome, BackendError> {
        let x = round_vec(dt, &self.rng.vector(count, 3.0));
        let r = round_vec(dt, &self.rng.vector(count, 2.0));
        let (mut wide, mut narrow) = self.act_streams(dt)?;
        let device = self.device;
        let run = |stream: &mut dyn Stream| -> Result<Vec<f32>, BackendError> {
            let mut xb = put_act(device, stream, &x)?;
            let rb = put_act(device, stream, &r)?;
            stream.residual_add(&mut xb, &rb, -0.5)?;
            get_act(device, stream, &xb, count)
        };
        let want = run(wide.as_mut())?;
        let start = Instant::now();
        let got = run(narrow.as_mut())?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        let scale: Vec<f64> = x.iter().zip(&r).map(|(&a, &b)| f64::from(a).abs() + 0.5 * f64::from(b).abs() + scale_floor(dt)).collect();
        let u = unit_roundoff(dt);
        Ok(outcome(worst(&got, &as_f64(&want), &scale) / u, 1.5, ms))
    }

    fn act_glu(&mut self, dt: ActDtype, count: usize, geglu: bool) -> Result<Outcome, BackendError> {
        let g = round_vec(dt, &self.rng.vector(count, 4.0));
        let u_vals = round_vec(dt, &self.rng.vector(count, 2.0));
        let (mut wide, mut narrow) = self.act_streams(dt)?;
        let device = self.device;
        let run = |stream: &mut dyn Stream| -> Result<Vec<f32>, BackendError> {
            let gb = put_act(device, stream, &g)?;
            let ub = put_act(device, stream, &u_vals)?;
            let mut out = device.allocate_buffer(count * 4)?;
            if geglu {
                stream.geglu(&mut out, &gb, &ub)?;
            } else {
                stream.swiglu(&mut out, &gb, &ub)?;
            }
            get_act(device, stream, &out, count)
        };
        let want = run(wide.as_mut())?;
        let start = Instant::now();
        let got = run(narrow.as_mut())?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        let u = unit_roundoff(dt);
        Ok(outcome(worst(&got, &as_f64(&want), &magnitude_scale(dt, &want)) / u, 1.5, ms))
    }

    fn rope_params(rope_dim: usize, neox: bool) -> RopeParams {
        let half = rope_dim / 2;
        let positions = 64usize;
        let (mut cos, mut sin) = (Vec::with_capacity(positions * half), Vec::with_capacity(positions * half));
        for p in 0..positions {
            for d in 0..half {
                let angle = (p as f32 + 1.0) * 0.013 * (d as f32 + 1.0).sqrt();
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        RopeParams { heads: HEADS, kv_heads: KV_HEADS, dim: DIM, rope_dim, neox, cos: Arc::new(cos), sin: Arc::new(sin) }
    }

    /// Error scale of the rotated values: |a c| + |b s| for the first of a pair, |b c| + |a s| for the second, |x| for the unrotated tail.
    fn rope_scale(dt: ActDtype, x: &[f32], positions: &[usize], heads: usize, params: &RopeParams) -> Vec<f64> {
        let half = params.rope_dim / 2;
        let mut scale: Vec<f64> = x.iter().map(|&v| f64::from(v).abs() + scale_floor(dt)).collect();
        for (r, &p) in positions.iter().enumerate() {
            for h in 0..heads {
                let base = (r * heads + h) * DIM;
                for d in 0..half {
                    let (ia, ib) = if params.neox { (d, d + half) } else { (2 * d, 2 * d + 1) };
                    let (c, s) = (f64::from(params.cos[p * half + d]), f64::from(params.sin[p * half + d]));
                    let (a, b) = (f64::from(x[base + ia]).abs(), f64::from(x[base + ib]).abs());
                    scale[base + ia] = a * c.abs() + b * s.abs() + scale_floor(dt);
                    scale[base + ib] = b * c.abs() + a * s.abs() + scale_floor(dt);
                }
            }
        }
        scale
    }

    fn act_rope(&mut self, dt: ActDtype, rope_dim: usize, neox: bool) -> Result<Outcome, BackendError> {
        let positions = vec![0usize, 3, 7, 20, 31];
        let rows = positions.len();
        let q = round_vec(dt, &self.rng.vector(rows * HEADS * DIM, 2.0));
        let k = round_vec(dt, &self.rng.vector(rows * KV_HEADS * DIM, 2.0));
        let params = Self::rope_params(rope_dim, neox);
        let (mut wide, mut narrow) = self.act_streams(dt)?;
        let device = self.device;
        let run = |stream: &mut dyn Stream| -> Result<Vec<f32>, BackendError> {
            let mut qb = put_act(device, stream, &q)?;
            let mut kb = put_act(device, stream, &k)?;
            stream.rope(&mut qb, &mut kb, &positions, &params)?;
            let mut all = get_act(device, stream, &qb, q.len())?;
            all.extend(get_act(device, stream, &kb, k.len())?);
            Ok(all)
        };
        let want = run(wide.as_mut())?;
        let start = Instant::now();
        let got = run(narrow.as_mut())?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        let mut scale = Self::rope_scale(dt, &q, &positions, HEADS, &params);
        scale.extend(Self::rope_scale(dt, &k, &positions, KV_HEADS, &params));
        let u = unit_roundoff(dt);
        Ok(outcome(worst(&got, &as_f64(&want), &scale) / u, 1.5, ms))
    }

    /// Contents of the f16 pools of layer 0 as bytes.
    fn pool_bytes(stream: &mut dyn Stream, cache: &DeviceKvCache) -> Result<Vec<u8>, BackendError> {
        let mut all = vec![0u8; cache.key_pool.size];
        stream.download(&mut all, &cache.key_pool)?;
        let mut v = vec![0u8; cache.value_pool.size];
        stream.download(&mut v, &cache.value_pool)?;
        all.extend(v);
        Ok(all)
    }

    fn kv_layout(rows: usize) -> (Vec<usize>, usize) {
        // one sequence, positions 0 .. rows
        let slots: Vec<usize> = (0..rows).collect();
        (slots, rows.div_ceil(BLOCK).max(1) + 1)
    }

    /// The cache write of narrow keys and values: the f16 words must be those the FP32 pipeline writes (a narrow value converts to f16 exactly when in range).
    fn act_kv_write(&mut self, dt: ActDtype, rows: usize) -> Result<Outcome, BackendError> {
        let k = round_vec(dt, &self.rng.vector(rows * KV_HEADS * DIM, 2.0));
        let v = round_vec(dt, &self.rng.vector(rows * KV_HEADS * DIM, 2.0));
        let (slots, blocks) = Self::kv_layout(rows);
        let (mut wide, mut narrow) = self.act_streams(dt)?;
        let device = self.device;
        let run = |stream: &mut dyn Stream| -> Result<Vec<u8>, BackendError> {
            let cache = device.allocate_kv_cache(1, blocks, BLOCK, KV_HEADS, DIM, DIM, 1)?;
            let kb = put_act(device, stream, &k)?;
            let vb = put_act(device, stream, &v)?;
            stream.kv_cache_write(&cache, 0, &slots, &kb, &vb)?;
            Self::pool_bytes(stream, &cache)
        };
        let want = run(wide.as_mut())?;
        let start = Instant::now();
        let got = run(narrow.as_mut())?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        // only the written slots are defined
        let used = rows * KV_HEADS * DIM * 2;
        let half = want.len() / 2;
        let differing = (0..used).filter(|&i| want[i] != got[i]).count() + (0..used).filter(|&i| want[half + i] != got[half + i]).count();
        Ok(outcome(differing as f64, 0.0, ms))
    }

    /// Rotation + cache write in one operation; with `norm` also the per-head RMS norm of q and k in front. Rotated q within a few units of u; the cache
    /// words as f16 values within two f16 units.
    fn act_rope_kv(&mut self, dt: ActDtype, rows: usize, rope_dim: usize, neox: bool, norm: bool) -> Result<Outcome, BackendError> {
        let q = round_vec(dt, &self.rng.vector(rows * HEADS * DIM, 2.0));
        let k = round_vec(dt, &self.rng.vector(rows * KV_HEADS * DIM, 2.0));
        let v = round_vec(dt, &self.rng.vector(rows * KV_HEADS * DIM, 2.0));
        let qw: Vec<f32> = self.rng.vector(DIM, 0.4).iter().map(|x| x + 1.0).collect();
        let kw: Vec<f32> = self.rng.vector(DIM, 0.4).iter().map(|x| x + 1.0).collect();
        let positions: Vec<usize> = (0..rows).map(|r| r * 2 + 1).collect();
        let (slots, blocks) = Self::kv_layout(rows);
        let params = Self::rope_params(rope_dim, neox);
        let eps = 1e-6f32;
        // f64 reference: norm per head, then the rotation
        let normed = |x: &[f32], heads: usize, w: &[f32]| -> Vec<f32> {
            let mut out = x.to_vec();
            if norm {
                for head in out.chunks_mut(DIM).take(rows * heads) {
                    let ms: f64 = head.iter().map(|&t| f64::from(t) * f64::from(t)).sum::<f64>() / DIM as f64;
                    let inv = 1.0 / (ms + f64::from(eps)).sqrt();
                    for (t, &g) in head.iter_mut().zip(w) {
                        *t = (f64::from(*t) * inv * f64::from(g)) as f32;
                    }
                }
            }
            out
        };
        let (qn, kn) = (normed(&q, HEADS, &qw), normed(&k, KV_HEADS, &kw));
        let half = rope_dim / 2;
        let rotate = |x: &[f32], heads: usize| -> Vec<f64> {
            let mut out = as_f64(x);
            for (r, &p) in positions.iter().enumerate() {
                for h in 0..heads {
                    let base = (r * heads + h) * DIM;
                    for d in 0..half {
                        let (ia, ib) = if neox { (d, d + half) } else { (2 * d, 2 * d + 1) };
                        let (c, s) = (f64::from(params.cos[p * half + d]), f64::from(params.sin[p * half + d]));
                        let (a, b) = (f64::from(x[base + ia]), f64::from(x[base + ib]));
                        out[base + ia] = a * c - b * s;
                        out[base + ib] = b * c + a * s;
                    }
                }
            }
            out
        };
        let (want_q, want_k) = (rotate(&qn, HEADS), rotate(&kn, KV_HEADS));
        let (scale_q, scale_k) = (Self::rope_scale(dt, &qn, &positions, HEADS, &params), Self::rope_scale(dt, &kn, &positions, KV_HEADS, &params));
        let (_, mut narrow) = self.act_streams(dt)?;
        let device = self.device;
        let run = |stream: &mut dyn Stream| -> Result<Option<RopeKvOut>, BackendError> {
            let cache = device.allocate_kv_cache(1, blocks, BLOCK, KV_HEADS, DIM, DIM, 1)?;
            let mut qb = put_act(device, stream, &q)?;
            let kb = put_act(device, stream, &k)?;
            let vb = put_act(device, stream, &v)?;
            let qwb = put(device, stream, &qw)?;
            let kwb = put(device, stream, &kw)?;
            let done = if norm {
                stream.rope_kv_write_qknorm(&mut qb, &kb, &vb, Some(&qwb), Some(&kwb), eps, &cache, 0, &positions, &slots, &params)?
            } else {
                stream.rope_kv_write(&mut qb, &kb, &vb, &cache, 0, &positions, &slots, &params)?
            };
            if !done {
                return Ok(None);
            }
            let bytes = Self::pool_bytes(stream, &cache)?;
            let half_len = bytes.len() / 2;
            let words = |b: &[u8], n: usize| -> Vec<f32> { b.as_chunks::<2>().0.iter().take(n).map(|c| f16::from_le_bytes(*c).to_f32()).collect() };
            Ok(Some((get_act(device, stream, &qb, q.len())?, words(&bytes[..half_len], rows * KV_HEADS * DIM), words(&bytes[half_len..], rows * KV_HEADS * DIM))))
        };
        let start = Instant::now();
        let Some((got_q, got_k, got_v)) = run(narrow.as_mut())? else {
            return Err(BackendError::UnsupportedOperation(if norm { "rope_kv_write_qknorm" } else { "rope_kv_write" }.into()));
        };
        let ms = start.elapsed().as_secs_f64() * 1e3;
        let u = unit_roundoff(dt);
        let u16_ = unit_roundoff(ActDtype::F16);
        // q: result rounded to the narrow type (+ the norm's FP32 arithmetic); k: rounded to f16 (<= 2 units of the f16 cache plus the f32 arithmetic)
        let err_q = worst(&got_q, &want_q, &scale_q) / u;
        let err_k = worst(&got_k, &want_k, &scale_k) / u16_;
        let err_v = worst(&got_v, &as_f64(&v), &magnitude_scale(ActDtype::F16, &v)) / u16_;
        // bf16 values beyond f16 range do not occur in the test data, so v is exact up to the f16 rounding of a bf16 value (none: 8 bits fit)
        Ok(outcome(err_q.max(err_k).max(err_v), 3.0, ms))
    }

    /// The Marlin 4-bit GEMM on narrow activations against the f64 reference (weights and scales as the narrow pipeline stores them).
    fn act_linear(&mut self, dt: ActDtype, kind: u32, rows: usize, cols: usize, batch: usize, with_bias: bool) -> Result<Outcome, BackendError> {
        self.claim = Some(kind);
        let weights = random_kind(&mut self.rng, kind, rows, cols);
        let x = round_vec(dt, &self.rng.vector(batch * cols, 1.0));
        let bias = with_bias.then(|| self.rng.vector(rows, 1.0));
        let w = crate::compute::dequantize(&weights, kind, rows * cols).map_err(BackendError::DriverError)?;
        let mut want = vec![0.0f64; batch * rows];
        let mut scale = vec![0.0f64; batch * rows];
        for b in 0..batch {
            for r in 0..rows {
                let (mut sum, mut mag) = (0.0f64, 0.0f64);
                for (&xv, &wv) in x[b * cols..(b + 1) * cols].iter().zip(&w[r * cols..(r + 1) * cols]) {
                    let term = f64::from(xv) * f64::from(wv);
                    sum += term;
                    mag += term.abs();
                }
                if let Some(bias) = &bias {
                    sum += f64::from(bias[r]);
                    mag += f64::from(bias[r]).abs();
                }
                want[b * rows + r] = sum;
                scale[b * rows + r] = mag + scale_floor(dt);
            }
        }
        self.device.set_activation_dtype(dt);
        let weight = self.device.upload_weight_scaled("selftest", &weights, QuantType::from_gguf_type_id(kind), rows, cols, bias.as_deref(), 1.0);
        let stream = self.device.create_stream();
        self.device.set_activation_dtype(ActDtype::F32);
        let (weight, mut stream) = (weight?, stream?);
        let xb = put_act(self.device, stream.as_mut(), &x)?;
        let mut out = self.device.allocate_buffer(batch * rows * 4)?;
        let start = Instant::now();
        stream.quantized_linear(&mut out, &xb, &weight, weight.bias.as_ref())?;
        stream.synchronize()?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        let got = get_act(self.device, stream.as_mut(), &out, batch * rows)?;
        // Marlin: output rounding u + the group scales in the narrow type (BF16 scales: 2^-9 per scale) + fp32 accumulation: 2u for FP16, 3u for BF16.
        // Other formats run on FP32 copies or on int8 activations (the FP32 pipeline holds them to FAST_TOL), plus the rounding of the narrow result.
        let u = unit_roundoff(dt);
        let tol = if QuantType::from_gguf_type_id(kind).q4z_group().is_none() {
            Self::FAST_TOL + 2.0 * u
        } else if dt == ActDtype::Bf16 {
            3.0 * u
        } else {
            Self::FAST_TOL
        };
        Ok(outcome(worst(&got, &want, &scale), tol, ms))
    }

    /// `res += 0.5 * (W x)` through `quantized_linear_add` on narrow activations.
    fn act_linear_add(&mut self, dt: ActDtype, kind: u32, rows: usize, cols: usize, batch: usize) -> Result<Outcome, BackendError> {
        self.claim = Some(kind);
        let weights = random_kind(&mut self.rng, kind, rows, cols);
        let x = round_vec(dt, &self.rng.vector(batch * cols, 1.0));
        let res0 = round_vec(dt, &self.rng.vector(batch * rows, 1.0));
        let w = crate::compute::dequantize(&weights, kind, rows * cols).map_err(BackendError::DriverError)?;
        let mut want = vec![0.0f64; batch * rows];
        let mut scale = vec![0.0f64; batch * rows];
        for b in 0..batch {
            for r in 0..rows {
                let (mut sum, mut mag) = (0.0f64, 0.0f64);
                for (&xv, &wv) in x[b * cols..(b + 1) * cols].iter().zip(&w[r * cols..(r + 1) * cols]) {
                    let term = 0.5 * f64::from(xv) * f64::from(wv);
                    sum += term;
                    mag += term.abs();
                }
                want[b * rows + r] = f64::from(res0[b * rows + r]) + sum;
                scale[b * rows + r] = f64::from(res0[b * rows + r]).abs() + mag + scale_floor(dt);
            }
        }
        self.device.set_activation_dtype(dt);
        let weight = self.device.upload_weight_scaled("selftest", &weights, QuantType::from_gguf_type_id(kind), rows, cols, None, 1.0);
        let stream = self.device.create_stream();
        self.device.set_activation_dtype(ActDtype::F32);
        let (weight, mut stream) = (weight?, stream?);
        let xb = put_act(self.device, stream.as_mut(), &x)?;
        let mut res = put_act(self.device, stream.as_mut(), &res0)?;
        let start = Instant::now();
        if !stream.quantized_linear_add(&mut res, &xb, &weight, None, 0.5)? {
            // the caller's fallback: linear + residual add
            let mut tmp = self.device.allocate_buffer(batch * rows * 4)?;
            stream.quantized_linear(&mut tmp, &xb, &weight, None)?;
            stream.residual_add(&mut res, &tmp, 0.5)?;
        }
        stream.synchronize()?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        let got = get_act(self.device, stream.as_mut(), &res, batch * rows)?;
        let u = unit_roundoff(dt);
        let tol = if dt == ActDtype::Bf16 { 3.0 * u } else { Self::FAST_TOL };
        Ok(outcome(worst(&got, &want, &scale), tol, ms))
    }

    /// Two feed-forward blocks (norm, gate and up GEMMs, SwiGLU, down GEMM added to the residual stream) on the narrow stream against the same blocks on
    /// the FP32 stream (each with the weights uploaded for its own type). Error: largest difference over the RMS of the FP32 result; the bound is the
    /// accumulated rounding of about eight narrow roundings (random data adds up like a random walk): 12 u.
    fn act_ffn_chain(&mut self, dt: ActDtype, rows: usize) -> Result<Outcome, BackendError> {
        let (hidden, ff) = (512usize, 1024usize);
        let kind = 0x8000_0012u32;
        self.claim = Some(kind);
        let mk = |rng: &mut Lcg, n: usize, m: usize| random_kind(rng, kind, n, m);
        let blocks: Vec<[Vec<u8>; 3]> = (0..2).map(|_| [mk(&mut self.rng, ff, hidden), mk(&mut self.rng, ff, hidden), mk(&mut self.rng, hidden, ff)]).collect();
        let norms: Vec<Vec<f32>> = (0..2).map(|_| self.rng.vector(hidden, 0.3).iter().map(|v| v + 1.0).collect()).collect();
        let x0 = round_vec(dt, &self.rng.vector(rows * hidden, 1.0));
        let device = self.device;
        let qt = QuantType::from_gguf_type_id(kind);
        let run = |narrow: bool| -> Result<Vec<f32>, BackendError> {
            device.set_activation_dtype(if narrow { dt } else { ActDtype::F32 });
            let weights: Result<Vec<[DeviceWeight; 3]>, BackendError> = blocks
                .iter()
                .map(|b| {
                    Ok([
                        device.upload_weight_scaled("selftest", &b[0], qt, ff, hidden, None, 1.0)?,
                        device.upload_weight_scaled("selftest", &b[1], qt, ff, hidden, None, 1.0)?,
                        device.upload_weight_scaled("selftest", &b[2], qt, hidden, ff, None, 1.0)?,
                    ])
                })
                .collect();
            let stream = device.create_stream();
            device.set_activation_dtype(ActDtype::F32);
            let (weights, mut stream) = (weights?, stream?);
            let stream = stream.as_mut();
            let mut x = put_act(device, stream, &x0)?;
            let mut keep = Vec::new();
            for (block, norm) in weights.iter().zip(&norms) {
                let nw = put(device, stream, norm)?;
                let mut normed = device.allocate_buffer(rows * hidden * 4)?;
                let (mut gate, mut up, mut act) = (device.allocate_buffer(rows * ff * 4)?, device.allocate_buffer(rows * ff * 4)?, device.allocate_buffer(rows * ff * 4)?);
                stream.rms_norm(&mut normed, &x, &nw, 1e-6)?;
                stream.quantized_linear(&mut gate, &normed, &block[0], None)?;
                stream.quantized_linear(&mut up, &normed, &block[1], None)?;
                stream.swiglu(&mut act, &gate, &up)?;
                if !stream.quantized_linear_add(&mut x, &act, &block[2], None, 1.0)? {
                    let mut out = device.allocate_buffer(rows * hidden * 4)?;
                    stream.quantized_linear(&mut out, &act, &block[2], None)?;
                    stream.residual_add(&mut x, &out, 1.0)?;
                    keep.push(out);
                }
                keep.extend([nw, normed, gate, up, act]);
            }
            let result = get_act(device, stream, &x, rows * hidden);
            drop(keep);
            result
        };
        let want = run(false)?;
        let start = Instant::now();
        let got = run(true)?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        let rms = (want.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / want.len() as f64).sqrt().max(1e-9);
        let scale = vec![rms; want.len()];
        let u = unit_roundoff(dt);
        Ok(outcome(worst(&got, &as_f64(&want), &scale) / u, 12.0, ms))
    }

    /// Attention with narrow q and output over a f16 paged cache (prompt rows of one sequence, or a few decode-style rows).
    fn act_attention(&mut self, dt: ActDtype, runs: &[usize]) -> Result<Outcome, BackendError> {
        let (mut slots, mut tables, mut lengths) = (Vec::new(), Vec::new(), Vec::new());
        for (s, &n) in runs.iter().enumerate() {
            let table: Vec<i64> = (0..STRIDE).map(|i| if i * BLOCK < n { (s * STRIDE + i) as i64 } else { -1 }).collect();
            for p in 0..n {
                slots.push(table[p / BLOCK] as usize * BLOCK + p % BLOCK);
                tables.extend_from_slice(&table);
                lengths.push((p + 1) as i64);
            }
        }
        let rows = lengths.len();
        let k = round_vec(dt, &self.rng.vector(rows * KV_HEADS * DIM, 1.0));
        let v = round_vec(dt, &self.rng.vector(rows * KV_HEADS * DIM, 1.0));
        let q = round_vec(dt, &self.rng.vector(rows * HEADS * DIM, 1.0));
        let pool_blocks = runs.len() * STRIDE;
        let params = AttentionParams {
            num_heads: HEADS,
            num_kv_heads: KV_HEADS,
            head_dim: DIM,
            value_dim: DIM,
            stride: STRIDE,
            block_size: BLOCK,
            scale: 1.0 / (DIM as f32).sqrt(),
            softcap: None,
            window: None,
            chunk: None,
            sinks: None,
            alibi_slopes: None,
        };
        let (mut wide, mut narrow) = self.act_streams(dt)?;
        let device = self.device;
        let run = |stream: &mut dyn Stream| -> Result<Vec<f32>, BackendError> {
            let cache = device.allocate_kv_cache(1, pool_blocks, BLOCK, KV_HEADS, DIM, DIM, 1)?;
            let kb = put_act(device, stream, &k)?;
            let vb = put_act(device, stream, &v)?;
            let qb = put_act(device, stream, &q)?;
            stream.kv_cache_write(&cache, 0, &slots, &kb, &vb)?;
            let mut out = device.allocate_buffer(rows * HEADS * DIM * 4)?;
            stream.begin_step();
            let r = stream.attention_paged(&mut out, &qb, &cache, &tables, &lengths, 0, &params);
            stream.end_step();
            r?;
            get_act(device, stream, &out, rows * HEADS * DIM)
        };
        let want = run(wide.as_mut())?;
        let start = Instant::now();
        let got = run(narrow.as_mut())?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        let rms = (want.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>() / want.len() as f64).sqrt().max(1e-9);
        let scale = vec![rms; want.len()];
        // the narrow output rounding, plus the FP16 output of the FlashInfer prompt kernel in the FP32 reference
        let u = unit_roundoff(dt) + unit_roundoff(ActDtype::F16);
        Ok(outcome(worst(&got, &as_f64(&want), &scale), 2.0 * u + 1e-4, ms))
    }

    /// `check_activations`: largest finite |value| and the NaN / Inf count of a buffer with two injected non-finite values.
    fn act_check(&mut self, dt: ActDtype) -> Result<Outcome, BackendError> {
        let mut x = round_vec(dt, &self.rng.vector(5000, 100.0));
        x[17] = 123.5;
        x[100] = f32::INFINITY;
        x[4999] = f32::NAN;
        let (_, mut narrow) = self.act_streams(dt)?;
        let xb = put_act(self.device, narrow.as_mut(), &x)?;
        let Some((max_abs, bad)) = narrow.check_activations(&xb, x.len())? else {
            return Err(BackendError::UnsupportedOperation("check_activations".into()));
        };
        let want_max = x.iter().filter(|v| v.is_finite()).fold(0f32, |m, v| m.max(v.abs()));
        let ok = (max_abs - want_max).abs() <= 1e-6 * want_max && bad == 2;
        Ok(outcome(if ok { 0.0 } else { 1.0 }, 0.0, 0.0))
    }

    /// Raw rows of three Marlin matrices one after the other (what the decoder uploads as one merged matrix).
    fn merged_weights(&mut self, kind: u32, rows: [usize; 3], cols: usize) -> ([Vec<u8>; 3], Vec<u8>) {
        let parts = [random_kind(&mut self.rng, kind, rows[0], cols), random_kind(&mut self.rng, kind, rows[1], cols), random_kind(&mut self.rng, kind, rows[2], cols)];
        let merged: Vec<u8> = parts.iter().flat_map(|p| p.iter().copied()).collect();
        (parts, merged)
    }

    /// q/k/v from one merged Marlin GEMM against three separate ones, on the narrow stream: the blocks (split, or rotated and cached by the fused kernel) must
    /// agree within the rounding of the narrow type (the reduction order of the two GEMM shapes differs).
    fn act_merged(&mut self, dt: ActDtype, batch: usize, with_rope: bool) -> Result<Outcome, BackendError> {
        let kind = 0x8000_0012u32;
        self.claim = Some(kind);
        let (nq, nk, nv, cols) = (HEADS * DIM, KV_HEADS * DIM, KV_HEADS * DIM, 512usize);
        let (parts, merged) = self.merged_weights(kind, [nq, nk, nv], cols);
        let x = round_vec(dt, &self.rng.vector(batch * cols, 1.0));
        let qt = QuantType::from_gguf_type_id(kind);
        self.device.set_activation_dtype(dt);
        let uploads = (|| -> Result<(Vec<DeviceWeight>, DeviceWeight), BackendError> {
            let separate = vec![
                self.device.upload_weight("selftest q", &parts[0], qt, nq, cols, None)?,
                self.device.upload_weight("selftest k", &parts[1], qt, nk, cols, None)?,
                self.device.upload_weight("selftest v", &parts[2], qt, nv, cols, None)?,
            ];
            Ok((separate, self.device.upload_weight("selftest qkv", &merged, qt, nq + nk + nv, cols, None)?))
        })();
        let stream = self.device.create_stream();
        self.device.set_activation_dtype(ActDtype::F32);
        let ((separate, merged_w), mut stream) = (uploads?, stream?);
        let stream = stream.as_mut();
        let device = self.device;
        let xb = put_act(device, stream, &x)?;
        let mut sep = [device.allocate_buffer(batch * nq * 4)?, device.allocate_buffer(batch * nk * 4)?, device.allocate_buffer(batch * nv * 4)?];
        for (out, w) in sep.iter_mut().zip(&separate) {
            stream.quantized_linear(out, &xb, w, None)?;
        }
        let mut qkv = device.allocate_buffer(batch * (nq + nk + nv) * 4)?;
        let start = Instant::now();
        stream.quantized_linear(&mut qkv, &xb, &merged_w, None)?;
        let u = unit_roundoff(dt);
        if !with_rope {
            let mut dense = [device.allocate_buffer(batch * nq * 4)?, device.allocate_buffer(batch * nk * 4)?, device.allocate_buffer(batch * nv * 4)?];
            let [d0, d1, d2] = &mut dense;
            if !stream.split_columns([d0, d1, d2], &qkv, batch, [nq, nk, nv])? {
                return Err(BackendError::UnsupportedOperation("split_columns".into()));
            }
            stream.synchronize()?;
            let ms = start.elapsed().as_secs_f64() * 1e3;
            let mut err = 0f64;
            for (a, b, n) in [(&dense[0], &sep[0], nq), (&dense[1], &sep[1], nk), (&dense[2], &sep[2], nv)] {
                let want = get_act(device, stream, b, batch * n)?;
                let got = get_act(device, stream, a, batch * n)?;
                err = err.max(worst(&got, &as_f64(&want), &magnitude_scale(dt, &want)) / u);
            }
            return Ok(outcome(err, 3.0, ms));
        }
        // rotation + cache write: the merged kernel against the kernel on the separate results
        let positions: Vec<usize> = (0..batch).map(|r| r * 3 + 1).collect();
        let (slots, blocks) = Self::kv_layout(batch);
        let params = Self::rope_params(DIM, true);
        let cache_a = device.allocate_kv_cache(1, blocks, BLOCK, KV_HEADS, DIM, DIM, 1)?;
        let cache_b = device.allocate_kv_cache(1, blocks, BLOCK, KV_HEADS, DIM, DIM, 1)?;
        let [mut sq, sk, sv] = sep;
        if !stream.rope_kv_write(&mut sq, &sk, &sv, &cache_a, 0, &positions, &slots, &params)? {
            return Err(BackendError::UnsupportedOperation("rope_kv_write".into()));
        }
        let mut q_out = device.allocate_buffer(batch * nq * 4)?;
        if !stream.rope_kv_write_qkv(&mut q_out, &qkv, [nq, nk, nv], None, None, None, 1e-6, &cache_b, 0, &positions, &slots, &params)? {
            return Err(BackendError::UnsupportedOperation("rope_kv_write_qkv".into()));
        }
        stream.synchronize()?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        let want_q = get_act(device, stream, &sq, batch * nq)?;
        let got_q = get_act(device, stream, &q_out, batch * nq)?;
        let mut err = worst(&got_q, &as_f64(&want_q), &magnitude_scale(dt, &want_q)) / u;
        let (a, b) = (Self::pool_bytes(stream, &cache_a)?, Self::pool_bytes(stream, &cache_b)?);
        let half = a.len() / 2;
        let used = batch * KV_HEADS * DIM;
        let words = |bytes: &[u8]| -> Vec<f32> { bytes.as_chunks::<2>().0.iter().take(used).map(|c| f16::from_le_bytes(*c).to_f32()).collect() };
        // the cache words are f16 values of the same numbers; their difference is bounded by the difference of the narrow results, in units of u of the activation type
        for offset in [0usize, half] {
            let (want, got) = (words(&a[offset..]), words(&b[offset..]));
            err = err.max(worst(&got, &as_f64(&want), &magnitude_scale(ActDtype::F16, &want)) / u);
        }
        Ok(outcome(err, 3.0, ms))
    }

    /// gate and up from one merged GEMM and the merged gated activation against two GEMMs and the plain one.
    fn act_merged_gate_up(&mut self, dt: ActDtype, batch: usize) -> Result<Outcome, BackendError> {
        let kind = 0x8000_0012u32;
        self.claim = Some(kind);
        let (ff, cols) = (1024usize, 512usize);
        let gate = random_kind(&mut self.rng, kind, ff, cols);
        let up = random_kind(&mut self.rng, kind, ff, cols);
        let merged: Vec<u8> = gate.iter().chain(&up).copied().collect();
        let x = round_vec(dt, &self.rng.vector(batch * cols, 1.0));
        let qt = QuantType::from_gguf_type_id(kind);
        self.device.set_activation_dtype(dt);
        let uploads = (|| -> Result<[DeviceWeight; 3], BackendError> {
            Ok([
                self.device.upload_weight("selftest gate", &gate, qt, ff, cols, None)?,
                self.device.upload_weight("selftest up", &up, qt, ff, cols, None)?,
                self.device.upload_weight("selftest gate+up", &merged, qt, 2 * ff, cols, None)?,
            ])
        })();
        let stream = self.device.create_stream();
        self.device.set_activation_dtype(ActDtype::F32);
        let ([wg, wu, wm], mut stream) = (uploads?, stream?);
        let stream = stream.as_mut();
        let device = self.device;
        let xb = put_act(device, stream, &x)?;
        let (mut g, mut u_buf, mut want_act) = (device.allocate_buffer(batch * ff * 4)?, device.allocate_buffer(batch * ff * 4)?, device.allocate_buffer(batch * ff * 4)?);
        stream.quantized_linear(&mut g, &xb, &wg, None)?;
        stream.quantized_linear(&mut u_buf, &xb, &wu, None)?;
        stream.swiglu(&mut want_act, &g, &u_buf)?;
        let mut gu = device.allocate_buffer(batch * 2 * ff * 4)?;
        let mut got_act = device.allocate_buffer(batch * ff * 4)?;
        let start = Instant::now();
        stream.quantized_linear(&mut gu, &xb, &wm, None)?;
        if !stream.glu_merged(&mut got_act, &gu, batch, ff, false)? {
            return Err(BackendError::UnsupportedOperation("glu_merged".into()));
        }
        stream.synchronize()?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        let want = get_act(device, stream, &want_act, batch * ff)?;
        let got = get_act(device, stream, &got_act, batch * ff)?;
        // gate and up each carry the rounding of their GEMM (u) and the product one more: 4 u
        Ok(outcome(worst(&got, &as_f64(&want), &magnitude_scale(dt, &want)) / unit_roundoff(dt), 4.0, ms))
    }

    /// `FLLM_PDL` on against off: the same chain of kernels must give the same bits (on GPUs older than sm_90 both runs are plain launches).
    fn act_pdl(&mut self, dt: ActDtype, rows: usize) -> Result<Outcome, BackendError> {
        extern "C" {
            fn fllm_pdl_force(v: i32);
        }
        let cols = 1024usize;
        let x = round_vec(dt, &self.rng.vector(rows * cols, 2.0));
        let b = round_vec(dt, &self.rng.vector(rows * cols, 2.0));
        let w: Vec<f32> = self.rng.vector(cols, 0.4).iter().map(|v| v + 1.0).collect();
        let (_, mut narrow) = self.act_streams(dt)?;
        let device = self.device;
        let mut run = |on: i32| -> Result<Vec<f32>, BackendError> {
            // SAFETY: plain setter of a process-wide switch.
            unsafe { fllm_pdl_force(on) };
            let stream = narrow.as_mut();
            let mut xb = put_act(device, stream, &x)?;
            let bb = put_act(device, stream, &b)?;
            let nw = put(device, stream, &w)?;
            let (mut n1, mut act) = (device.allocate_buffer(rows * cols * 4)?, device.allocate_buffer(rows * cols * 4)?);
            let r = (|| -> Result<(), BackendError> {
                for _ in 0..3 {
                    stream.rms_norm(&mut n1, &xb, &nw, 1e-6)?;
                    stream.swiglu(&mut act, &n1, &bb)?;
                    stream.residual_add(&mut xb, &act, 0.5)?;
                }
                Ok(())
            })();
            // SAFETY: back to the environment choice.
            unsafe { fllm_pdl_force(-1) };
            r?;
            get_act(device, stream, &xb, rows * cols)
        };
        let off = run(0)?;
        let start = Instant::now();
        let on = run(1)?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        let differing = off.iter().zip(&on).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        Ok(outcome(differing as f64, 0.0, ms))
    }
}
