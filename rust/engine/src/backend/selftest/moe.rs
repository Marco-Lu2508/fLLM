//! Self-test cases of the device mixture-of-experts operations (`Stream::moe_route_ex`, `moe_align`, `moe_grouped_linear`, `moe_combine_ex`,
//! `moe_gate_scale`): routing variants against an f64 reference (softmax and sigmoid scores, selection bias, groups, renormalisation,
//! scale), the expert-major sort, the grouped GEMM per weight format (plain, fused gate/up with the activation, scaled input, bias, both pair
//! tiles, realistic expert counts and widths) and a whole layer against an f64 implementation. The group `moebench` (`--only moebench`)
//! times the operations per call at the shapes of real models.
//!
//! A backend without the operations answers `UnsupportedOperation` and every case is reported as skipped.
use super::*;
use crate::backend::moe::{align_reference, moe_block_for, moe_sort_bytes, route_scores, MoeGlu, MoeGrouped, MoeRoute, MoeSort};

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn get_i32(stream: &mut dyn Stream, buf: &DeviceBuffer, count: usize) -> Result<Vec<i32>, BackendError> {
    let mut bytes = vec![0u8; count * 4];
    stream.download(&mut bytes, buf)?;
    Ok(bytes.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c)).collect())
}

/// The gated activation in f64 (the formulas of the kernels).
fn glu_reference(g: f64, u: f64, glu: MoeGlu) -> f64 {
    match glu.mode {
        0 => g / (1.0 + (-g).exp()) * u,
        1 => 0.5 * g * (1.0 + (0.7978846 * (g + 0.044715 * g * g * g)).tanh()) * u,
        _ => {
            let (alpha, limit) = (f64::from(glu.alpha), f64::from(glu.limit));
            let gc = g.min(limit);
            let uc = u.clamp(-limit, limit);
            (uc + 1.0) * gc / (1.0 + (-alpha * gc).exp())
        }
    }
}

/// Lipschitz constants of the activation in `g` and `u` around a point (central differences, with a margin), to turn the error scales of the
/// two products into the error scale of the activation.
fn glu_lipschitz(g: f64, u: f64, glu: MoeGlu) -> (f64, f64) {
    let h = 1e-3;
    let dg = (glu_reference(g + h, u, glu) - glu_reference(g - h, u, glu)).abs() / (2.0 * h);
    let du = (glu_reference(g, u + h, glu) - glu_reference(g, u - h, glu)).abs() / (2.0 * h);
    (dg * 1.2 + 1e-3, du * 1.2 + 1e-3)
}

/// Seconds-per-call helper of the timing group: two warm-up calls, `iters` back-to-back calls, one synchronisation.
fn time_us(
    stream: &mut dyn Stream,
    iters: usize,
    mut f: impl FnMut(&mut dyn Stream) -> Result<(), BackendError>,
) -> Result<f64, BackendError> {
    f(stream)?;
    f(stream)?;
    stream.synchronize()?;
    let start = Instant::now();
    for _ in 0..iters {
        f(stream)?;
    }
    stream.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1e6 / iters as f64)
}

/// One whole-layer case.
struct LayerCase {
    name: &'static str,
    experts: usize,
    k: usize,
    hidden: usize,
    ff: usize,
    tokens: usize,
    route: MoeRoute,
    bias: bool,
    shared: bool,
    shared_gate: bool,
    scaled: bool,
}

impl Tester<'_> {
    pub(super) fn moe_device_cases(&mut self) {
        // routing: the gating variants of the model families (DeepSeek-V3 / Kimi K2: sigmoid + correction bias + groups, V2: softmax with
        // groups, Qwen3: softmax + renormalisation, Llama 4: sigmoid, k = 1, LFM2-MoE: epsilon in the denominator)
        let softmax = |normalize: bool, scale: f32| MoeRoute::softmax(normalize, scale);
        let sigmoid = |groups: usize, used: usize, top2: bool, scale: f32, eps: f32| MoeRoute {
            sigmoid: true,
            normalize: true,
            scale,
            eps,
            groups,
            groups_used: used,
            group_top2: top2,
        };
        let routes: Vec<(&str, usize, usize, MoeRoute, bool)> = vec![
            ("softmax", 8, 2, softmax(true, 1.0), false),
            ("softmax", 64, 8, softmax(false, 1.0), false),
            ("softmax norm", 128, 8, softmax(true, 0.5), false),
            ("softmax norm", 256, 8, softmax(true, 1.0), false),
            ("softmax groups", 160, 6, MoeRoute { groups: 8, groups_used: 3, ..softmax(false, 16.0) }, false),
            ("sigmoid", 8, 1, MoeRoute { normalize: false, ..sigmoid(0, 0, false, 1.0, 0.0) }, false),
            ("sigmoid bias", 64, 8, sigmoid(0, 0, false, 2.5, 0.0), true),
            ("sigmoid bias groups", 256, 8, sigmoid(8, 4, true, 2.5, 0.0), true),
            ("sigmoid bias 384", 384, 8, sigmoid(1, 1, true, 2.827, 0.0), true),
            ("sigmoid eps", 32, 4, sigmoid(0, 0, false, 1.0, 1e-6), true),
            ("sigmoid groups", 64, 6, sigmoid(4, 2, false, 1.0, 0.0), false),
        ];
        for (name, experts, k, route, bias) in routes {
            for rows in [1usize, 5, 37] {
                let r = self.moe_route_check(rows, experts, k, route, bias);
                self.record("moe route", &format!("{name} {rows}x{experts} k{k}{}", if bias { " +bias" } else { "" }), r);
            }
        }
        // the older whole-call operations of the trait, which CUDA now serves on the device
        for (rows, experts, k) in [(1usize, 8usize, 2usize), (37, 64, 8)] {
            let r = self.moe_route_legacy_check(rows, experts, k);
            self.record("moe route (trait)", &format!("{rows}x{experts} k{k}"), r);
        }
        for kind in self.kinds(&[8u32, 12, 1]) {
            let r = self.moe_expert_gemm_check(kind, 6, 20, 2, 130, 512);
            self.record(&format!("moe expert_gemm {}", format_name(kind)), "6 experts 20 tokens k2 130x512", r);
        }
        // the sort: small, realistic and expert-sliced (expert parallelism) cases, both pair tiles
        for (pairs, experts, offset, local, block) in [
            (13usize, 8usize, 0usize, 8usize, 16usize),
            (100, 8, 0, 8, 64),
            (2048 * 8, 128, 0, 128, 64),
            (64 * 8, 128, 0, 128, 16),
            (4096 * 8, 256, 0, 256, 64),
            (1000, 64, 16, 32, 16),
        ] {
            let r = self.moe_align_check(pairs, experts, offset, local, block);
            self.record("moe align", &format!("{pairs} pairs {local}/{experts} experts @{offset} tile {block}"), r);
        }
        // the grouped GEMM: every format of the kernel, both tiles, with bias; K-quants need widths of 256
        let mode_swiglu = MoeGlu { mode: 0, alpha: 0.0, limit: 0.0 };
        for kind in self.kinds(&[1u32, 30, 2, 3, 6, 7, 8, 12, 13, 14, 20, 39]) {
            for block in [16usize, 64] {
                let tokens = if block == 16 { 6 } else { 70 };
                let r = self.moe_grouped_check(kind, 8, tokens, 2, 130, 512, block, 2, None, true, false, usize::MAX);
                self.record(&format!("moe grouped {}", format_name(kind)), &format!("8 experts {tokens} tokens k2 130x512 tile {block} +bias"), r);
            }
            let r = self.moe_grouped_check(kind, 8, 70, 2, 256, 512, 64, 1, None, false, false, usize::MAX);
            self.record(&format!("moe grouped {}", format_name(kind)), "down shape 8 experts 140 pairs 256x512", r);
            let r = self.moe_grouped_check(kind, 6, 40, 3, 130, 512, 16, 3, Some(mode_swiglu), true, false, usize::MAX);
            self.record(&format!("moe gate/up {}", format_name(kind)), "6 experts 40 tokens k3 130x512 swiglu +bias", r);
        }
        let glu_modes = [
            ("geglu", MoeGlu { mode: 1, alpha: 0.0, limit: 0.0 }),
            ("swiglu_oai", MoeGlu { mode: 2, alpha: 1.702, limit: 7.0 }),
        ];
        for kind in self.kinds(&[12u32, 8, 1]) {
            for (name, glu) in glu_modes {
                let r = self.moe_grouped_check(kind, 6, 40, 3, 130, 512, 64, 3, Some(glu), true, false, usize::MAX);
                self.record(&format!("moe gate/up {}", format_name(kind)), &format!("{name} 6 experts 40 tokens k3 130x512 +bias"), r);
            }
            let r = self.moe_grouped_check(kind, 8, 33, 1, 256, 512, 16, 1, Some(mode_swiglu), false, true, usize::MAX);
            self.record(&format!("moe gate/up {}", format_name(kind)), "scaled input (Llama 4) 8 experts 33 tokens k1 256x512", r);
        }
        // realistic shapes (hidden 2048, intermediate 768 / 1408 / 2048, 8..256 experts, top-k 2..8): a sample of the pairs is checked against f64
        let shapes: [(usize, usize, usize, usize, usize, usize); 7] = [
            (8, 2, 2048, 2048, 96, 64),
            (64, 8, 2048, 1408, 96, 64),
            (64, 6, 2048, 1408, 12, 16),
            (128, 8, 2048, 768, 128, 64),
            (128, 8, 2048, 768, 8, 16),
            (256, 4, 2048, 768, 64, 64),
            (256, 8, 2048, 512, 6, 16),
        ];
        for kind in self.kinds(&[12u32, 8, 1]) {
            for (experts, k, hidden, ff, tokens, block) in shapes {
                let distinct = experts.min(12);
                let r = self.moe_grouped_check(kind, distinct, tokens, k.min(distinct), ff, hidden, block, k.min(distinct), Some(mode_swiglu), false, false, 40);
                self.record(
                    &format!("moe gate/up {}", format_name(kind)),
                    &format!("{distinct}/{experts} experts {tokens} tokens k{k} {ff}x{hidden} tile {block}"),
                    r,
                );
            }
        }
        // whole layers: route -> sort -> fused gate/up -> down -> combine (+ shared expert), against an f64 layer built from the device's own routing
        let layers = [
            LayerCase { name: "mixtral-like", experts: 8, k: 2, hidden: 512, ff: 256, tokens: 70, route: softmax(true, 1.0), bias: false, shared: false, shared_gate: false, scaled: false },
            LayerCase { name: "mixtral-like decode", experts: 8, k: 2, hidden: 512, ff: 256, tokens: 3, route: softmax(true, 1.0), bias: false, shared: false, shared_gate: false, scaled: false },
            LayerCase {
                name: "deepseek-like",
                experts: 16,
                k: 4,
                hidden: 512,
                ff: 256,
                tokens: 45,
                route: sigmoid(4, 2, true, 2.5, 0.0),
                bias: true,
                shared: true,
                shared_gate: false,
                scaled: false,
            },
            LayerCase {
                name: "qwen2-moe-like",
                experts: 16,
                k: 4,
                hidden: 512,
                ff: 256,
                tokens: 45,
                route: softmax(false, 1.0),
                bias: false,
                shared: true,
                shared_gate: true,
                scaled: false,
            },
            LayerCase {
                name: "llama4-like",
                experts: 8,
                k: 1,
                hidden: 512,
                ff: 256,
                tokens: 40,
                route: MoeRoute { normalize: false, ..sigmoid(0, 0, false, 1.0, 0.0) },
                bias: false,
                shared: true,
                shared_gate: false,
                scaled: true,
            },
        ];
        for kind in self.kinds(&[12u32, 8, 1]) {
            for case in &layers {
                let r = self.moe_layer_check(kind, case);
                self.record(
                    &format!("moe layer {}", format_name(kind)),
                    &format!("{} {}e k{} {} tokens", case.name, case.experts, case.k, case.tokens),
                    r,
                );
            }
        }
    }

    fn alloc(&self, bytes: usize) -> Result<DeviceBuffer, BackendError> {
        self.device.allocate_buffer(bytes.max(4))
    }

    /// Token-expert assignment: `k` distinct experts per token, skewed to low expert numbers like the hot experts of a real router.
    fn route_ids(&mut self, tokens: usize, k: usize, experts: usize) -> Vec<i32> {
        let mut ids = Vec::with_capacity(tokens * k);
        for _ in 0..tokens {
            let mut chosen: Vec<usize> = Vec::with_capacity(k);
            while chosen.len() < k {
                let a = (self.rng.next() % experts as u64) as usize;
                let b = (self.rng.next() % experts as u64) as usize;
                let e = if k * 2 > experts { a } else { a.min(b) };
                if !chosen.contains(&e) {
                    chosen.push(e);
                }
            }
            ids.extend(chosen.iter().map(|&e| e as i32));
        }
        ids
    }

    fn moe_route_check(&mut self, rows: usize, experts: usize, k: usize, route: MoeRoute, with_bias: bool) -> Result<Outcome, BackendError> {
        let logits = self.rng.vector(rows * experts, 3.0);
        let bias = if with_bias { self.rng.vector(experts, 0.5) } else { Vec::new() };
        let mut stream = self.device.create_stream()?;
        let lb = put(self.device, stream.as_mut(), &logits)?;
        let bb = if with_bias { Some(put(self.device, stream.as_mut(), &bias)?) } else { None };
        let mut ids = self.alloc(rows * k * 4)?;
        let mut weights = self.alloc(rows * k * 4)?;
        let mut ms = 0.0;
        timed(&mut ms, stream.as_mut(), |s| s.moe_route_ex(&lb, bb.as_ref(), &mut ids, &mut weights, rows, experts, k, &route))?;
        let have_ids = get_i32(stream.as_mut(), &ids, rows * k)?;
        let have_w = get(stream.as_mut(), &weights, rows * k)?;
        let mut worst_err = 0.0f64;
        for r in 0..rows {
            let row = &logits[r * experts..(r + 1) * experts];
            let (scores, select) = route_scores(row, with_bias.then_some(bias.as_slice()), &route);
            let mut sorted_select: Vec<f64> = select.clone();
            sorted_select.sort_by(|a, b| b.total_cmp(a));
            let kth = sorted_select[k - 1];
            let slack = 1e-5 * (1.0 + kth.abs());
            let chosen = &have_ids[r * k..(r + 1) * k];
            // the device's picks must be distinct experts that the reference ranks among the best k (ties may break differently)
            for (j, &e) in chosen.iter().enumerate() {
                if e < 0 || e as usize >= experts || chosen[..j].contains(&e) || select[e as usize] < kth - slack {
                    return Err(BackendError::DriverError(format!(
                        "row {r}: the device chose experts {chosen:?}; selection value of expert {e} is {:?} but the {k}-th best is {kth}",
                        select.get(e as usize)
                    )));
                }
                if j > 0 && select[e as usize] > select[chosen[j - 1] as usize] + slack {
                    return Err(BackendError::DriverError(format!("row {r}: experts {chosen:?} are not in selection order")));
                }
            }
            let total: f64 = chosen.iter().map(|&e| scores[e as usize]).sum::<f64>() + f64::from(route.eps);
            for (j, &e) in chosen.iter().enumerate() {
                let sc = scores[e as usize];
                let want = if route.normalize { sc / total } else { sc } * f64::from(route.scale);
                let err = (f64::from(have_w[r * k + j]) - want).abs() / want.abs().max(1e-9);
                worst_err = worst_err.max(if err.is_nan() { f64::INFINITY } else { err });
            }
        }
        Ok(Outcome { device: worst_err, cpu: 0.0, tol: 2e-5, cpu_tol: 1.0, ms })
    }

    /// `Stream::moe_route`: softmax, best `k`, renormalised; indices as u32.
    fn moe_route_legacy_check(&mut self, rows: usize, experts: usize, k: usize) -> Result<Outcome, BackendError> {
        let logits = self.rng.vector(rows * experts, 3.0);
        let mut stream = self.device.create_stream()?;
        let lb = put(self.device, stream.as_mut(), &logits)?;
        let mut ids = self.alloc(rows * k * 4)?;
        let mut weights = self.alloc(rows * k * 4)?;
        let mut ms = 0.0;
        timed(&mut ms, stream.as_mut(), |s| s.moe_route(&mut weights, &mut ids, &lb, experts, k))?;
        let have_ids = get_i32(stream.as_mut(), &ids, rows * k)?;
        let have_w = get(stream.as_mut(), &weights, rows * k)?;
        let route = MoeRoute::softmax(true, 1.0);
        let mut worst_err = 0.0f64;
        for r in 0..rows {
            let row = &logits[r * experts..(r + 1) * experts];
            let (scores, _) = route_scores(row, None, &route);
            let chosen = &have_ids[r * k..(r + 1) * k];
            let mut best = scores.clone();
            best.sort_by(|a, b| b.total_cmp(a));
            let total: f64 = chosen.iter().map(|&e| scores.get(e as usize).copied().unwrap_or(f64::NAN)).sum();
            for (j, &e) in chosen.iter().enumerate() {
                let sc = scores.get(e as usize).copied().unwrap_or(f64::NAN);
                // the chosen experts must be among the k best scores (ties may break either way)
                if sc.is_nan() || sc < best[k - 1] * (1.0 - 1e-5) {
                    return Err(BackendError::DriverError(format!("row {r}: experts {chosen:?} are not the {k} best")));
                }
                let err = (f64::from(have_w[r * k + j]) - sc / total).abs() / (sc / total).max(1e-9);
                worst_err = worst_err.max(if err.is_nan() { f64::INFINITY } else { err });
            }
        }
        Ok(Outcome { device: worst_err, cpu: 0.0, tol: 2e-5, cpu_tol: 1.0, ms })
    }

    /// `Stream::moe_expert_gemm`: `out[t] = sum_j w[t][j] * (W[e[t][j]] x[t])` over a list of experts.
    fn moe_expert_gemm_check(&mut self, kind: u32, experts: usize, tokens: usize, k: usize, rows: usize, cols: usize) -> Result<Outcome, BackendError> {
        let quant = QuantType::from_gguf_type_id(kind);
        let mut dense = Vec::new();
        let mut weights_dev = Vec::new();
        for _ in 0..experts {
            let raw = random_kind(&mut self.rng, kind, rows, cols);
            dense.push(crate::compute::dequantize(&raw, kind, rows * cols).map_err(BackendError::DriverError)?);
            weights_dev.push(self.device.upload_weight("selftest", &raw, quant, rows, cols, None)?);
        }
        let ids = self.route_ids(tokens, k, experts);
        let w = self.rng.vector(tokens * k, 1.0);
        let x = self.rng.vector(tokens * cols, 1.0);
        let mut stream = self.device.create_stream()?;
        let xb = put(self.device, stream.as_mut(), &x)?;
        let wb = put(self.device, stream.as_mut(), &w)?;
        let mut ib = self.alloc(tokens * k * 4)?;
        stream.upload(&mut ib, &i32_bytes(&ids))?;
        let mut out = self.alloc(tokens * rows * 4)?;
        let mut ms = 0.0;
        timed(&mut ms, stream.as_mut(), |s| s.moe_expert_gemm(&mut out, &xb, &weights_dev, &ib, &wb, k))?;
        let have = get(stream.as_mut(), &out, tokens * rows)?;
        let mut want = vec![0.0f64; tokens * rows];
        let mut scale = vec![0.0f64; tokens * rows];
        for t in 0..tokens {
            for j in 0..k {
                let e = ids[t * k + j] as usize;
                let wt = f64::from(w[t * k + j]);
                for r in 0..rows {
                    let mut sum = 0.0f64;
                    let mut mag = 0.0f64;
                    for c in 0..cols {
                        let term = f64::from(dense[e][r * cols + c]) * f64::from(x[t * cols + c]);
                        sum += term;
                        mag += term.abs();
                    }
                    want[t * rows + r] += wt * sum;
                    scale[t * rows + r] += wt.abs() * mag;
                }
            }
        }
        Ok(Outcome { device: worst(&have, &want, &scale), cpu: 0.0, tol: Self::FAST_TOL, cpu_tol: 1.0, ms })
    }

    fn moe_align_check(&mut self, pairs: usize, experts: usize, offset: usize, local: usize, block: usize) -> Result<Outcome, BackendError> {
        let ids: Vec<i32> = (0..pairs)
            .map(|_| {
                let a = (self.rng.next() % experts as u64) as i32;
                let b = (self.rng.next() % experts as u64) as i32;
                a.min(b)
            })
            .collect();
        let (sb, bb, pb) = moe_sort_bytes(pairs, local, block);
        let mut stream = self.device.create_stream()?;
        let mut idb = self.alloc(pairs * 4)?;
        stream.upload(&mut idb, &i32_bytes(&ids))?;
        let mut sort = MoeSort { sorted: self.alloc(sb)?, block_expert: self.alloc(bb)?, post: self.alloc(pb)?, pairs, block, max_blocks: bb / 4 };
        let mut ms = 0.0;
        timed(&mut ms, stream.as_mut(), |s| s.moe_align(&idb, &mut sort, offset, local))?;
        let sorted = get_i32(stream.as_mut(), &sort.sorted, sb / 4)?;
        let block_expert = get_i32(stream.as_mut(), &sort.block_expert, bb / 4)?;
        let post = get_i32(stream.as_mut(), &sort.post, 1)?[0] as usize;
        let reference = align_reference(&ids, offset, local);
        let want_post: usize = reference.iter().map(|v| v.len().div_ceil(block) * block).sum();
        let fail = |msg: String| Err(BackendError::DriverError(format!("moe_align: {msg}")));
        if post != want_post {
            return fail(format!("padded length {post}, expected {want_post}"));
        }
        let mut found: Vec<Vec<usize>> = vec![Vec::new(); local];
        for b in 0..bb / 4 {
            let e = block_expert[b];
            if b * block >= post {
                if e != -1 {
                    return fail(format!("unused block {b} names expert {e}"));
                }
                continue;
            }
            if e < 0 || e as usize >= local {
                return fail(format!("block {b} has expert {e}"));
            }
            for &p in &sorted[b * block..(b + 1) * block] {
                if p as usize == pairs {
                    continue;
                }
                if p < 0 || p as usize > pairs || ids[p as usize] as i64 - offset as i64 != i64::from(e) {
                    return fail(format!("block {b} (expert {e}) holds pair {p}, which is routed to expert {:?}", ids.get(p as usize)));
                }
                found[e as usize].push(p as usize);
            }
        }
        if sorted[post.min(sorted.len())..].iter().any(|&p| p as usize != pairs) {
            return fail("entries past the padded length are not the padding value".into());
        }
        for (e, (have, want)) in found.iter_mut().zip(&reference).enumerate() {
            have.sort_unstable();
            if have != want {
                return fail(format!("expert {e} got {} pairs, expected {}", have.len(), want.len()));
            }
            // every block of an expert is full except the last one
            let blocks: Vec<usize> = (0..post / block).filter(|&b| block_expert[b] == e as i32).collect();
            if blocks.len() != want.len().div_ceil(block) {
                return fail(format!("expert {e} has {} blocks, expected {}", blocks.len(), want.len().div_ceil(block)));
            }
        }
        Ok(Outcome { device: 0.0, cpu: 0.0, tol: 0.0, cpu_tol: 1.0, ms })
    }

    /// One grouped product (optionally the fused gate/up with the activation) against f64 for `sample` pairs (all of them with
    /// `usize::MAX`). `rows` x `cols` is the shape of one expert matrix.
    #[allow(clippy::too_many_arguments)]
    fn moe_grouped_check(
        &mut self,
        kind: u32,
        experts: usize,
        tokens: usize,
        k: usize,
        rows: usize,
        cols: usize,
        block: usize,
        x_div: usize,
        glu: Option<MoeGlu>,
        with_bias: bool,
        scaled: bool,
        sample: usize,
    ) -> Result<Outcome, BackendError> {
        let quant = QuantType::from_gguf_type_id(kind);
        let pairs = tokens * k;
        let projections = if glu.is_some() { 2 } else { 1 };
        let mut raws: Vec<Vec<Vec<u8>>> = Vec::new();
        let mut dense: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut biases: Vec<Vec<Vec<f32>>> = Vec::new();
        for _ in 0..projections {
            let (mut r, mut d, mut b) = (Vec::new(), Vec::new(), Vec::new());
            for _ in 0..experts {
                let raw = random_kind(&mut self.rng, kind, rows, cols);
                d.push(crate::compute::dequantize(&raw, kind, rows * cols).map_err(BackendError::DriverError)?);
                r.push(raw);
                b.push(if with_bias { self.rng.vector(rows, 1.0) } else { Vec::new() });
            }
            raws.push(r);
            dense.push(d);
            biases.push(b);
        }
        let mut weights: Vec<Vec<crate::backend::DeviceWeight>> = Vec::new();
        for p in 0..projections {
            let mut list = Vec::new();
            for e in 0..experts {
                list.push(self.device.upload_weight("selftest", &raws[p][e], quant, rows, cols, with_bias.then_some(biases[p][e].as_slice()))?);
            }
            weights.push(list);
        }
        let device_quant = weights[0][0].quant_type;
        let refs: Vec<Vec<&crate::backend::DeviceWeight>> = weights.iter().map(|l| l.iter().collect()).collect();
        let tables: Vec<DeviceBuffer> = refs.iter().map(|r| self.device.upload_pointer_table(r)).collect::<Result<_, _>>()?;
        let bias_tables: Vec<Option<DeviceBuffer>> = refs
            .iter()
            .map(|r| if with_bias { self.device.upload_bias_table(r).map(Some) } else { Ok(None) })
            .collect::<Result<_, _>>()?;

        let ids = self.route_ids(tokens, k, experts);
        let pw: Vec<f32> = (0..pairs).map(|_| 0.2 + (self.rng.next() % 1000) as f32 / 1000.0).collect();
        let inputs = pairs.div_ceil(x_div);
        let x = self.rng.vector(inputs * cols, 1.0);

        let mut stream = self.device.create_stream()?;
        let xb = put(self.device, stream.as_mut(), &x)?;
        let pwb = if scaled { Some(put(self.device, stream.as_mut(), &pw)?) } else { None };
        let mut idb = self.alloc(pairs * 4)?;
        stream.upload(&mut idb, &i32_bytes(&ids))?;
        let (sb, bb, pb) = moe_sort_bytes(pairs, experts, block);
        let mut sort = MoeSort { sorted: self.alloc(sb)?, block_expert: self.alloc(bb)?, post: self.alloc(pb)?, pairs, block, max_blocks: bb / 4 };
        stream.moe_align(&idb, &mut sort, 0, experts)?;
        let mut out = self.alloc(pairs * rows * 4)?;
        let mut ms = 0.0;
        timed(&mut ms, stream.as_mut(), |s| {
            s.moe_grouped_linear(MoeGrouped {
                out: &mut out,
                x: &xb,
                table: &tables[0],
                table_up: glu.map(|_| &tables[1]),
                bias: bias_tables[0].as_ref(),
                bias_up: if glu.is_some() { bias_tables[1].as_ref() } else { None },
                input_weights: pwb.as_ref(),
                sort: &sort,
                quant_type: device_quant,
                rows,
                cols,
                x_div,
                glu,
            })
        })?;
        let have = get(stream.as_mut(), &out, pairs * rows)?;

        let step = if sample >= pairs { 1 } else { (pairs / sample).max(1) };
        let (mut got, mut want, mut scale) = (Vec::new(), Vec::new(), Vec::new());
        for p in (0..pairs).step_by(step) {
            let e = ids[p] as usize;
            let xv = &x[(p / x_div) * cols..(p / x_div + 1) * cols];
            let factor = if scaled { f64::from(pw[p]) } else { 1.0 };
            for r in 0..rows {
                let mut products = [(0.0f64, 0.0f64); 2];
                for (q, slot) in products.iter_mut().enumerate().take(projections) {
                    let w = &dense[q][e][r * cols..(r + 1) * cols];
                    let (mut sum, mut mag) = (0.0f64, 0.0f64);
                    for (&a, &b) in xv.iter().zip(w) {
                        let term = f64::from(a) * factor * f64::from(b);
                        sum += term;
                        mag += term.abs();
                    }
                    if with_bias {
                        let bv = f64::from(biases[q][e][r]);
                        sum += bv;
                        mag += bv.abs();
                    }
                    *slot = (sum, mag);
                }
                got.push(have[p * rows + r]);
                match glu {
                    Some(g) => {
                        let (gv, uv) = (products[0].0, products[1].0);
                        let (lg, lu) = glu_lipschitz(gv, uv, g);
                        want.push(glu_reference(gv, uv, g));
                        scale.push(lg * products[0].1 + lu * products[1].1);
                    }
                    None => {
                        want.push(products[0].0);
                        scale.push(products[0].1);
                    }
                }
            }
        }
        Ok(Outcome { device: worst(&got, &want, &scale), cpu: 0.0, tol: Self::FAST_TOL, cpu_tol: 1.0, ms })
    }

    /// Route -> sort -> fused gate/up -> down -> combine (+ shared expert) on the device against an f64 layer that uses the device's routing.
    fn moe_layer_check(&mut self, kind: u32, c: &LayerCase) -> Result<Outcome, BackendError> {
        let quant = QuantType::from_gguf_type_id(kind);
        let (experts, k, hidden, ff, tokens) = (c.experts, c.k, c.hidden, c.ff, c.tokens);
        let pairs = tokens * k;
        let projections = [(ff, hidden), (ff, hidden), (hidden, ff)];
        let count = experts + usize::from(c.shared);
        // raw and dense weights per projection (gate, up, down) for the routed experts followed by the shared one
        let mut raw: Vec<Vec<Vec<u8>>> = vec![Vec::new(); 3];
        let mut dense: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 3];
        for (p, &(rows, cols)) in projections.iter().enumerate() {
            for _ in 0..count {
                let bytes = random_kind(&mut self.rng, kind, rows, cols);
                dense[p].push(crate::compute::dequantize(&bytes, kind, rows * cols).map_err(BackendError::DriverError)?);
                raw[p].push(bytes);
            }
        }
        let mut dev: Vec<Vec<crate::backend::DeviceWeight>> = vec![Vec::new(); 3];
        for (p, &(rows, cols)) in projections.iter().enumerate() {
            for bytes in &raw[p] {
                dev[p].push(self.device.upload_weight("selftest", bytes, quant, rows, cols, None)?);
            }
        }
        let routed: Vec<Vec<&crate::backend::DeviceWeight>> = dev.iter().map(|l| l[..experts].iter().collect()).collect();
        let tables: Vec<DeviceBuffer> = routed.iter().map(|r| self.device.upload_pointer_table(r)).collect::<Result<_, _>>()?;
        let logits = self.rng.vector(tokens * experts, 2.0);
        let sel_bias = if c.bias { self.rng.vector(experts, 0.3) } else { Vec::new() };
        // The grouped GEMMs take FP16 activations, so the SwiGLU outputs (~ gate * up) must stay below 65504. The random k-quant
        // weights are far larger than real ones (q4_k: about 3 per weight), which put single activations of this layer past the
        // FP16 range (inf * 0 = NaN). The input is uniform in +-x_scale, which gives the gate / up projections a standard deviation of about 14 (SwiGLU outputs of a few hundred).
        let weight_ms = {
            let (mut sum, mut n) = (0.0f64, 0usize);
            for w in dense.iter().take(2).flatten() {
                sum += w.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>();
                n += w.len();
            }
            sum / n.max(1) as f64
        };
        let x_scale = (24.0 / (weight_ms.sqrt() * (hidden as f64).sqrt()).max(1e-9)).min(1.0) as f32;
        let x = self.rng.vector(tokens * hidden, x_scale);
        let gate_vec = if c.shared_gate { self.rng.vector(hidden, 0.3) } else { Vec::new() };

        let mut stream = self.device.create_stream()?;
        let xb = put(self.device, stream.as_mut(), &x)?;
        let lb = put(self.device, stream.as_mut(), &logits)?;
        let bb = if c.bias { Some(put(self.device, stream.as_mut(), &sel_bias)?) } else { None };
        let gb = if c.shared_gate { Some(put(self.device, stream.as_mut(), &gate_vec)?) } else { None };
        let mut ids = self.alloc(pairs * 4)?;
        let mut weights = self.alloc(pairs * 4)?;
        let mut ms = 0.0;
        let block = moe_block_for(pairs, experts);
        let (sb, blb, pb) = moe_sort_bytes(pairs, experts, block);
        let mut sort = MoeSort { sorted: self.alloc(sb)?, block_expert: self.alloc(blb)?, post: self.alloc(pb)?, pairs, block, max_blocks: blb / 4 };
        let mut act = self.alloc(pairs * ff * 4)?;
        let mut y = self.alloc(pairs * hidden * 4)?;
        let mut out = self.alloc(tokens * hidden * 4)?;
        let mut shared_out = if c.shared { Some(self.alloc(tokens * hidden * 4)?) } else { None };
        let mut gate_scale = if c.shared_gate { Some(self.alloc(tokens * 4)?) } else { None };
        let glu = MoeGlu { mode: 0, alpha: 0.0, limit: 0.0 };
        timed(&mut ms, stream.as_mut(), |s| {
            s.moe_route_ex(&lb, bb.as_ref(), &mut ids, &mut weights, tokens, experts, k, &c.route)?;
            s.moe_align(&ids, &mut sort, 0, experts)?;
            s.moe_grouped_linear(MoeGrouped {
                out: &mut act,
                x: &xb,
                table: &tables[0],
                table_up: Some(&tables[1]),
                bias: None,
                bias_up: None,
                input_weights: c.scaled.then_some(&weights),
                sort: &sort,
                quant_type: dev[0][0].quant_type,
                rows: ff,
                cols: hidden,
                x_div: k,
                glu: Some(glu),
            })?;
            s.moe_grouped_linear(MoeGrouped {
                out: &mut y,
                x: &act,
                table: &tables[2],
                table_up: None,
                bias: None,
                bias_up: None,
                input_weights: None,
                sort: &sort,
                quant_type: dev[2][0].quant_type,
                rows: hidden,
                cols: ff,
                x_div: 1,
                glu: None,
            })?;
            if let Some(so) = shared_out.as_mut() {
                let mut g = self.device.allocate_buffer(tokens * ff * 4)?;
                let mut u = self.device.allocate_buffer(tokens * ff * 4)?;
                let mut a = self.device.allocate_buffer(tokens * ff * 4)?;
                s.quantized_linear(&mut g, &xb, &dev[0][experts], None)?;
                s.quantized_linear(&mut u, &xb, &dev[1][experts], None)?;
                s.swiglu(&mut a, &g, &u)?;
                s.quantized_linear(so, &a, &dev[2][experts], None)?;
            }
            if let (Some(gs), Some(g)) = (gate_scale.as_mut(), gb.as_ref()) {
                s.moe_gate_scale(gs, &xb, g, tokens, hidden)?;
            }
            s.moe_combine_ex(&mut out, &y, (!c.scaled).then_some(&weights), shared_out.as_ref(), gate_scale.as_ref(), tokens, k, hidden)
        })?;
        let have = get(stream.as_mut(), &out, tokens * hidden)?;
        let ids_h = get_i32(stream.as_mut(), &ids, pairs)?;
        let w_h = get(stream.as_mut(), &weights, pairs)?;

        // f64 layer from the device's routing decisions
        let mut want = vec![0.0f64; tokens * hidden];
        let mut scale = vec![0.0f64; tokens * hidden];
        // one expert FFN for one input vector: (output, error scale of the output)
        let ffn = |e: usize, xv: &[f64], dense: &[Vec<Vec<f32>>]| -> (Vec<f64>, Vec<f64>) {
            let mut a = vec![0.0f64; ff];
            let mut a_scale = vec![0.0f64; ff];
            for i in 0..ff {
                let (mut g, mut gs, mut u, mut us) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for c in 0..hidden {
                    let tg = f64::from(dense[0][e][i * hidden + c]) * xv[c];
                    let tu = f64::from(dense[1][e][i * hidden + c]) * xv[c];
                    g += tg;
                    gs += tg.abs();
                    u += tu;
                    us += tu.abs();
                }
                a[i] = glu_reference(g, u, glu);
                let (lg, lu) = glu_lipschitz(g, u, glu);
                a_scale[i] = lg * gs + lu * us;
            }
            let mut o = vec![0.0f64; hidden];
            let mut os = vec![0.0f64; hidden];
            for h in 0..hidden {
                for i in 0..ff {
                    let w = f64::from(dense[2][e][h * ff + i]);
                    o[h] += w * a[i];
                    os[h] += w.abs() * (a[i].abs() + a_scale[i]);
                }
            }
            (o, os)
        };
        for t in 0..tokens {
            let xt: Vec<f64> = x[t * hidden..(t + 1) * hidden].iter().map(|&v| f64::from(v)).collect();
            for j in 0..k {
                let e = ids_h[t * k + j] as usize;
                let wt = f64::from(w_h[t * k + j]);
                let xin: Vec<f64> = if c.scaled { xt.iter().map(|v| v * wt).collect() } else { xt.clone() };
                let (o, os) = ffn(e, &xin, &dense);
                let factor = if c.scaled { 1.0 } else { wt };
                for h in 0..hidden {
                    want[t * hidden + h] += factor * o[h];
                    scale[t * hidden + h] += factor.abs() * os[h];
                }
            }
            if c.shared {
                let g = if c.shared_gate {
                    let dot: f64 = xt.iter().zip(&gate_vec).map(|(a, &b)| a * f64::from(b)).sum();
                    1.0 / (1.0 + (-dot).exp())
                } else {
                    1.0
                };
                let (o, os) = ffn(experts, &xt, &dense);
                for h in 0..hidden {
                    want[t * hidden + h] += g * o[h];
                    scale[t * hidden + h] += g * os[h];
                }
            }
        }
        Ok(Outcome { device: worst(&have, &want, &scale), cpu: 0.0, tol: Self::FAST_TOL, cpu_tol: 1.0, ms })
    }

    /// The group `moebench`: microseconds per call of every stage at the shapes of real models, for comparison with vLLM's `fused_moe`
    /// (`benchmarks/kernels/benchmark_moe.py`) on the same GPU. `FLLM_MOE_BENCH_TOKENS=1,16,256` changes the token counts, `FLLM_MOE_BENCH_ITERS`
    /// the number of timed calls (default 20), `--quants` the weight formats (default q4_k, q8_0, f16). Experts beyond the 24th reuse the
    /// weights of the earlier ones (random weights of 3 x 512 experts would take minutes to generate), so the L2 cache sees fewer distinct bytes
    /// than a real model; the numbers are a lower bound of the time for the models with more than 24 experts.
    pub(super) fn moe_bench_cases(&mut self) {
        let env_list = |name: &str, default: &[usize]| -> Vec<usize> {
            std::env::var(name)
                .ok()
                .map(|v| v.split(',').filter_map(|t| t.trim().parse().ok()).collect::<Vec<usize>>())
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| default.to_vec())
        };
        let tokens_list = env_list("FLLM_MOE_BENCH_TOKENS", &[1, 4, 16, 64, 256, 1024, 4096]);
        let iters = env_list("FLLM_MOE_BENCH_ITERS", &[20])[0].max(1);
        let configs: [(&str, usize, usize, usize, usize); 5] = [
            ("qwen3-30b-a3b", 128, 8, 2048, 768),
            ("olmoe-1b-7b", 64, 8, 2048, 1024),
            ("deepseek-v2-lite", 64, 6, 2048, 1408),
            ("qwen3-next-80b", 512, 10, 2048, 512),
            ("gpt-oss-20b", 32, 4, 2880, 2880),
        ];
        println!("moebench: microseconds per call, {iters} timed calls after two warm-up calls; all experts have distinct weights (no aliasing)");
        for kind in self.kinds(&[12u32, 8, 1]) {
            for &(name, experts, k, hidden, ff) in &configs {
                if let Err(e) = self.moe_bench_model(kind, name, experts, k, hidden, ff, &tokens_list, iters) {
                    self.record(&format!("moebench {}", format_name(kind)), name, Err(e));
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn moe_bench_model(&mut self, kind: u32, name: &str, experts: usize, k: usize, hidden: usize, ff: usize, tokens_list: &[usize], iters: usize) -> Result<(), BackendError> {
        let quant = QuantType::from_gguf_type_id(kind);
        let distinct = std::env::var("FLLM_MOE_BENCH_MAX_EXPERTS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(experts)
            .min(experts);
        let projections = [(ff, hidden), (ff, hidden), (hidden, ff)];
        let mut dev: Vec<Vec<crate::backend::DeviceWeight>> = vec![Vec::new(); 3];
        for (p, &(rows, cols)) in projections.iter().enumerate() {
            for _ in 0..distinct {
                let bytes = random_kind(&mut self.rng, kind, rows, cols);
                dev[p].push(self.device.upload_weight("bench", &bytes, quant, rows, cols, None)?);
            }
        }
        let tables: Vec<DeviceBuffer> = dev
            .iter()
            .map(|l| {
                let refs: Vec<&crate::backend::DeviceWeight> = (0..experts).map(|e| &l[e % distinct]).collect();
                self.device.upload_pointer_table(&refs)
            })
            .collect::<Result<_, _>>()?;
        let dq = dev[0][0].quant_type;
        let route = MoeRoute::softmax(true, 1.0);
        let weight_bytes = |q: QuantType, rows: usize, cols: usize| (rows * cols / q.block_size().max(1) * q.block_bytes()) as f64;
        for &tokens in tokens_list {
            let pairs = tokens * k;
            let block = moe_block_for(pairs, experts);
            let x = self.rng.vector(tokens * hidden, 1.0);
            let logits = self.rng.vector(tokens * experts, 2.0);
            let mut stream = self.device.create_stream()?;
            let xb = put(self.device, stream.as_mut(), &x)?;
            let lb = put(self.device, stream.as_mut(), &logits)?;
            let mut ids = self.alloc(pairs * 4)?;
            let mut weights = self.alloc(pairs * 4)?;
            let (sb, bb, pb) = moe_sort_bytes(pairs, experts, block);
            let mut sort = MoeSort { sorted: self.alloc(sb)?, block_expert: self.alloc(bb)?, post: self.alloc(pb)?, pairs, block, max_blocks: bb / 4 };
            let mut act = self.alloc(pairs * ff * 4)?;
            let mut y = self.alloc(pairs * hidden * 4)?;
            let mut out = self.alloc(tokens * hidden * 4)?;
            let glu = MoeGlu { mode: 0, alpha: 0.0, limit: 0.0 };
            let t_route = time_us(stream.as_mut(), iters, |s| s.moe_route_ex(&lb, None, &mut ids, &mut weights, tokens, experts, k, &route))?;
            let t_align = time_us(stream.as_mut(), iters, |s| s.moe_align(&ids, &mut sort, 0, experts))?;
            let t_gate_up = time_us(stream.as_mut(), iters, |s| {
                s.moe_grouped_linear(MoeGrouped {
                    out: &mut act,
                    x: &xb,
                    table: &tables[0],
                    table_up: Some(&tables[1]),
                    bias: None,
                    bias_up: None,
                    input_weights: None,
                    sort: &sort,
                    quant_type: dq,
                    rows: ff,
                    cols: hidden,
                    x_div: k,
                    glu: Some(glu),
                })
            })?;
            let t_down = time_us(stream.as_mut(), iters, |s| {
                s.moe_grouped_linear(MoeGrouped {
                    out: &mut y,
                    x: &act,
                    table: &tables[2],
                    table_up: None,
                    bias: None,
                    bias_up: None,
                    input_weights: None,
                    sort: &sort,
                    quant_type: dq,
                    rows: hidden,
                    cols: ff,
                    x_div: 1,
                    glu: None,
                })
            })?;
            let t_combine = time_us(stream.as_mut(), iters, |s| s.moe_combine_ex(&mut out, &y, Some(&weights), None, None, tokens, k, hidden))?;
            let total = t_route + t_align + t_gate_up + t_down + t_combine;
            // bytes the grouped products read: the experts that get at least one token (about min(experts, pairs)) times three projections
            let touched = (experts as f64).min(pairs as f64);
            let bytes = touched * (2.0 * weight_bytes(dq, ff, hidden) + weight_bytes(dq, hidden, ff));
            // the decode kernels of the earlier path, where they exist (integer dot products, a launch per projection)
            let legacy = if tokens * k <= 256 {
                let slots = pairs;
                let mut gate = self.alloc(slots * ff * 4)?;
                let mut up = self.alloc(slots * ff * 4)?;
                let mut dn = self.alloc(slots * hidden * 4)?;
                time_us(stream.as_mut(), iters, |s| {
                    s.moe_linear(&mut gate, &xb, &tables[0], None, &ids, dq, ff, hidden, slots, k)?;
                    s.moe_linear(&mut up, &xb, &tables[1], None, &ids, dq, ff, hidden, slots, k)?;
                    s.moe_linear(&mut dn, &act, &tables[2], None, &ids, dq, hidden, ff, slots, 1)
                })
                .ok()
            } else {
                None
            };
            if self.printed_group != self.group {
                println!("== {} ==", self.group);
                self.printed_group = self.group;
            }
            println!(
                "  moebench {:<16} {:<6} {experts:>3}e k{k:<2} {hidden}x{ff} {tokens:>5} tokens tile {block:<2}: route {t_route:7.1}  align {t_align:7.1}  gate+up {t_gate_up:8.1}  down {t_down:8.1}  combine {t_combine:7.1}  total {total:9.1} us  ({:.1} GB/s weights, {:.0} tokens/s){}",
                name,
                format_name(kind),
                bytes / (t_gate_up + t_down) / 1e3,
                tokens as f64 / total * 1e6,
                legacy.map_or(String::new(), |l| format!("  | earlier moe_linear x3: {l:8.1} us"))
            );
            self.record(
                &format!("moebench {}", format_name(kind)),
                &format!("{name} {experts}e k{k} {tokens} tokens"),
                Ok(Outcome { device: 0.0, cpu: 0.0, tol: 1.0, cpu_tol: 1.0, ms: total / 1e3 }),
            );
        }
        Ok(())
    }
}
