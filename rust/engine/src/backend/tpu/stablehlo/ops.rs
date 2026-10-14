//! Composite StableHLO operations of the transformer: softmax, normalisations, rotary embedding, projections, gated activations,
//! paged attention and the cache update, written on top of [`Func`].
//!
//! Every function here is a pure program fragment: it takes values of the function under construction and returns values; it does not
//! know where the data lives. The backend (`backend/ops.rs`) wraps the fragments into programs whose parameters are device buffers,
//! and `model.rs` chains them into whole transformer blocks. Numerical contract: the formulas are those of the CPU backend
//! (`backend/cpu.rs`); only the order of summations differs.
//!
//! Level of verification: L0 (parsed and verified by an MLIR StableHLO parser; never compiled for or run on a TPU).

use super::{Cmp, DType, DotDims, Func, Precision, Reducer, Value};

/// How exact the arithmetic is.
///
/// `Exact` keeps every operand in f32 and asks the matrix unit for full precision (on a TPU a float32 product is split into several
/// bfloat16 passes), which is what `selftest-gpu` and greedy-decoding comparisons with the CPU need. `Fast` feeds the matrix unit
/// bfloat16 operands and accumulates in f32, the usual TPU inference trade (one pass).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Numerics {
    Exact,
    Fast,
}

impl Numerics {
    pub fn precision(self) -> Precision {
        match self {
            Self::Exact => Precision::Highest,
            Self::Fast => Precision::Default,
        }
    }

    /// The type matrix operands are converted to.
    pub fn operand(self) -> DType {
        match self {
            Self::Exact => DType::F32,
            Self::Fast => DType::Bf16,
        }
    }
}

/// A weight matrix `[rows, cols]` as the program receives it.
#[derive(Debug, Clone)]
pub struct Weight {
    /// `bf16`, `f32` or `i8`.
    pub w: Value,
    /// For `i8`: `[rows, groups]` f32 scales, each covering `cols / groups` columns.
    pub scales: Option<Value>,
}

/// Parameters of the paged attention fragment.
#[derive(Debug, Clone)]
pub struct AttentionSpec {
    pub rows: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub value_dim: usize,
    /// Entries per row of the block table.
    pub stride: usize,
    pub block_size: usize,
    /// Key positions materialised per row: `blocks * block_size`, at least the longest sequence of the batch.
    pub positions: usize,
    pub scale: f32,
    pub softcap: Option<f32>,
    pub window: Option<usize>,
    pub chunk: Option<usize>,
    pub numerics: Numerics,
}

/// The device-side inputs of the attention fragment.
pub struct AttentionInputs<'a> {
    /// `[rows, heads * head_dim]` f32
    pub q: &'a Value,
    /// `[layers * slots, kv_heads * head_dim]` f16
    pub key_pool: &'a Value,
    /// `[layers * slots, kv_heads * value_dim]` f16
    pub value_pool: &'a Value,
    /// Scalar i32: first row of the layer in the pools (`layer * slots`).
    pub layer_row0: &'a Value,
    /// `[rows, stride]` i32
    pub tables: &'a Value,
    /// `[rows]` i32
    pub lengths: &'a Value,
    /// `[heads]` f32
    pub sinks: Option<&'a Value>,
    /// `[heads]` f32
    pub alibi: Option<&'a Value>,
}

fn dims1(n: usize) -> [i64; 1] {
    [n as i64]
}

impl Func {
    /// Broadcast a reduction result `r` back over the dimensions `reduced` of an array of shape `full`.
    pub fn unreduce(&mut self, r: &Value, full: &[i64], reduced: &[i64]) -> Value {
        let map: Vec<i64> = (0..full.len() as i64)
            .filter(|d| !reduced.contains(d))
            .collect();
        self.broadcast_in_dim(r, full, &map)
    }

    /// Softmax over the last dimension.
    pub fn softmax_last(&mut self, x: &Value) -> Value {
        let last = x.ty.rank() as i64 - 1;
        let m = self.reduce(x, Reducer::Max, &[last]);
        let mb = self.unreduce(&m, x.dims(), &[last]);
        let shifted = self.sub(x, &mb);
        let e = self.exp(&shifted);
        let s = self.sum(&e, &[last]);
        let sb = self.unreduce(&s, x.dims(), &[last]);
        self.div(&e, &sb)
    }

    /// RMSNorm over the last dimension of `x` (`[rows, cols]`): `x * weight / sqrt(mean(x^2) + eps)`.
    pub fn rms_norm(&mut self, x: &Value, weight: &Value, eps: f32) -> Value {
        let (rows, cols) = (x.dims()[0], x.dims()[1]);
        assert_eq!(
            weight.dims(),
            [cols],
            "rms_norm: weight {} for width {cols}",
            weight.ty
        );
        let squares = self.mul(x, x);
        let total = self.sum(&squares, &[1]);
        let mean = self.mul_const(&total, 1.0 / cols as f64);
        let shifted = self.add_const(&mean, f64::from(eps));
        let root = self.sqrt(&shifted);
        let one = self.constant(1.0, &[rows], DType::F32);
        let inv = self.div(&one, &root);
        let inv_b = self.unreduce(&inv, x.dims(), &[1]);
        let scaled = self.mul(x, &inv_b);
        let w_b = self.broadcast_in_dim(weight, x.dims(), &[1]);
        self.mul(&scaled, &w_b)
    }

    /// LayerNorm over the last dimension of `x` (`[rows, cols]`).
    pub fn layer_norm(
        &mut self,
        x: &Value,
        weight: &Value,
        bias: Option<&Value>,
        eps: f32,
    ) -> Value {
        let (rows, cols) = (x.dims()[0], x.dims()[1]);
        let total = self.sum(x, &[1]);
        let mean = self.mul_const(&total, 1.0 / cols as f64);
        let mean_b = self.unreduce(&mean, x.dims(), &[1]);
        let centered = self.sub(x, &mean_b);
        let squares = self.mul(&centered, &centered);
        let var_total = self.sum(&squares, &[1]);
        let var = self.mul_const(&var_total, 1.0 / cols as f64);
        let shifted = self.add_const(&var, f64::from(eps));
        let root = self.sqrt(&shifted);
        let one = self.constant(1.0, &[rows], DType::F32);
        let inv = self.div(&one, &root);
        let inv_b = self.unreduce(&inv, x.dims(), &[1]);
        let normed = self.mul(&centered, &inv_b);
        let w_b = self.broadcast_in_dim(weight, x.dims(), &[1]);
        let scaled = self.mul(&normed, &w_b);
        match bias {
            Some(b) => {
                let b_b = self.broadcast_in_dim(b, x.dims(), &[1]);
                self.add(&scaled, &b_b)
            }
            None => scaled,
        }
    }

    /// Rotary embedding of `x` (`[rows, heads, dim]`) with `cos` / `sin` (`[rows, rope_dim / 2]`): the first `rope_dim` values of every
    /// head rotate in pairs `(d, d + half)` (`neox`) or `(2d, 2d + 1)`; the rest stays.
    pub fn rope_rotate(
        &mut self,
        x: &Value,
        cos: &Value,
        sin: &Value,
        rope_dim: usize,
        neox: bool,
    ) -> Value {
        let (rows, heads, dim) = (x.dims()[0], x.dims()[1], x.dims()[2]);
        let (rope, half) = (rope_dim as i64, (rope_dim / 2) as i64);
        assert!(
            rope <= dim && rope % 2 == 0 && half > 0,
            "rope_rotate: rope width {rope} for head width {dim}"
        );
        assert_eq!(cos.dims(), [rows, half], "rope_rotate: cos {}", cos.ty);
        let shape = [rows, heads, half];
        let cos_b = self.broadcast_in_dim(cos, &shape, &[0, 2]);
        let sin_b = self.broadcast_in_dim(sin, &shape, &[0, 2]);
        let (a, b) = if neox {
            (
                self.slice(x, &[0, 0, 0], &[rows, heads, half]),
                self.slice(x, &[0, 0, half], &[rows, heads, rope]),
            )
        } else {
            let pairs = self.slice(x, &[0, 0, 0], &[rows, heads, rope]);
            let pairs = self.reshape(&pairs, &[rows, heads, half, 2]);
            let a = self.slice(&pairs, &[0, 0, 0, 0], &[rows, heads, half, 1]);
            let b = self.slice(&pairs, &[0, 0, 0, 1], &[rows, heads, half, 2]);
            (self.reshape(&a, &shape), self.reshape(&b, &shape))
        };
        let ac = self.mul(&a, &cos_b);
        let bs = self.mul(&b, &sin_b);
        let rot_a = self.sub(&ac, &bs);
        let bc = self.mul(&b, &cos_b);
        let as_ = self.mul(&a, &sin_b);
        let rot_b = self.add(&bc, &as_);
        let rotated = if neox {
            self.concatenate(&[&rot_a, &rot_b], 2)
        } else {
            let a4 = self.reshape(&rot_a, &[rows, heads, half, 1]);
            let b4 = self.reshape(&rot_b, &[rows, heads, half, 1]);
            let joined = self.concatenate(&[&a4, &b4], 3);
            self.reshape(&joined, &[rows, heads, rope])
        };
        if rope == dim {
            rotated
        } else {
            let rest = self.slice(x, &[0, 0, rope], &[rows, heads, dim]);
            self.concatenate(&[&rotated, &rest], 2)
        }
    }

    /// `x * weight^T` for `x` `[batch, cols]` f32 and a weight `[rows, cols]`: `[batch, rows]` f32.
    pub fn linear(&mut self, x: &Value, weight: &Weight, numerics: Numerics) -> Value {
        let (batch, cols) = (x.dims()[0], x.dims()[1]);
        let rows = weight.w.dims()[0];
        assert_eq!(
            weight.w.dims()[1],
            cols,
            "linear: weight {} for {cols} columns",
            weight.w.ty
        );
        assert_eq!(x.dtype(), DType::F32, "linear: activations are f32");
        let dequantized = if weight.w.dtype() == DType::I8 {
            let scales = weight.scales.as_ref().expect("an int8 weight needs scales");
            let groups = scales.dims()[1];
            assert!(
                scales.dims()[0] == rows && groups > 0 && cols % groups == 0,
                "linear: scales {} for {rows}x{cols}",
                scales.ty
            );
            let wf = self.convert(&weight.w, DType::F32);
            let wf3 = self.reshape(&wf, &[rows, groups, cols / groups]);
            let s3 = self.broadcast_in_dim(scales, &[rows, groups, cols / groups], &[0, 1]);
            let scaled = self.mul(&wf3, &s3);
            self.reshape(&scaled, &[rows, cols])
        } else {
            weight.w.clone()
        };
        let operand = numerics.operand();
        let lhs = self.convert(x, operand);
        let rhs = self.convert(&dequantized, operand);
        let dims = DotDims {
            lhs_contract: vec![1],
            rhs_contract: vec![1],
            ..Default::default()
        };
        let _ = batch;
        self.dot_general(&lhs, &rhs, &dims, DType::F32, numerics.precision())
    }

    /// `x + bias` for `x` `[batch, rows]` and `bias` `[rows]`.
    pub fn add_bias(&mut self, x: &Value, bias: &Value) -> Value {
        let b = self.broadcast_in_dim(bias, x.dims(), &[1]);
        self.add(x, &b)
    }

    /// SwiGLU: `gate * sigmoid(gate) * up`, written as `gate / (1 + exp(-gate)) * up` like the CPU backend.
    pub fn swiglu(&mut self, gate: &Value, up: &Value) -> Value {
        let neg = self.neg(gate);
        let e = self.exp(&neg);
        let denom = self.add_const(&e, 1.0);
        let silu = self.div(gate, &denom);
        self.mul(&silu, up)
    }

    /// GeGLU with the tanh approximation of GELU: `0.5 g (1 + tanh(0.7978846 (g + 0.044715 g^3))) up`.
    pub fn geglu(&mut self, gate: &Value, up: &Value) -> Value {
        let g2 = self.mul(gate, gate);
        let g3 = self.mul(&g2, gate);
        let cubic = self.mul_const(&g3, 0.044_715);
        let inner = self.add(gate, &cubic);
        let arg = self.mul_const(&inner, 0.797_884_6);
        let t = self.tanh(&arg);
        let one_plus = self.add_const(&t, 1.0);
        let half_g = self.mul_const(gate, 0.5);
        let gelu = self.mul(&half_g, &one_plus);
        self.mul(&gelu, up)
    }

    /// gpt-oss SwiGLU: `min(g, limit) * sigmoid(alpha * min(g, limit)) * (clamp(u, -limit, limit) + 1)`.
    pub fn swiglu_oai(&mut self, gate: &Value, up: &Value, alpha: f32, limit: f32) -> Value {
        let lim = self.constant(f64::from(limit), gate.dims(), DType::F32);
        let g = self.min(gate, &lim);
        let scaled = self.mul_const(&g, f64::from(alpha));
        let sig = self.logistic(&scaled);
        let glu = self.mul(&g, &sig);
        let lo = self.constant(-f64::from(limit), up.dims(), DType::F32);
        let clamped = self.clamp(&lo, up, &lim);
        let plus_one = self.add_const(&clamped, 1.0);
        self.mul(&glu, &plus_one)
    }

    /// Index of the largest value of every row of `x` (`[rows, vocab]` f32): the first one on a tie, NaN ignored. `[rows]` i32.
    pub fn argmax_rows(&mut self, x: &Value) -> Value {
        let (rows, vocab) = (x.dims()[0], x.dims()[1]);
        let nan = self.is_nan(x);
        let floor = self.infinity(true, x.dims(), DType::F32);
        let clean = self.select(&nan, &floor, x);
        let best = self.reduce(&clean, Reducer::Max, &[1]);
        let best_b = self.unreduce(&best, x.dims(), &[1]);
        let hit = self.compare(&clean, &best_b, Cmp::Eq);
        let index = self.iota(x.dims(), 1, DType::I32);
        let none = self.constant(vocab as f64, x.dims(), DType::I32);
        let candidate = self.select(&hit, &index, &none);
        let first = self.reduce(&candidate, Reducer::Min, &[1]);
        // an all-NaN row has no hit among the real values; its max is -inf, which every clean value equals: index 0
        let _ = rows;
        first
    }

    /// Softmax routing of mixture-of-experts layers: the `k` largest of `softmax(logits)` per row (ties to the lower index) as
    /// `(weights [rows, k] f32, ids [rows, k] i32)`, renormalised over the selected experts when `normalize`, then times `scale`.
    pub fn moe_topk(
        &mut self,
        logits: &Value,
        k: usize,
        normalize: bool,
        scale: f32,
    ) -> (Value, Value) {
        let (rows, experts) = (logits.dims()[0], logits.dims()[1]);
        let mut probs = self.softmax_last(logits);
        let index = self.iota(probs.dims(), 1, DType::I32);
        let minus_one = self.constant(-1.0, probs.dims(), DType::F32);
        let mut weights = Vec::new();
        let mut ids = Vec::new();
        for _ in 0..k {
            let best = self.reduce(&probs, Reducer::Max, &[1]);
            let best_b = self.unreduce(&best, probs.dims(), &[1]);
            let hit = self.compare(&probs, &best_b, Cmp::Eq);
            let none = self.constant(experts as f64, probs.dims(), DType::I32);
            let candidate = self.select(&hit, &index, &none);
            let first = self.reduce(&candidate, Reducer::Min, &[1]);
            let first_b = self.unreduce(&first, probs.dims(), &[1]);
            let chosen = self.compare(&index, &first_b, Cmp::Eq);
            probs = self.select(&chosen, &minus_one, &probs);
            weights.push(self.reshape(&best, &[rows, 1]));
            ids.push(self.reshape(&first, &[rows, 1]));
        }
        let weight_refs: Vec<&Value> = weights.iter().collect();
        let id_refs: Vec<&Value> = ids.iter().collect();
        let mut w = self.concatenate(&weight_refs, 1);
        let i = self.concatenate(&id_refs, 1);
        if normalize {
            let total = self.sum(&w, &[1]);
            let floor = self.constant(1e-8, &[rows], DType::F32);
            let safe = self.max(&total, &floor);
            let safe_b = self.unreduce(&safe, w.dims(), &[1]);
            w = self.div(&w, &safe_b);
        }
        if scale != 1.0 {
            w = self.mul_const(&w, f64::from(scale));
        }
        (w, i)
    }

    /// `out[t] = sum_j weights[t * k + j] * y[t * k + j]` for `y` `[tokens * k, hidden]` and `weights` `[tokens * k]`.
    pub fn moe_combine(&mut self, y: &Value, weights: &Value, tokens: usize, k: usize) -> Value {
        let hidden = y.dims()[1];
        let (t, k, h) = (tokens as i64, k as i64, hidden);
        let y3 = self.reshape(y, &[t, k, h]);
        let w2 = self.reshape(weights, &[t, k]);
        let w3 = self.broadcast_in_dim(&w2, &[t, k, h], &[0, 1]);
        let weighted = self.mul(&y3, &w3);
        self.sum(&weighted, &[1])
    }

    /// Split the query projection of gated attention (`[rows, heads * 2 * dk]`) into queries and gates (both `[rows, heads * dk]`):
    /// `[q_h, gate_h]` side by side for every head when `per_head`, otherwise all queries, then all gates.
    pub fn split_q_gate(
        &mut self,
        projected: &Value,
        heads: usize,
        dk: usize,
        per_head: bool,
    ) -> (Value, Value) {
        let (rows, h, d) = (projected.dims()[0], heads as i64, dk as i64);
        if per_head {
            let p = self.reshape(projected, &[rows, h, 2, d]);
            let q = self.slice(&p, &[0, 0, 0, 0], &[rows, h, 1, d]);
            let g = self.slice(&p, &[0, 0, 1, 0], &[rows, h, 2, d]);
            (
                self.reshape(&q, &[rows, h * d]),
                self.reshape(&g, &[rows, h * d]),
            )
        } else {
            let p = self.reshape(projected, &[rows, 2, h * d]);
            let q = self.slice(&p, &[0, 0, 0], &[rows, 1, h * d]);
            let g = self.slice(&p, &[0, 1, 0], &[rows, 2, h * d]);
            (
                self.reshape(&q, &[rows, h * d]),
                self.reshape(&g, &[rows, h * d]),
            )
        }
    }

    /// Paged attention over the cache (see the module comment of `backend/ops.rs` for the layout): `[rows, heads * value_dim]` f32.
    pub fn paged_attention(&mut self, spec: &AttentionSpec, inputs: &AttentionInputs<'_>) -> Value {
        let (r, h, kvh) = (spec.rows as i64, spec.heads as i64, spec.kv_heads as i64);
        let (dk, dv, p) = (
            spec.head_dim as i64,
            spec.value_dim as i64,
            spec.positions as i64,
        );
        let (bs, stride) = (spec.block_size as i64, spec.stride as i64);
        let group = h / kvh;
        assert!(
            h % kvh == 0 && p % bs == 0 && p <= stride * bs,
            "paged_attention: heads {h}/{kvh}, positions {p}, block {bs}, stride {stride}"
        );
        let nb = p / bs;
        let numerics = spec.numerics;

        // slot (row of the pool) of every key position of every row: block * block_size + offset in the block + layer start
        let tab = self.slice(inputs.tables, &[0, 0], &[r, nb]);
        let zero = self.constant(0.0, &[r, nb], DType::I32);
        let tab = self.max(&tab, &zero);
        let tab3 = self.broadcast_in_dim(&tab, &[r, nb, bs], &[0, 1]);
        let base = self.mul_const(&tab3, bs as f64);
        let within = self.iota(&[r, nb, bs], 2, DType::I32);
        let slot = self.add(&base, &within);
        let layer = self.broadcast_in_dim(inputs.layer_row0, &[r, nb, bs], &[]);
        let slot = self.add(&slot, &layer);
        let ids = self.reshape(&slot, &[r * p]);

        let gathered_k = self.gather_rows(inputs.key_pool, &ids);
        let keys = self.reshape(&gathered_k, &[r, p, kvh, dk]);
        let gathered_v = self.gather_rows(inputs.value_pool, &ids);
        let values = self.reshape(&gathered_v, &[r, p, kvh, dv]);

        // scores [rows, kv_heads, group, positions]
        let q4 = self.reshape(inputs.q, &[r, kvh, group, dk]);
        let operand = numerics.operand();
        let q_op = self.convert(&q4, operand);
        let k_op = self.convert(&keys, operand);
        let score_dims = DotDims {
            lhs_batch: vec![0, 1],
            rhs_batch: vec![0, 2],
            lhs_contract: vec![3],
            rhs_contract: vec![3],
        };
        let raw = self.dot_general(&q_op, &k_op, &score_dims, DType::F32, numerics.precision());
        let shape = [r, kvh, group, p];
        let mut scores = self.mul_const(&raw, f64::from(spec.scale));

        let position = self.iota(&shape, 3, DType::I32);
        let n = self.broadcast_in_dim(inputs.lengths, &shape, &[0]);
        if let Some(slopes) = inputs.alibi {
            let by_group = self.reshape(slopes, &[kvh, group]);
            let slope = self.broadcast_in_dim(&by_group, &shape, &[1, 2]);
            let last = self.add_const(&n, -1.0);
            let distance = self.sub(&position, &last);
            let distance = self.convert(&distance, DType::F32);
            let bias = self.mul(&slope, &distance);
            scores = self.add(&scores, &bias);
        }
        if let Some(cap) = spec.softcap {
            let scaled = self.mul_const(&scores, 1.0 / f64::from(cap));
            let squashed = self.tanh(&scaled);
            scores = self.mul_const(&squashed, f64::from(cap));
        }

        // visible positions: start..n, where start is the beginning of the chunk, of the window, or 0
        let start = match (spec.chunk, spec.window) {
            (Some(chunk), _) => {
                let last = self.add_const(&n, -1.0);
                let c = self.constant(chunk as f64, &shape, DType::I32);
                let blocks = self.div(&last, &c);
                self.mul(&blocks, &c)
            }
            (None, Some(window)) => {
                let w = self.constant(window as f64, &shape, DType::I32);
                let from = self.sub(&n, &w);
                let zero = self.constant(0.0, &shape, DType::I32);
                self.max(&from, &zero)
            }
            (None, None) => self.constant(0.0, &shape, DType::I32),
        };
        let after_start = self.compare(&position, &start, Cmp::Ge);
        let before_end = self.compare(&position, &n, Cmp::Lt);
        let visible = self.and(&after_start, &before_end);
        let floor = self.infinity(true, &shape, DType::F32);
        let masked = self.select(&visible, &scores, &floor);

        // softmax over the positions; a sink adds exp(sink) to the denominator without a value
        let mut best = self.reduce(&masked, Reducer::Max, &[3]);
        let sink = inputs.sinks.map(|s| {
            let by_group = self.reshape(s, &[kvh, group]);
            self.broadcast_in_dim(&by_group, &[r, kvh, group], &[1, 2])
        });
        if let Some(sink) = &sink {
            best = self.max(&best, sink);
        }
        let best_b = self.unreduce(&best, &shape, &[3]);
        let shifted = self.sub(&masked, &best_b);
        let e = self.exp(&shifted);
        let mut total = self.sum(&e, &[3]);
        if let Some(sink) = &sink {
            let d = self.sub(sink, &best);
            let sink_e = self.exp(&d);
            total = self.add(&total, &sink_e);
        }
        let total_b = self.unreduce(&total, &shape, &[3]);
        let weights = self.div(&e, &total_b);

        let w_op = self.convert(&weights, operand);
        let v_op = self.convert(&values, operand);
        let out_dims = DotDims {
            lhs_batch: vec![0, 1],
            rhs_batch: vec![0, 2],
            lhs_contract: vec![3],
            rhs_contract: vec![1],
        };
        let out = self.dot_general(&w_op, &v_op, &out_dims, DType::F32, numerics.precision());
        self.reshape(&out, &[r, h * dv])
    }

    /// The cache pool with the rows `ids[i]` replaced by `rows[i]` (`[n, kv]` f32 converted to the pool's f16, round to nearest even).
    pub fn cache_write(&mut self, pool: &Value, ids: &Value, rows: &Value) -> Value {
        let encoded = self.convert(rows, pool.dtype());
        self.scatter_rows(pool, ids, &encoded, false, true)
    }
}

/// Dimensions as a fixed array for the common one-dimensional case.
pub fn flat(n: usize) -> [i64; 1] {
    dims1(n)
}
