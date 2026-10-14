//! Self-test cases for the prefill speed-ups of the CUDA backend (docs/performance-notes.md): the per-step cache of the attention host
//! data, the fusions around the Marlin GEMMs (residual add into the RMS norm, FP16 copy of the norm and attention results) and the
//! Marlin launch configuration for prompt batches. Every switch is flipped at run time and the device is compared with itself: the
//! fusions and the cache must not change a single bit, the Marlin variant must stay within the tolerance of the fast kernels.
use super::*;
use crate::backend::cuda::{prefill_switch, set_prefill_switch, PrefillSwitch};

const HEADS: usize = 8;
const KV_HEADS: usize = 2;
const DIM: usize = 64;
const BLOCK: usize = 16;
const STRIDE: usize = 16;

/// One run of a check on a fresh stream: the outputs to compare.
type VariantRun<'a> = dyn Fn(&dyn Backend, &mut dyn Stream) -> Result<Vec<f32>, BackendError> + 'a;

impl Tester<'_> {
    pub(super) fn prefill_cases(&mut self) {
        // attention host data: one prompt, two prompts, a prompt next to single decode rows (the layouts the run detection tells apart)
        for runs in [&[48usize][..], &[40, 20][..], &[40, 3][..], &[130][..]] {
            let r = self.prefill_attention(runs, &[PrefillSwitch::StepCache]);
            self.record("prefill step cache", &format!("attention runs {runs:?}"), r);
        }
        // the residual add and the FP16 copy of the norm: add -> norm -> Marlin, and chains with another weight format in between
        for (mode, label) in [
            (Chain::Norm, "add + norm"),
            (Chain::Marlin, "add + norm + marlin"),
            (Chain::Q4k, "add + norm + q4_k + marlin"),
            (Chain::Perm, "add + norm + act-order + marlin"),
        ] {
            for batch in [16usize, 40, 129] {
                let r = self.prefill_chain(batch, mode);
                self.record("prefill fusion", &format!("{label} batch {batch}"), r);
            }
        }
        // the attention result handed to the output projection as FP16
        for rows in [24usize, 48, 130] {
            let r = self.prefill_attention_projection(rows);
            self.record("prefill fusion", &format!("attention + o_proj {rows} rows"), r);
        }
        // Marlin with the M tail inside the last tile: against the CPU reference, with the fast-kernel tolerance
        let saved = prefill_switch(PrefillSwitch::MarlinCfg);
        set_prefill_switch(PrefillSwitch::MarlinCfg, 1);
        for kind in self.kinds(&[0x8000_0010u32, 0x8000_0011, 0x8000_0012, 0x8000_0014]) {
            for batch in [65usize, 100, 129, 300, 1000] {
                let with_bias = batch == 129;
                let r = self.linear(kind, 384, 1024, batch, with_bias).map(|mut o| {
                    o.tol = o.tol.max(Self::FAST_TOL);
                    o
                });
                self.record(
                    &format!("prefill marlin {}", format_name(kind)),
                    &format!("384x1024 batch {batch}{}", if with_bias { " +bias" } else { "" }),
                    r,
                );
            }
        }
        set_prefill_switch(PrefillSwitch::MarlinCfg, saved);
    }

    /// Run `run` with every switch in `switches` off and on (fresh stream each time) and report the largest difference of the outputs.
    fn prefill_variants(
        &mut self,
        switches: &[PrefillSwitch],
        run: &VariantRun<'_>,
    ) -> Result<Outcome, BackendError> {
        let saved: Vec<i32> = switches.iter().map(|&s| prefill_switch(s)).collect();
        let mut outputs = Vec::new();
        let mut ms = 0.0;
        let mut result = Ok(());
        for on in [0, 1] {
            for &s in switches {
                set_prefill_switch(s, on);
            }
            let attempt = self.device.create_stream().and_then(|mut stream| {
                let start = Instant::now();
                let out = run(self.device, stream.as_mut())?;
                if on == 1 {
                    ms = start.elapsed().as_secs_f64() * 1e3;
                }
                Ok(out)
            });
            match attempt {
                Ok(out) => outputs.push(out),
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        }
        for (&s, &v) in switches.iter().zip(&saved) {
            set_prefill_switch(s, v);
        }
        result?;
        let (plain, fused) = (&outputs[0], &outputs[1]);
        let diff = if plain.len() != fused.len() || plain.is_empty() {
            f64::INFINITY
        } else {
            plain
                .iter()
                .zip(fused)
                .map(|(a, b)| if a.to_bits() == b.to_bits() { 0.0 } else { (f64::from(*a) - f64::from(*b)).abs().max(1e-30) })
                .fold(0.0, |m: f64, d| if d.is_nan() { f64::INFINITY } else { m.max(d) })
        };
        // the variants compute the same operations: not one bit may differ
        Ok(Outcome { device: diff, cpu: 0.0, tol: 0.0, cpu_tol: 0.0, ms })
    }

    /// Prompt attention over `runs` (token counts of the prompts in the batch) through two layers.
    fn prefill_attention(&mut self, runs: &[usize], switches: &[PrefillSwitch]) -> Result<Outcome, BackendError> {
        let case = AttentionInputs::new(&mut self.rng, runs);
        self.prefill_variants(switches, &|backend, stream| {
            let outs = case.run(backend, stream, 2)?;
            let mut all = Vec::new();
            for out in &outs {
                all.extend(get(stream, out, case.lengths.len() * HEADS * DIM)?);
            }
            Ok(all)
        })
    }

    /// Prompt attention followed by the output projection (Marlin, added to the residual stream) and the norm after it.
    fn prefill_attention_projection(&mut self, rows: usize) -> Result<Outcome, BackendError> {
        let case = AttentionInputs::new(&mut self.rng, &[rows]);
        let hidden = 512;
        let kind = 0x8000_0011u32;
        let weight = random_kind(&mut self.rng, kind, hidden, HEADS * DIM);
        let residual = self.rng.vector(rows * hidden, 1.0);
        let norm: Vec<f32> = self.rng.vector(hidden, 0.5).iter().map(|v| v + 1.0).collect();
        self.prefill_variants(
            &[PrefillSwitch::FuseAttnF16, PrefillSwitch::FuseNormF16, PrefillSwitch::FuseAddNorm],
            &|backend, stream| {
                let w = backend.upload_weight_scaled("selftest", &weight, QuantType::from_gguf_type_id(kind), hidden, HEADS * DIM, None, 1.0)?;
                let nw = put(backend, stream, &norm)?;
                let mut res = put(backend, stream, &residual)?;
                // the Marlin scratch exists once a Marlin GEMM has run (the attention result goes there); in a model q, k and v come first
                let warm_in = put(backend, stream, &vec![0.5; rows * HEADS * DIM])?;
                let mut warm_out = backend.allocate_buffer(rows * hidden * 4)?;
                stream.quantized_linear(&mut warm_out, &warm_in, &w, None)?;
                let attn = case.run(backend, stream, 1)?.pop().ok_or_else(|| BackendError::DriverError("no attention buffer".into()))?;
                if !stream.quantized_linear_add(&mut res, &attn, &w, None, 1.0)? {
                    return Err(BackendError::UnsupportedOperation("quantized_linear_add".into()));
                }
                let mut normed = backend.allocate_buffer(rows * hidden * 4)?;
                stream.rms_norm(&mut normed, &res, &nw, 1e-6)?;
                let mut all = get(stream, &res, rows * hidden)?;
                all.extend(get(stream, &normed, rows * hidden)?);
                Ok(all)
            },
        )
    }

    /// `x += o_proj(a)`, `n = rms_norm(x)`, then the consumers of `n` named by `chain`.
    fn prefill_chain(&mut self, batch: usize, chain: Chain) -> Result<Outcome, BackendError> {
        let hidden = 512;
        let o_kind = 0x8000_0011u32;
        let o_weight = random_kind(&mut self.rng, o_kind, hidden, 1024);
        let a = self.rng.vector(batch * 1024, 1.0);
        let residual = self.rng.vector(batch * hidden, 1.0);
        let norm: Vec<f32> = self.rng.vector(hidden, 0.5).iter().map(|v| v + 1.0).collect();
        let bias = self.rng.vector(hidden, 0.5);
        let next_kind = if chain == Chain::Perm { 0x8000_0015u32 } else { 0x8000_0011 };
        let next_rows = 640;
        let next_weight = random_kind(&mut self.rng, next_kind, next_rows, hidden);
        let q4k_weight = random_kind(&mut self.rng, 12, 256, hidden);
        self.claim = Some(o_kind);
        self.prefill_variants(&[PrefillSwitch::FuseAddNorm, PrefillSwitch::FuseNormF16], &|backend, stream| {
            let o = backend.upload_weight_scaled("selftest", &o_weight, QuantType::from_gguf_type_id(o_kind), hidden, 1024, Some(&bias), 1.0)?;
            let next = backend.upload_weight_scaled("selftest", &next_weight, QuantType::from_gguf_type_id(next_kind), next_rows, hidden, None, 1.0)?;
            let q4k = backend.upload_weight_scaled("selftest", &q4k_weight, QuantType::from_gguf_type_id(12), 256, hidden, None, 1.0)?;
            stream.synchronize()?;
            let x = put(backend, stream, &a)?;
            let nw = put(backend, stream, &norm)?;
            let mut res = put(backend, stream, &residual)?;
            // all buffers stay alive to the end: a freed allocation could be handed out again at the same address
            let mut normed = backend.allocate_buffer(batch * hidden * 4)?;
            let mut outs = [backend.allocate_buffer(batch * next_rows * 4)?, backend.allocate_buffer(batch * next_rows * 4)?];
            let mut mid = backend.allocate_buffer(batch * 256 * 4)?;
            if !stream.quantized_linear_add(&mut res, &x, &o, o.bias.as_ref(), 0.5)? {
                return Err(BackendError::UnsupportedOperation("quantized_linear_add".into()));
            }
            stream.rms_norm(&mut normed, &res, &nw, 1e-6)?;
            let mut count = 0;
            if chain != Chain::Norm {
                if chain == Chain::Q4k {
                    stream.quantized_linear(&mut mid, &normed, &q4k, None)?;
                }
                stream.quantized_linear(&mut outs[0], &normed, &next, None)?;
                stream.quantized_linear(&mut outs[1], &normed, &next, None)?;
                count = 2;
            }
            let mut all = get(stream, &res, batch * hidden)?;
            all.extend(get(stream, &normed, batch * hidden)?);
            for out in outs.iter().take(count) {
                all.extend(get(stream, out, batch * next_rows)?);
            }
            if chain == Chain::Q4k {
                all.extend(get(stream, &mid, batch * 256)?);
            }
            Ok(all)
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Chain {
    /// add, norm
    Norm,
    /// add, norm, Marlin GEMM of the same format family
    Marlin,
    /// add, norm, a Q4_K GEMM, then a Marlin GEMM (the FP16 copy must not be taken for another layout)
    Q4k,
    /// add, norm, a GEMM with an activation permutation (the FP16 copy is in natural order)
    Perm,
}

/// Random prompts and their paged f16 cache.
struct AttentionInputs {
    k: Vec<f32>,
    v: Vec<f32>,
    q: Vec<f32>,
    slots: Vec<usize>,
    tables: Vec<i64>,
    lengths: Vec<i64>,
    pool_blocks: usize,
}

impl AttentionInputs {
    fn new(rng: &mut Lcg, runs: &[usize]) -> Self {
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
        Self {
            k: rng.vector(rows * KV_HEADS * DIM, 1.0),
            v: rng.vector(rows * KV_HEADS * DIM, 1.0),
            q: rng.vector(rows * HEADS * DIM, 1.0),
            slots,
            tables,
            lengths,
            pool_blocks: runs.len() * STRIDE,
        }
    }

    /// Fills a cache and runs `layers` attention calls inside one step; returns their output buffers (not downloaded: the caller may
    /// hand the last one to a GEMM first).
    fn run(&self, backend: &dyn Backend, stream: &mut dyn Stream, layers: usize) -> Result<Vec<DeviceBuffer>, BackendError> {
        let rows = self.lengths.len();
        let cache = backend.allocate_kv_cache(layers, self.pool_blocks, BLOCK, KV_HEADS, DIM, DIM, 1)?;
        let kb = put(backend, stream, &self.k)?;
        let vb = put(backend, stream, &self.v)?;
        let qb = put(backend, stream, &self.q)?;
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
        for layer in 0..layers {
            stream.kv_cache_write(&cache, layer, &self.slots, &kb, &vb)?;
        }
        let mut outs = Vec::new();
        stream.begin_step();
        for layer in 0..layers {
            let mut out = backend.allocate_buffer(rows * HEADS * DIM * 4)?;
            if let Err(e) = stream.attention_paged(&mut out, &qb, &cache, &self.tables, &self.lengths, layer, &params) {
                stream.end_step();
                return Err(e);
            }
            outs.push(out);
        }
        stream.end_step();
        Ok(outs)
    }
}
