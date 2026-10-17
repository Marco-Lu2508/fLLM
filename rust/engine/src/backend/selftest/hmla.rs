//! Self-test cases: latent-cache multi-head latent attention (CUDA backend, `FLLM_MLA=latent`, `cuda/mla_latent.cu`, `cuda/mla_decode.cu`).
//!
//! Every case builds random MLA data (kv_b matrix, latents, rope keys, queries), writes the latent cache through the device kernels and checks the
//! four stages of one attention layer separately against f64 references (`crate::mla_latent::reference`), then the whole chain against the expanded
//! formulation (per-head keys and values, ordinary attention) that the CPU decoder and the default CUDA path use:
//!
//! | check                    | compares                                                                                       | tolerance (relative to the largest value of the head) |
//! |--------------------------|------------------------------------------------------------------------------------------------|-------------------------------------------------------|
//! | `mla latent kv`          | cache row `[latent \| rotated rope key]` (FP32)                                                 | 1e-5                                                  |
//! | `mla latent query`       | absorbed queries (FP16) against `q_nope W_UK` and the rotated `q_pe`                            | 2e-3 (FP16 rounding of the result)                    |
//! | `mla latent attention`   | tensor-core attention over the cache, fed with the device's own FP16 queries                    | 6e-3 (FP16 probabilities and output)                  |
//! | `mla latent attn ref`    | the CUDA-core reference kernel on the same inputs (tells a kernel defect from a layout defect)  | 2e-3                                                  |
//! | `mla latent output`      | `W_UV` expansion of the device's own FP16 attention output                                      | 5e-4                                                  |
//! | `mla latent vs expanded` | the whole chain against the expanded formulation of the unrounded data                         | 1e-2                                                  |
//!
//! The references read the cache as the device stores it (`KvLayout::encode_slot` / `decode_slot` for the quantized formats), so the quantization of
//! the fp8 / q8_0 / int8 caches does not enter the tolerances. A defect in indexing, masking, scaling or the rotary tables is off by far more.
//! `FLLM_SELFTEST_BENCH=1` adds timings: the latent attention against the expanded attention of the same layer at 4k contexts, with the cache sizes.
use super::*;
use crate::backend::{AttentionParams, MlaShape, RopeParams};
use crate::kv_format::{pack_kind, KvFormat, KvLayout};
use crate::mla_latent::reference as rf;
use crate::mla_latent::LATENT_VALUE_DIM;

const BLOCK: usize = 64;

/// One layer shape and the batch: per sequence `(tokens cached before the rows, rows)`; a row is a query at the position of its token.
#[derive(Clone)]
struct MlaCase {
    heads: usize,
    nope: usize,
    rope: usize,
    value: usize,
    kv_lora: usize,
    neox: bool,
    runs: Vec<(usize, usize)>,
}

impl MlaCase {
    fn new(heads: usize, nope: usize, rope: usize, value: usize, kv_lora: usize, runs: &[(usize, usize)]) -> Self {
        Self { heads, nope, rope, value, kv_lora, neox: false, runs: runs.to_vec() }
    }
    fn label(&self, fmt: KvFormat) -> String {
        format!(
            "{} heads {}+{} rope {} v {} {}{} {:?}",
            self.heads,
            self.nope,
            self.kv_lora,
            self.rope,
            self.value,
            fmt.name(),
            if self.neox { " neox" } else { "" },
            self.runs
        )
    }
    fn layer(&self) -> rf::Layer {
        rf::Layer {
            heads: self.heads,
            nope: self.nope,
            rope: self.rope,
            value: self.value,
            kv_lora: self.kv_lora,
            neox: self.neox,
            scale: 1.0 / ((self.nope + self.rope) as f64).sqrt(),
        }
    }
    fn shape(&self) -> MlaShape {
        MlaShape { heads: self.heads, nope: self.nope, rope: self.rope, value: self.value, kv_lora: self.kv_lora }
    }
}

/// The layer shapes with the cache formats they run in. Decode batches, prompts (16 or more rows of one sequence take the row-packed kernel), a
/// prompt behind a cached context, mixes of both, a context long enough for the split-KV path, head counts that are not a multiple of 16.
fn cases() -> Vec<(MlaCase, Vec<KvFormat>)> {
    let all = vec![KvFormat::F16, KvFormat::Fp8E4m3, KvFormat::Q8_0];
    vec![
        // DeepSeek-V2-Lite: 512 + 64, 128 + 128 per head
        (MlaCase::new(16, 128, 64, 128, 512, &[(36, 1), (299, 1), (0, 1)]), all.clone()),
        (MlaCase::new(4, 128, 64, 128, 512, &[(0, 40)]), vec![KvFormat::F16, KvFormat::Fp8E4m3]),
        (MlaCase::new(4, 128, 64, 128, 512, &[(130, 20), (6, 1), (0, 17)]), vec![KvFormat::F16]),
        // 24 heads: the second head tile of a decode row is half empty
        (MlaCase::new(24, 128, 64, 128, 512, &[(49, 1), (1, 1)]), vec![KvFormat::F16]),
        // MiniCPM3-like: 256 + 32
        (MlaCase::new(8, 64, 32, 64, 256, &[(1499, 1)]), all.clone()),
        (MlaCase { neox: true, ..MlaCase::new(8, 64, 32, 64, 256, &[(20, 18), (63, 1), (64, 1), (65, 1)]) }, vec![KvFormat::F16, KvFormat::Q8_0]),
        // small widths
        (MlaCase { neox: true, ..MlaCase::new(4, 32, 64, 32, 128, &[(3, 1), (70, 1), (0, 33)]) }, vec![KvFormat::F16, KvFormat::Int8]),
        (MlaCase::new(3, 16, 32, 32, 64, &[(40, 1), (9, 20)]), vec![KvFormat::F16, KvFormat::Fp8E4m3]),
    ]
}

fn put_bytes(backend: &dyn Backend, stream: &mut dyn Stream, bytes: &[u8]) -> Result<DeviceBuffer, BackendError> {
    let mut buf = backend.allocate_buffer(bytes.len().max(4))?;
    stream.upload(&mut buf, bytes)?;
    Ok(buf)
}

fn get_f16(stream: &mut dyn Stream, buf: &DeviceBuffer, count: usize) -> Result<Vec<f32>, BackendError> {
    let mut bytes = vec![0u8; count * 2];
    stream.download(&mut bytes, buf)?;
    Ok(bytes.as_chunks::<2>().0.iter().map(|c| f16::from_le_bytes(*c).to_f32()).collect())
}

/// Largest error relative to the largest reference value of the segment (`seg` values: one head of one row), logged for the failure message.
fn segment_error(have: &[f32], want: &[f64], seg: usize) -> f64 {
    let scale: Vec<f64> = want.chunks(seg).flat_map(|c| std::iter::repeat_n(c.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-6), c.len())).collect();
    worst(have, want, &scale)
}

fn outcome(device: f64, tol: f64, ms: f64) -> Outcome {
    Outcome { device, cpu: 0.0, tol, cpu_tol: 0.0, ms }
}

impl Tester<'_> {
    pub(super) fn hmla_cases(&mut self) {
        for (case, formats) in cases() {
            for fmt in formats {
                let label = case.label(fmt);
                match self.mla_latent_case(&case, fmt) {
                    Ok(checks) => {
                        for (op, o) in checks {
                            self.record(op, &label, Ok(o));
                        }
                    }
                    Err(e) => self.record("mla latent", &label, Err(e)),
                }
            }
        }
        if bench_enabled() {
            for (heads, fmt) in [(16usize, KvFormat::F16), (16, KvFormat::Fp8E4m3), (128, KvFormat::F16)] {
                let label = format!("{heads} heads 4095 tokens x 8 seqs {}", fmt.name());
                let r = self.mla_bench(heads, fmt);
                self.record("mla latent bench", &label, r);
            }
        }
    }

    fn mla_latent_case(&mut self, c: &MlaCase, fmt: KvFormat) -> Result<Vec<(&'static str, Outcome)>, BackendError> {
        let shape = c.shape();
        let mut stream = self.device.create_stream()?;
        if !stream.supports_mla_latent(&shape, fmt) {
            return Err(BackendError::UnsupportedOperation(format!("latent MLA for {}+{} in a {} cache", c.kv_lora, c.rope, fmt.name())));
        }
        let layer = c.layer();
        let (heads, dk, width) = (c.heads, c.nope + c.rope, c.kv_lora + c.rope);
        // ---- data ----
        let w: Vec<f32> = self.rng.vector(heads * (c.nope + c.value) * c.kv_lora, 0.15).iter().map(|&v| f16::from_f32(v).to_f32()).collect();
        let max_len = c.runs.iter().map(|&(p, n)| p + n).max().unwrap_or(1);
        let (cos, sin) = rf::rope_tables(max_len, c.rope, 10_000.0);
        let seq_len: Vec<usize> = c.runs.iter().map(|&(p, n)| p + n).collect();
        let blocks: Vec<usize> = seq_len.iter().map(|l| l.div_ceil(BLOCK)).collect();
        let stride = blocks.iter().copied().max().unwrap_or(1) + 1;
        let needed: usize = blocks.iter().sum();
        let pool_blocks = needed + 2;
        // block ids in reverse order, so the table of a sequence is not the identity
        let mut next = pool_blocks - 1;
        let mut tables: Vec<Vec<i64>> = Vec::new();
        for &b in &blocks {
            let mut t = vec![0i64; stride];
            for e in t.iter_mut().take(b) {
                *e = next as i64;
                next -= 1;
            }
            tables.push(t);
        }
        let slot = |s: usize, t: usize| tables[s][t / BLOCK] as usize * BLOCK + t % BLOCK;
        let mut raw: Vec<f32> = Vec::new(); // kv_a rows of all tokens: [latent | rope key before the rotation]
        let mut latent: Vec<f32> = Vec::new();
        let mut positions_all: Vec<usize> = Vec::new();
        let mut slots_all: Vec<usize> = Vec::new();
        let mut ref_rows: Vec<Vec<Vec<f32>>> = Vec::new(); // per sequence, per token: the f32 cache row
        for (s, &len) in seq_len.iter().enumerate() {
            let mut rows = Vec::new();
            for t in 0..len {
                let lat = self.rng.vector(c.kv_lora, 1.0);
                let pe = self.rng.vector(c.rope, 1.0);
                raw.extend(&lat);
                raw.extend(&pe);
                latent.extend(&lat);
                positions_all.push(t);
                slots_all.push(slot(s, t));
                rows.push(rf::cache_row(&layer, &lat, &pe, t, &cos, &sin));
            }
            ref_rows.push(rows);
        }
        let rope_params = RopeParams {
            heads,
            kv_heads: 1,
            dim: c.rope,
            rope_dim: c.rope,
            neox: c.neox,
            cos: Arc::new(cos.clone()),
            sin: Arc::new(sin.clone()),
        };
        let n_all = positions_all.len();
        let kv = self.device.allocate_kv_cache(2, pool_blocks, BLOCK, 1, width, LATENT_VALUE_DIM, pack_kind(fmt, KvFormat::F16))?;
        let layer_idx = 1;
        let mut checks: Vec<(&'static str, Outcome)> = Vec::new();

        // ---- cache rows ----
        let kv_a = put(self.device, stream.as_mut(), &raw)?;
        let lat_buf = put(self.device, stream.as_mut(), &latent)?;
        let mut k_row = self.device.allocate_buffer(n_all * width * 4)?;
        let start = Instant::now();
        stream.mla_latent_kv(&mut k_row, &kv_a, &lat_buf, &positions_all, &shape, &rope_params)?;
        stream.synchronize()?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        let have = get(stream.as_mut(), &k_row, n_all * width)?;
        let want: Vec<f64> = ref_rows.iter().flatten().flatten().map(|&v| f64::from(v)).collect();
        checks.push(("mla latent kv", outcome(segment_error(&have, &want, width), 1e-5, ms)));
        let v_dummy = put(self.device, stream.as_mut(), &vec![0.0f32; n_all * LATENT_VALUE_DIM])?;
        stream.kv_cache_write(&kv, layer_idx, &slots_all, &k_row, &v_dummy)?;

        // the cache as the device holds it: the same rows through the layout encoder / decoder
        let slots_total = pool_blocks * BLOCK;
        let layout = KvLayout::new(fmt, slots_total, 1, width);
        let mut words = vec![0u16; layout.words];
        for (s, rows) in ref_rows.iter().enumerate() {
            for (t, row) in rows.iter().enumerate() {
                layout.encode_slot(&mut words, slot(s, t), row);
            }
        }
        let decoded: Vec<Vec<Vec<f32>>> = ref_rows
            .iter()
            .enumerate()
            .map(|(s, rows)| {
                (0..rows.len())
                    .map(|t| {
                        let mut out = vec![0.0f32; width];
                        layout.decode_slot(&words, slot(s, t), &mut out);
                        out
                    })
                    .collect()
            })
            .collect();

        // ---- the rows under test ----
        struct Row {
            seq: usize,
            pos: usize,
        }
        let mut rows_list: Vec<Row> = Vec::new();
        for (s, &(p, n)) in c.runs.iter().enumerate() {
            for i in 0..n {
                rows_list.push(Row { seq: s, pos: p + i });
            }
        }
        let rows = rows_list.len();
        let positions: Vec<usize> = rows_list.iter().map(|r| r.pos).collect();
        let lengths: Vec<i64> = rows_list.iter().map(|r| r.pos as i64 + 1).collect();
        let block_tables: Vec<i64> = rows_list.iter().flat_map(|r| tables[r.seq].clone()).collect();
        let q_full = self.rng.vector(rows * heads * dk, 1.0);
        let w_bits: Vec<u8> = w.iter().flat_map(|&v| f16::from_f32(v).to_le_bytes()).collect();
        let w_buf = put_bytes(self.device, stream.as_mut(), &w_bits)?;
        let q_buf = put(self.device, stream.as_mut(), &q_full)?;

        let mut q_lat = self.device.allocate_buffer(rows * heads * width * 2)?;
        let start = Instant::now();
        stream.mla_latent_query(&mut q_lat, &q_buf, &w_buf, &positions, &shape, &rope_params)?;
        stream.synchronize()?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        let q_have = get_f16(stream.as_mut(), &q_lat, rows * heads * width)?;
        let mut q_want: Vec<f64> = Vec::new();
        for (r, row) in rows_list.iter().enumerate() {
            q_want.extend(rf::absorbed_query(&layer, &w, &q_full[r * heads * dk..(r + 1) * heads * dk], row.pos, &cos, &sin));
        }
        checks.push(("mla latent query", outcome(segment_error(&q_have, &q_want, width), 2e-3, ms)));

        let params = AttentionParams {
            num_heads: heads,
            num_kv_heads: 1,
            head_dim: width,
            value_dim: LATENT_VALUE_DIM,
            stride,
            block_size: BLOCK,
            scale: layer.scale as f32,
            softcap: None,
            window: None,
            chunk: None,
            sinks: None,
            alibi_slopes: None,
        };
        // the attention checks take the queries the device produced (rounded to FP16) as their input
        let q_in: Vec<f64> = q_have.iter().map(|&v| f64::from(v)).collect();
        let mut o_want: Vec<f64> = Vec::new();
        for (r, row) in rows_list.iter().enumerate() {
            let (o_lat, _) = rf::attend_latent(&layer, &w, &q_in[r * heads * width..(r + 1) * heads * width], &decoded[row.seq], row.pos + 1);
            o_want.extend(o_lat);
        }
        let mut o_lat_tc: Option<DeviceBuffer> = None;
        for (op, reference, tol) in [("mla latent attention", false, 6e-3), ("mla latent attn ref", true, 2e-3)] {
            let mut o_lat = self.device.allocate_buffer(rows * heads * c.kv_lora * 2)?;
            let start = Instant::now();
            stream.mla_latent_attention(&mut o_lat, &q_lat, &kv, &block_tables, &lengths, layer_idx, &params, &shape, reference)?;
            stream.synchronize()?;
            let ms = start.elapsed().as_secs_f64() * 1e3;
            let o_have = get_f16(stream.as_mut(), &o_lat, rows * heads * c.kv_lora)?;
            checks.push((op, outcome(segment_error(&o_have, &o_want, c.kv_lora), tol, ms)));
            if !reference {
                o_lat_tc = Some(o_lat);
            }
        }
        let o_lat = o_lat_tc.ok_or_else(|| BackendError::DriverError("no attention output".into()))?;

        // ---- output expansion, from the device's own attention output ----
        let o_in = get_f16(stream.as_mut(), &o_lat, rows * heads * c.kv_lora)?;
        let mut out_want: Vec<f64> = Vec::new();
        for r in 0..rows {
            for h in 0..heads {
                for v in 0..c.value {
                    let wr = &w[(h * (c.nope + c.value) + c.nope + v) * c.kv_lora..(h * (c.nope + c.value) + c.nope + v + 1) * c.kv_lora];
                    let o = &o_in[(r * heads + h) * c.kv_lora..(r * heads + h + 1) * c.kv_lora];
                    out_want.push(wr.iter().zip(o).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum());
                }
            }
        }
        let mut out = self.device.allocate_buffer(rows * heads * c.value * 4)?;
        let start = Instant::now();
        stream.mla_latent_output(&mut out, &o_lat, &w_buf, rows, &shape)?;
        stream.synchronize()?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        let out_have = get(stream.as_mut(), &out, rows * heads * c.value)?;
        checks.push(("mla latent output", outcome(segment_error(&out_have, &out_want, c.value), 5e-4, ms)));

        // ---- the whole chain against the expanded formulation of the cache as stored ----
        let expanded: Vec<(Vec<f64>, Vec<f64>)> = decoded.iter().map(|rows| rf::expand(&layer, &w, rows)).collect();
        let mut exp_want: Vec<f64> = Vec::new();
        for (r, row) in rows_list.iter().enumerate() {
            let (k, v) = &expanded[row.seq];
            exp_want.extend(rf::attend_expanded(
                &layer,
                &q_full[r * heads * dk..(r + 1) * heads * dk],
                row.pos,
                &cos,
                &sin,
                &decoded[row.seq],
                k,
                v,
                row.pos + 1,
            ));
        }
        checks.push(("mla latent vs expanded", outcome(segment_error(&out_have, &exp_want, c.value), 1e-2, 0.0)));
        Ok(checks)
    }

    /// Timings (`FLLM_SELFTEST_BENCH=1`): attention over 8 sequences of 4095 tokens, latent kernel against the expanded attention of the same layer.
    fn mla_bench(&mut self, heads: usize, fmt: KvFormat) -> Result<Outcome, BackendError> {
        let c = MlaCase::new(heads, 128, 64, 128, 512, &[(4094, 1); 8]);
        let shape = c.shape();
        let mut stream = self.device.create_stream()?;
        if !stream.supports_mla_latent(&shape, fmt) {
            return Err(BackendError::UnsupportedOperation("latent MLA".into()));
        }
        let (width, dk, seqs, len) = (c.kv_lora + c.rope, c.nope + c.rope, 8usize, 4095usize);
        let blocks = len.div_ceil(BLOCK);
        let stride = blocks + 1;
        let pool_blocks = seqs * blocks + 2;
        let kv = self.device.allocate_kv_cache(1, pool_blocks, BLOCK, 1, width, LATENT_VALUE_DIM, pack_kind(fmt, KvFormat::F16))?;
        let tables: Vec<i64> = (0..seqs).flat_map(|s| (0..stride).map(move |b| if b < blocks { (s * blocks + b) as i64 } else { 0 })).collect();
        let lengths = vec![len as i64; seqs];
        let q_lat = self.device.allocate_buffer(seqs * heads * width * 2)?;
        let mut o_lat = self.device.allocate_buffer(seqs * heads * c.kv_lora * 2)?;
        let params = AttentionParams {
            num_heads: heads,
            num_kv_heads: 1,
            head_dim: width,
            value_dim: LATENT_VALUE_DIM,
            stride,
            block_size: BLOCK,
            scale: 1.0 / (dk as f32).sqrt(),
            softcap: None,
            window: None,
            chunk: None,
            sinks: None,
            alibi_slopes: None,
        };
        let time = |f: &mut dyn FnMut(&mut dyn Stream) -> Result<(), BackendError>, stream: &mut dyn Stream| -> Result<f64, BackendError> {
            f(stream)?; // warm-up
            stream.synchronize()?;
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let start = Instant::now();
                f(stream)?;
                stream.synchronize()?;
                best = best.min(start.elapsed().as_secs_f64() * 1e3);
            }
            Ok(best)
        };
        let latent_ms = time(&mut |s| s.mla_latent_attention(&mut o_lat, &q_lat, &kv, &tables, &lengths, 0, &params, &shape, false), stream.as_mut())?;
        // the expanded cache of the same layer: per-head keys (nope + rope) and values, 16-token pages
        let eblock = 16usize;
        let eblocks = len.div_ceil(eblock);
        let estride = eblocks + 1;
        let epool = seqs * eblocks + 2;
        let ekv = self.device.allocate_kv_cache(1, epool, eblock, heads, dk, c.value, pack_kind(fmt, fmt))?;
        let etables: Vec<i64> = (0..seqs).flat_map(|s| (0..estride).map(move |b| if b < eblocks { (s * eblocks + b) as i64 } else { 0 })).collect();
        let eq = self.device.allocate_buffer(seqs * heads * dk * 4)?;
        let mut eo = self.device.allocate_buffer(seqs * heads * c.value * 4)?;
        let eparams = AttentionParams {
            num_heads: heads,
            num_kv_heads: heads,
            head_dim: dk,
            value_dim: c.value,
            stride: estride,
            block_size: eblock,
            scale: 1.0 / (dk as f32).sqrt(),
            softcap: None,
            window: None,
            chunk: None,
            sinks: None,
            alibi_slopes: None,
        };
        let expanded_ms = time(&mut |s| s.attention_paged(&mut eo, &eq, &ekv, &etables, &lengths, 0, &eparams), stream.as_mut())?;
        let latent_mb = (seqs * len) as f64 * crate::mla_latent::latent_bytes_per_token(&crate::config::MlaConfig { q_lora: None, kv_lora: c.kv_lora, nope: c.nope, rope: c.rope, value: c.value, rotary: true }, fmt) as f64 / 1e6;
        let expanded_mb = (seqs * len) as f64
            * crate::mla_latent::expanded_bytes_per_token(heads, &crate::config::MlaConfig { q_lora: None, kv_lora: c.kv_lora, nope: c.nope, rope: c.rope, value: c.value, rotary: true }, fmt) as f64
            / 1e6;
        println!(
            "mla latent bench   {heads} heads: latent {latent_ms:.3} ms over {latent_mb:.0} MB per layer, expanded {expanded_ms:.3} ms over {expanded_mb:.0} MB per layer ({:.1}x less cache, {:.2}x time)",
            expanded_mb / latent_mb,
            latent_ms / expanded_ms
        );
        // timing only: the error columns stay zero, the time column is the latent attention
        Ok(outcome(0.0, 1.0, latent_ms))
    }
}

/// `FLLM_SELFTEST_BENCH=1`: the large timing cases run as well.
fn bench_enabled() -> bool {
    std::env::var("FLLM_SELFTEST_BENCH").is_ok_and(|v| !v.is_empty() && v != "0")
}
