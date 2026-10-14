//! Dry-run tests of the TPU backend: every operation of the `Stream` trait runs on a backend without a device (`TpuBackend::dry_run`),
//! which builds the StableHLO program of the operation and records its text instead of compiling it. The tests check that no
//! operation fails, that every recorded program passes the structure check of `stablehlo::validate`, and (with
//! `FLLM_TPU_DUMP_DIR=<dir>`) write the programs out so that an external MLIR StableHLO parser can verify them as well.

use super::mem::Settings;
use super::weights::WeightFormat;
use super::TpuBackend;
use crate::backend::tpu::stablehlo::ops::Numerics;
use crate::backend::tpu::stablehlo::validate;
use crate::backend::{
    AttentionParams, Backend, BackendError, DeviceBuffer, QuantType, RopeParams, SamplingParams,
    Stream,
};
use std::sync::Arc;

fn backend(numerics: Numerics, weights: WeightFormat) -> TpuBackend {
    TpuBackend::dry_run(Settings { numerics, weights })
}

fn floats(backend: &TpuBackend, stream: &mut dyn Stream, count: usize, value: f32) -> DeviceBuffer {
    let mut buffer = backend.allocate_buffer(count * 4).expect("allocate");
    let bytes: Vec<u8> = (0..count)
        .flat_map(|i| (value + i as f32 * 1e-3).to_le_bytes())
        .collect();
    stream.upload(&mut buffer, &bytes).expect("upload");
    buffer
}

fn blank(backend: &TpuBackend, count: usize) -> DeviceBuffer {
    backend.allocate_buffer(count * 4).expect("allocate")
}

/// F32 weight bytes (`rows x cols` zeros are a valid weight).
fn weight(
    backend: &TpuBackend,
    quant: QuantType,
    rows: usize,
    cols: usize,
    bias: bool,
) -> crate::backend::DeviceWeight {
    let bytes = match quant {
        QuantType::F32 => rows * cols * 4,
        QuantType::Q8_0 => rows * (cols / 32) * 34,
        QuantType::Q4_0 => rows * (cols / 32) * 18,
        other => panic!("test has no sizes for {other:?}"),
    };
    let raw = vec![0u8; bytes];
    let bias_values: Vec<f32> = (0..rows).map(|i| i as f32 * 0.01).collect();
    backend
        .upload_weight(
            "test",
            &raw,
            quant,
            rows,
            cols,
            bias.then_some(bias_values.as_slice()),
        )
        .expect("upload_weight")
}

fn rope_params(
    heads: usize,
    kv_heads: usize,
    dim: usize,
    rope_dim: usize,
    neox: bool,
) -> RopeParams {
    let half = rope_dim / 2;
    let table: Vec<f32> = (0..64 * half).map(|i| (i as f32 * 0.01).cos()).collect();
    RopeParams {
        heads,
        kv_heads,
        dim,
        rope_dim,
        neox,
        cos: Arc::new(table.clone()),
        sin: Arc::new(table),
    }
}

fn all_ops(backend: &TpuBackend) {
    let mut s = backend.create_stream().expect("stream");
    let s = s.as_mut();

    // data movement: whole upload, a window of a larger buffer, downloads, copies, zero fill
    let big = floats(backend, s, 64, 1.0);
    let mut window = big.slice(16 * 4, 8 * 4).expect("slice");
    s.upload(&mut window, &[0u8; 32]).expect("windowed upload");
    let mut bytes = vec![0u8; 32];
    s.download(&mut bytes, &window).expect("windowed download");
    let mut whole = vec![0u8; 64 * 4];
    s.download(&mut whole, &big).expect("download");
    let mut copy = blank(backend, 64);
    s.copy_buffer(&mut copy, &big).expect("copy");
    let mut part = copy.slice(8 * 4, 8 * 4).expect("slice");
    s.copy_buffer(&mut part, &window).expect("windowed copy");
    s.zero_buffer(&mut part).expect("zero");
    let mut never_written = blank(backend, 8);
    s.copy_buffer(&mut never_written, &blank(backend, 8))
        .expect("copy of a never written buffer");
    s.synchronize().expect("synchronize");

    // norms: separate output, in place, a window of a pooled buffer
    let x = floats(backend, s, 3 * 24, 0.5);
    let w = floats(backend, s, 24, 1.0);
    let b = floats(backend, s, 24, 0.0);
    let mut out = blank(backend, 3 * 24);
    s.rms_norm(&mut out, &x, &w, 1e-6).expect("rms_norm");
    s.layer_norm(&mut out, &x, &w, Some(&b), 1e-5)
        .expect("layer_norm");
    s.layer_norm(&mut out, &x, &w, None, 1e-5)
        .expect("layer_norm without bias");
    let mut in_place = x.clone();
    s.rms_norm(&mut in_place, &x, &w, 1e-6)
        .expect("in-place rms_norm");
    let pool = blank(backend, 1024);
    let mut pooled = pool.slice(0, 3 * 24 * 4).expect("slice");
    s.rms_norm(&mut pooled, &x, &w, 1e-6)
        .expect("rms_norm into a pooled window");

    // projections: float and quantised weights, bias, batches, fused residual
    for quant in [QuantType::F32, QuantType::Q8_0, QuantType::Q4_0] {
        let cols = 64;
        let rows = 13;
        let weight = weight(backend, quant, rows, cols, true);
        for batch in [1usize, 5] {
            let x = floats(backend, s, batch * cols, 0.25);
            let mut y = blank(backend, batch * rows);
            s.quantized_linear(&mut y, &x, &weight, weight.bias.as_ref())
                .expect("linear");
            s.quantized_linear(&mut y, &x, &weight, None)
                .expect("linear without bias");
            let mut residual = floats(backend, s, batch * rows, 1.0);
            assert!(s
                .quantized_linear_add(&mut residual, &x, &weight, weight.bias.as_ref(), 0.5)
                .expect("linear_add"));
        }
    }

    // embedding
    let table = weight(backend, QuantType::Q8_0, 40, 64, false);
    let mut embedded = blank(backend, 5 * 64);
    s.embedding_lookup(&mut embedded, &[0, 39, 7, 7, 3], &table)
        .expect("embedding");
    assert!(s.embedding_lookup(&mut embedded, &[40], &table).is_err());

    // rotary embedding: neox and pairs, partial rotation, q and k in separate buffers and in one
    for (heads, kv_heads, dim, rope_dim, neox) in [
        (4usize, 2usize, 16usize, 16usize, true),
        (4, 2, 16, 8, false),
        (2, 2, 8, 8, false),
    ] {
        let params = rope_params(heads, kv_heads, dim, rope_dim, neox);
        let rows = 3;
        let mut q = floats(backend, s, rows * heads * dim, 0.1);
        let mut k = floats(backend, s, rows * kv_heads * dim, 0.2);
        s.rope(&mut q, &mut k, &[0, 5, 9], &params).expect("rope");
    }
    let params = rope_params(2, 2, 8, 8, true);
    let both = floats(backend, s, 2 * 2 * 8 * 2, 0.3);
    let mut q = both.slice(0, 2 * 2 * 8 * 4).expect("slice");
    let mut k = both.slice(2 * 2 * 8 * 4, 2 * 2 * 8 * 4).expect("slice");
    s.rope(&mut q, &mut k, &[1, 2], &params)
        .expect("rope with q and k in one array");

    // activations and residuals
    let g = floats(backend, s, 100, 0.1);
    let u = floats(backend, s, 100, 0.2);
    let mut act = blank(backend, 100);
    s.swiglu(&mut act, &g, &u).expect("swiglu");
    s.geglu(&mut act, &g, &u).expect("geglu");
    s.swiglu_oai(&mut act, &g, &u, 1.702, 7.0)
        .expect("swiglu_oai");
    let mut x = floats(backend, s, 100, 1.0);
    s.residual_add(&mut x, &g, 0.5).expect("residual_add");
    s.residual_add(&mut x, &g, 1.0)
        .expect("residual_add with scale 1");
    s.sigmoid_gate_mul(&mut x, &g).expect("sigmoid_gate_mul");
    assert!(s.supports_attention_gate());
    let projected = floats(backend, s, 3 * 4 * 2 * 8, 0.1);
    for per_head in [true, false] {
        let mut q = blank(backend, 3 * 4 * 8);
        let mut gate = blank(backend, 3 * 4 * 8);
        s.split_q_gate(&mut q, &mut gate, &projected, 3, 4, 8, per_head)
            .expect("split_q_gate");
    }

    // KV cache and attention: f16 cache, plain and with every option of the decoder
    for (dk, dv) in [(16usize, 16usize), (16, 8)] {
        let (layers, pool_blocks, block_size, kv_heads, heads) =
            (3usize, 6usize, 4usize, 2usize, 4usize);
        let cache = backend
            .allocate_kv_cache(layers, pool_blocks, block_size, kv_heads, dk, dv, 1)
            .expect("kv cache");
        let slots = [5usize, 6, 7, 20];
        let k = floats(backend, s, slots.len() * kv_heads * dk, 0.1);
        let v = floats(backend, s, slots.len() * kv_heads * dv, 0.2);
        s.kv_cache_write(&cache, 1, &slots, &k, &v)
            .expect("kv_cache_write");
        let stride = 4usize;
        let lengths = [3i64, 7, 12];
        let mut tables = vec![-1i64; lengths.len() * stride];
        for (r, row) in tables.chunks_mut(stride).enumerate() {
            for (i, entry) in row
                .iter_mut()
                .enumerate()
                .take((lengths[r] as usize).div_ceil(block_size))
            {
                *entry = ((r + i) % pool_blocks) as i64;
            }
        }
        let q = floats(backend, s, lengths.len() * heads * dk, 0.3);
        let mut out = blank(backend, lengths.len() * heads * dv);
        let plain = AttentionParams {
            num_heads: heads,
            num_kv_heads: kv_heads,
            head_dim: dk,
            value_dim: dv,
            stride,
            block_size,
            scale: 1.0 / (dk as f32).sqrt(),
            softcap: None,
            window: None,
            chunk: None,
            sinks: None,
            alibi_slopes: None,
        };
        s.attention_paged(&mut out, &q, &cache, &tables, &lengths, 1, &plain)
            .expect("attention");
        let variants = [
            AttentionParams {
                softcap: Some(30.0),
                ..plain.clone()
            },
            AttentionParams {
                window: Some(5),
                ..plain.clone()
            },
            AttentionParams {
                chunk: Some(4),
                ..plain.clone()
            },
            AttentionParams {
                sinks: Some(Arc::new(vec![0.5; heads])),
                ..plain.clone()
            },
            AttentionParams {
                alibi_slopes: Some(Arc::new(
                    (0..heads).map(|h| 0.5f32.powi(h as i32 + 1)).collect(),
                )),
                ..plain.clone()
            },
            AttentionParams {
                scale: 0.3,
                window: Some(6),
                softcap: Some(20.0),
                sinks: Some(Arc::new(vec![1.0; heads])),
                ..plain.clone()
            },
        ];
        for params in &variants {
            s.attention_paged(&mut out, &q, &cache, &tables, &lengths, 2, params)
                .expect("attention variant");
        }
        // one row, and a window of a larger output buffer
        let big_out = blank(backend, 64 * heads * dv);
        let mut part = big_out.slice(0, heads * dv * 4).expect("slice");
        s.attention_paged(
            &mut part,
            &q.slice(0, heads * dk * 4).expect("slice"),
            &cache,
            &tables[..stride],
            &lengths[..1],
            0,
            &plain,
        )
        .expect("one row");
        // the q8 cache is not available
        assert!(matches!(
            backend.allocate_kv_cache(layers, pool_blocks, block_size, kv_heads, dk, dv, 2),
            Err(BackendError::UnsupportedOperation(_))
        ));
    }

    // sampling and mixture of experts
    let logits = floats(backend, s, 3 * 50, 0.0);
    assert_eq!(s.argmax_rows(&logits, 3, 50).expect("argmax").len(), 3);
    let mut ids = [0u32; 3];
    s.sampling_logits(
        &mut ids,
        &logits,
        &SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            rows: 3,
            vocab_size: 50,
        },
    )
    .expect("sampling");
    let router = floats(backend, s, 4 * 8, 0.0);
    let (mut weights_out, mut indices_out) = (blank(backend, 4 * 2), blank(backend, 4 * 2));
    s.moe_route(&mut weights_out, &mut indices_out, &router, 8, 2)
        .expect("moe_route");
    s.moe_topk(
        &router,
        &mut indices_out,
        &mut weights_out,
        4,
        8,
        2,
        false,
        2.5,
    )
    .expect("moe_topk");
    let y = floats(backend, s, 4 * 2 * 16, 0.1);
    let mut combined = blank(backend, 4 * 16);
    s.moe_combine(&mut combined, &y, &weights_out, 4, 2, 16)
        .expect("moe_combine");
    let src = floats(backend, s, 6 * 16, 0.1);
    let mut gathered = blank(backend, 3 * 16);
    s.gather_rows(&mut gathered, &src, &[5, 0, 2], Some(&[0.5, 1.0, 2.0]), 16)
        .expect("gather_rows");
    s.gather_rows(&mut gathered, &src, &[5, 0, 2], None, 16)
        .expect("gather_rows without weights");
    let mut target = blank(backend, 6 * 16);
    s.zero_buffer(&mut target).expect("zero");
    s.scatter_add_rows(
        &mut target,
        &gathered,
        &[1, 4, 3],
        Some(&[1.0, 0.5, 0.25]),
        16,
    )
    .expect("scatter_add_rows");
    s.scatter_add_rows(&mut target, &gathered, &[1, 4, 3], None, 16)
        .expect("scatter_add_rows without weights");
    assert!(matches!(
        s.moe_expert_gemm(&mut combined, &y, &[], &indices_out, &weights_out, 2),
        Err(BackendError::UnsupportedOperation(_))
    ));
}

fn check_programs(backend: &TpuBackend, label: &str) -> usize {
    let programs = backend.shared().recorded();
    assert!(
        programs.len() > 40,
        "{label}: only {} programs recorded",
        programs.len()
    );
    let mut distinct: Vec<&String> = programs.iter().collect();
    distinct.sort();
    distinct.dedup();
    for (i, text) in distinct.iter().enumerate() {
        if let Err(e) = validate::check(text) {
            panic!("{label}: program {i} fails the structure check: {e}\n{text}");
        }
    }
    if let Some(dir) = std::env::var_os("FLLM_TPU_DUMP_DIR").filter(|d| !d.is_empty()) {
        let dir = std::path::Path::new(&dir).join(label);
        std::fs::create_dir_all(&dir).expect("dump directory");
        for (i, text) in distinct.iter().enumerate() {
            let name = text
                .lines()
                .next()
                .unwrap_or("module")
                .replace(['@', ' ', '{'], "");
            std::fs::write(dir.join(format!("{i:04}_{name}.mlir")), text).expect("write program");
        }
    }
    distinct.len()
}

#[test]
fn every_operation_emits_valid_programs_exact() {
    let b = backend(Numerics::Exact, WeightFormat::F32);
    all_ops(&b);
    assert!(check_programs(&b, "exact_f32") > 30);
}

#[test]
fn every_operation_emits_valid_programs_fast_bf16() {
    let b = backend(Numerics::Fast, WeightFormat::Bf16);
    all_ops(&b);
    assert!(check_programs(&b, "fast_bf16") > 30);
}

#[test]
fn every_operation_emits_valid_programs_fast_int8() {
    let b = backend(Numerics::Fast, WeightFormat::Int8);
    all_ops(&b);
    assert!(check_programs(&b, "fast_int8") > 30);
}

#[test]
fn programs_do_not_depend_on_call_data() {
    // the same operation with other positions and other window offsets must produce the same program text
    let b = backend(Numerics::Fast, WeightFormat::Bf16);
    let mut s = b.create_stream().expect("stream");
    let params = rope_params(2, 2, 8, 8, true);
    let both = floats(&b, s.as_mut(), 2 * 2 * 8 * 4, 0.3);
    let span = 2 * 2 * 8 * 4;
    for (qoff, koff, positions) in [(0usize, 2usize, [1usize, 2]), (1, 3, [7, 9])] {
        let mut q = both.slice(qoff * span, span).expect("slice");
        let mut k = both.slice(koff * span, span).expect("slice");
        s.rope(&mut q, &mut k, &positions, &params).expect("rope");
    }
    // q and k live in one array here, so every call runs two programs (one for q, one for k)
    let programs = b.shared().recorded();
    let rope: Vec<&String> = programs.iter().filter(|p| p.contains("@rope")).collect();
    assert_eq!(rope.len(), 4);
    let distinct: std::collections::HashSet<&&String> = rope.iter().collect();
    assert_eq!(distinct.len(), 1, "the rope programs differ between calls");
}

/// On a TPU: the same checks `fllm-rs selftest-gpu --device tpu` runs. Use
/// `FLLM_TPU_EXACT=1 cargo test --release --features tpu-pjrt -- --ignored tpu_hardware`.
#[test]
#[ignore = "needs a TPU and the libtpu plugin"]
fn tpu_hardware_selftest() {
    let backend = TpuBackend::new(0).expect("a TPU (set FLLM_TPU_LIBRARY to libtpu.so)");
    assert_eq!(
        backend.shared().settings.numerics,
        Numerics::Exact,
        "set FLLM_TPU_EXACT=1"
    );
    crate::backend::selftest::run_backend(&backend).expect("selftest");
}
