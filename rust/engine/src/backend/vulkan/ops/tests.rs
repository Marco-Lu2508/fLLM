//! Hardware-free tests of the dispatch plans: grid and push-constant arithmetic, and the consistency of every plan with the `// FLLM-SHADER:` header and the
//! `push_constant` block of the shader it names (so a reordered field or binding is caught without a GPU).

#![allow(clippy::needless_range_loop)]

use super::attention::*;
use super::moe::*;
use super::recurrent::*;
use super::*;
use crate::backend::{AttentionParams, DeltaShape, QuantType};
use std::path::PathBuf;

fn word(plan: &OpPlan, i: usize) -> u32 {
    u32::from_le_bytes(plan.push_constants[i * 4..i * 4 + 4].try_into().unwrap())
}

fn field(plan: &OpPlan, name: &str) -> u32 {
    let i = plan
        .push_names
        .iter()
        .position(|n| *n == name)
        .unwrap_or_else(|| panic!("no push field {name} in {}", plan.shader_name));
    word(plan, i)
}

fn ffield(plan: &OpPlan, name: &str) -> f32 {
    f32::from_bits(field(plan, name))
}

fn attn_params(
    heads: usize,
    kv_heads: usize,
    dk: usize,
    dv: usize,
    bs: usize,
    stride: usize,
) -> AttentionParams {
    AttentionParams {
        num_heads: heads,
        num_kv_heads: kv_heads,
        head_dim: dk,
        value_dim: dv,
        stride,
        block_size: bs,
        scale: 1.0 / (dk as f32).sqrt(),
        softcap: None,
        window: None,
        chunk: None,
        sinks: None,
        alibi_slopes: None,
    }
}

fn kv_layout(
    kind: i32,
    kv_heads: usize,
    dk: usize,
    dv: usize,
    bs: usize,
    blocks: usize,
) -> KvLayout {
    KvLayout {
        kind,
        kv_heads,
        head_dim: dk,
        value_dim: dv,
        block_size: bs,
        pool_blocks: blocks,
        key_base: 0,
        value_base: 0,
    }
}

/// Block tables of `seqs` sequences, each `len` long, with distinct blocks.
fn tables(lengths: &[i64], bs: usize, stride: usize) -> Vec<i64> {
    let mut next = 0i64;
    let mut out = vec![0i64; lengths.len() * stride];
    for (r, &n) in lengths.iter().enumerate() {
        for b in 0..(n as usize).div_ceil(bs) {
            out[r * stride + b] = next;
            next += 1;
        }
    }
    out
}

#[test]
fn grids() {
    assert_eq!(groups_rows(5), [5, 1, 1]);
    assert_eq!(groups_rows(65535), [65535, 1, 1]);
    assert_eq!(groups_rows(65536), [65535, 2, 1]);
    assert_eq!(groups_rows(200_000), [65535, 4, 1]);
    assert_eq!(groups_1d(1, 256), [1, 1, 1]);
    assert_eq!(groups_1d(256, 256), [1, 1, 1]);
    assert_eq!(groups_1d(257, 256), [2, 1, 1]);
    assert_eq!(groups_1d(1 << 40, 256), [65535, 1, 1]);
    assert_eq!(groups_1d(0, 256), [0, 1, 1]);
}

#[test]
fn packing() {
    assert_eq!(
        pack_u32(&[1, 0x01020304]).unwrap(),
        vec![1, 0, 0, 0, 4, 3, 2, 1]
    );
    assert!(pack_u32(&[1 << 33]).is_err());
    assert_eq!(
        pack_i32(&[-1, 2]).unwrap(),
        vec![255, 255, 255, 255, 2, 0, 0, 0]
    );
    assert!(pack_i32(&[1 << 40]).is_err());
    assert_eq!(pack_f32(&[1.0]), 1.0f32.to_le_bytes().to_vec());
}

#[test]
fn norm_plans() {
    let p = plan_rms_norm(7, 4096, 1e-6, true).unwrap();
    assert_eq!(p.shader_name, "norm_rms");
    assert_eq!(p.groups, [7, 1, 1]);
    assert_eq!(p.push_constants.len(), 32);
    assert_eq!(
        (
            field(&p, "cols"),
            field(&p, "rows"),
            field(&p, "x_stride"),
            field(&p, "o_stride"),
            field(&p, "flags")
        ),
        (4096, 7, 4096, 4096, 1)
    );
    assert_eq!(ffield(&p, "eps"), 1e-6);
    assert_eq!(
        p.bindings.iter().map(|b| b.name).collect::<Vec<_>>(),
        ["x", "weight", "out"]
    );
    let q = plan_rms_norm(70_000, 64, 1e-5, false).unwrap();
    assert_eq!(q.groups, [65535, 2, 1]);
    assert_eq!(field(&q, "flags"), 0);
    assert_eq!(q.slot("weight").unwrap().source, SlotSource::Dummy);
    let l = plan_layer_norm(3, 768, 1e-5, true).unwrap();
    assert_eq!(l.shader_name, "norm_layer");
    assert_eq!(l.slot("bias").unwrap().source, SlotSource::Buffer);
    assert!(plan_rms_norm(0, 8, 1e-6, true).is_err());
    let g = plan_rms_norm_grouped(3, 256, 4, 1e-6, true).unwrap();
    assert_eq!(g.shader_name, "norm_rms_grouped");
    assert_eq!(g.groups, [12, 1, 1]);
    assert_eq!(g.push_constants.len(), 24);
    assert_eq!(
        (
            field(&g, "group_size"),
            field(&g, "groups"),
            field(&g, "total_groups"),
            field(&g, "cols_total"),
            field(&g, "flags"),
        ),
        (64, 4, 12, 256, 1)
    );
    assert_eq!(ffield(&g, "eps"), 1e-6);
    assert!(plan_rms_norm_grouped(0, 64, 4, 1e-6, true).is_err());
    assert!(plan_rms_norm_grouped(2, 64, 0, 1e-6, true).is_err());
    assert!(plan_rms_norm_grouped(2, 64, 3, 1e-6, true).is_err());
}

#[test]
fn rope_plan() {
    let p = plan_rope(&[5, 6, 7], 32, 8, 128, 64, true, 4096).unwrap();
    assert_eq!(p.shader_name, "rope_apply");
    assert_eq!(field(&p, "half_dim"), 32);
    assert_eq!(field(&p, "neox"), 1);
    // 3 rows x (32 + 8) heads x 32 pairs
    assert_eq!(p.groups, groups_1d(3 * 40 * 32, 256));
    match &p.slot("positions").unwrap().source {
        SlotSource::Host(b) => assert_eq!(b.as_slice(), &[5, 0, 0, 0, 6, 0, 0, 0, 7, 0, 0, 0]),
        other => panic!("{other:?}"),
    }
    assert!(plan_rope(&[4096], 32, 8, 128, 64, true, 4096).is_err());
    assert!(plan_rope(&[0], 32, 8, 64, 128, true, 4096).is_err());
    assert!(plan_rope(&[0], 32, 8, 64, 15, true, 4096).is_err());
}

#[test]
fn activation_plans() {
    let p = plan_glu(
        GluMode::SwigluOai {
            alpha: 1.702,
            limit: 7.0,
        },
        1000,
    )
    .unwrap();
    assert_eq!(p.shader_name, "act_swiglu_oai");
    assert_eq!((ffield(&p, "alpha"), ffield(&p, "limit")), (1.702, 7.0));
    assert_eq!(p.groups, [4, 1, 1]);
    assert_eq!(field(&p, "width"), 1000);
    let g = plan_glu(GluMode::Gelu, 8).unwrap();
    assert_eq!(g.slot("up").unwrap().source, SlotSource::Dummy);
    let s = plan_glu_strided(
        GluMode::Swiglu,
        3,
        100,
        GluLayout {
            gate_stride: 200,
            gate_off: 0,
            up_stride: 200,
            up_off: 100,
            out_stride: 100,
        },
    )
    .unwrap();
    assert_eq!((field(&s, "gate_stride"), field(&s, "up_off")), (200, 100));
    assert_eq!(plan_residual_add(5, 0.5).unwrap().groups, [1, 1, 1]);
    assert_eq!(ffield(&plan_residual_add(5, 0.5).unwrap(), "scale"), 0.5);
    assert_eq!(plan_sigmoid_gate_mul(257).unwrap().groups, [2, 1, 1]);
    let q = plan_split_q_gate(2, 16, 128, true).unwrap();
    assert_eq!(
        (field(&q, "total"), field(&q, "per_head")),
        (2 * 16 * 128, 1)
    );
    let a = plan_argmax(3, 151_936).unwrap();
    assert_eq!((a.groups, field(&a, "vocab")), ([3, 1, 1], 151_936));
}

#[test]
fn embedding_plans() {
    let table = EmbeddingTable {
        quant: QuantType::Q4_K,
        rows: 1000,
        cols: 512,
        planes: [10, 20, 30],
    };
    let p = plan_embedding(&table, &[3, 999, 0]).unwrap();
    assert_eq!(p.shader_name, "embed_q4_k");
    assert_eq!(
        (
            field(&p, "n_ids"),
            field(&p, "cols"),
            field(&p, "nb"),
            field(&p, "off1"),
            field(&p, "off3")
        ),
        (3, 512, 2, 10, 30)
    );
    assert_eq!(p.groups, groups_1d(3 * 64, 256));
    assert!(plan_embedding(&table, &[1000]).is_err());
    assert!(plan_embedding(&table, &[-1]).is_err());
    assert!(plan_embedding(&EmbeddingTable { cols: 500, ..table }, &[1]).is_err());
    let f = plan_embedding(
        &EmbeddingTable {
            quant: QuantType::F16,
            rows: 10,
            cols: 33,
            planes: [0; 3],
        },
        &[1],
    )
    .unwrap();
    assert_eq!(
        (f.shader_name, field(&f, "nb"), f.groups),
        ("embed_f16", 0, groups_1d(33, 256))
    );
    assert!(matches!(
        plan_embedding(
            &EmbeddingTable {
                quant: QuantType::IQ2_XS,
                ..table
            },
            &[1]
        ),
        Err(BackendError::UnsupportedQuant(_))
    ));
}

#[test]
fn kv_write_plans() {
    let kv = kv_layout(1, 8, 128, 128, 16, 64);
    let plans = plan_kv_write(&kv, &[3, 4, 100]).unwrap();
    assert_eq!(plans.len(), 2);
    assert_eq!(plans[0].shader_name, "kv_write_f16");
    assert_eq!(
        (
            field(&plans[0], "rows"),
            field(&plans[0], "kv"),
            field(&plans[0], "base")
        ),
        (3, 1024, 0)
    );
    assert_eq!(plans[0].groups, groups_1d(3 * 513, 256));
    assert_eq!(plans[0].slot("k").unwrap().name, "k");
    assert_eq!(
        plans[1].slot("value_pool").unwrap().access,
        Access::ReadWrite
    );
    // q8: group 32 for 128, the whole head for 80; scale base = ceil(slots * kv / 2)
    let q8 = KvLayout {
        kind: 2,
        ..kv_layout(1, 4, 128, 80, 16, 64)
    };
    let plans = plan_kv_write(&q8, &[0, 1]).unwrap();
    assert_eq!(plans[0].shader_name, "kv_write_q8");
    assert_eq!(
        (field(&plans[0], "group"), field(&plans[0], "scale_base")),
        (32, (64 * 16 * 4 * 128) / 2)
    );
    assert_eq!(
        (
            field(&plans[1], "group"),
            field(&plans[1], "kv"),
            field(&plans[1], "scale_base")
        ),
        (80, 320, (64 * 16 * 320) / 2)
    );
    assert_eq!(plans[1].groups, groups_1d(2 * 4, 128));
    assert!(plan_kv_write(&q8, &[64 * 16]).is_err());
    assert!(plan_kv_write(&q8, &[]).is_err());
    // a layer beyond 4 GiB of window must be rebased
    let big = KvLayout {
        key_base: 1 << 31,
        ..kv
    };
    assert!(plan_kv_write(&big, &[0]).is_err());
}

#[test]
fn attention_decode_splits() {
    let (heads, kvh, d, bs) = (32, 8, 128, 16);
    let kv = kv_layout(1, kvh, d, d, bs, 1024);
    let params = attn_params(heads, kvh, d, d, bs, 256);
    // one long row: split to fill the device
    let lengths = [4096i64];
    let plans = plan_attention(
        &kv,
        &params,
        &tables(&lengths, bs, 256),
        &lengths,
        &AttentionTuning::default(),
    )
    .unwrap();
    assert_eq!(plans.len(), 2);
    assert_eq!(plans[0].shader_name, "attn_decode_f16");
    // 512 / 32 heads = 16 splits (4096 / 256 = 16 keys-wise)
    assert_eq!(plans[0].groups, [16, 32, 1]);
    assert_eq!(field(&plans[0], "nsplit"), 16);
    assert_eq!(plans[1].shader_name, "attn_merge");
    assert_eq!(plans[1].groups, [32, 1, 1]);
    assert_eq!(field(&plans[1], "nsplit"), 16);
    assert_eq!(
        plans[0].slot("part").unwrap().source,
        SlotSource::Scratch {
            id: 0,
            bytes: 32 * 16 * (2 + d) * 4
        }
    );
    // a short row stays in one piece and needs no merge
    let short = [100i64];
    let plans = plan_attention(
        &kv,
        &params,
        &tables(&short, bs, 256),
        &short,
        &AttentionTuning::default(),
    )
    .unwrap();
    assert_eq!(plans.len(), 1);
    assert_eq!(plans[0].groups, [1, 32, 1]);
    assert_eq!(plans[0].slot("part").unwrap().source, SlotSource::Dummy);
    // many rows already fill the device
    let lengths = [900i64, 17, 3000, 5, 64, 800, 1200, 33, 2, 1000];
    let plans = plan_attention(
        &kv,
        &params,
        &tables(&lengths, bs, 256),
        &lengths,
        &AttentionTuning {
            use_prefill_kernel: false,
            ..Default::default()
        },
    )
    .unwrap();
    // 10 rows x 32 heads = 320 workgroups: two splits each (512 / 320), the longest row has 3000 keys
    assert_eq!(plans.len(), 2);
    assert_eq!(plans[0].groups, [2, 32, 10]);
    assert_eq!(plans[1].groups, [32, 10, 1]);
}

#[test]
fn attention_push_fields() {
    let (heads, kvh, dk, dv, bs) = (16, 4, 192, 128, 32);
    let mut params = attn_params(heads, kvh, dk, dv, bs, 64);
    params.window = Some(128);
    params.softcap = Some(30.0);
    params.sinks = Some(Arc::new(vec![0.5; heads]));
    let kv = KvLayout {
        key_base: 7,
        value_base: 9,
        ..kv_layout(2, kvh, dk, dv, bs, 512)
    };
    let lengths = [500i64];
    let plans = plan_attention(
        &kv,
        &params,
        &tables(&lengths, bs, 64),
        &lengths,
        &AttentionTuning::default(),
    )
    .unwrap();
    let p = &plans[0];
    assert_eq!(p.shader_name, "attn_decode_q8");
    assert_eq!(
        (field(p, "window"), field(p, "chunk"), field(p, "flags")),
        (128, 0, 1)
    );
    assert_eq!(ffield(p, "softcap"), 30.0);
    assert_eq!((field(p, "k_base"), field(p, "v_base")), (7, 9));
    assert_eq!((field(p, "k_group"), field(p, "v_group")), (32, 32));
    assert_eq!(field(p, "k_scale") as usize, 512 * 32 * 4 * 192 / 2);
    assert_eq!(field(p, "v_scale") as usize, 512 * 32 * 4 * 128 / 2);
    match &p.slot("bias").unwrap().source {
        SlotSource::Host(b) => {
            assert_eq!(b.len(), 2 * heads * 4);
            assert_eq!(f32::from_le_bytes(b[0..4].try_into().unwrap()), 0.5);
            assert_eq!(
                f32::from_le_bytes(b[heads * 4..heads * 4 + 4].try_into().unwrap()),
                0.0
            );
        }
        other => panic!("{other:?}"),
    }
    match &p.slot("meta").unwrap().source {
        SlotSource::Host(b) => assert_eq!(b.len(), (1 + 64) * 4),
        other => panic!("{other:?}"),
    }
}

#[test]
fn attention_prefill_segments() {
    let (heads, kvh, d, bs) = (8, 2, 64, 16);
    let kv = kv_layout(1, kvh, d, d, bs, 512);
    let params = attn_params(heads, kvh, d, d, bs, 64);
    // 40 rows of one sequence (lengths 11 .. 50, one shared table), then a decode row of another sequence
    let mut lengths: Vec<i64> = (11..51).collect();
    lengths.push(300);
    let mut t = vec![0i64; lengths.len() * 64];
    for r in 0..40 {
        for b in 0..64 {
            t[r * 64 + b] = b as i64; // same sequence: blocks 0, 1, 2, ...
        }
    }
    for b in 0..19 {
        t[40 * 64 + b] = 100 + b as i64;
    }
    let plans = plan_attention(&kv, &params, &t, &lengths, &AttentionTuning::default()).unwrap();
    let names: Vec<_> = plans.iter().map(|p| p.shader_name).collect();
    assert_eq!(names[0], "attn_prefill_f16_d128");
    assert_eq!(plans[0].groups, [3, 8, 1]); // rows 0..40 in blocks of 16
    assert_eq!(
        (
            field(&plans[0], "row0"),
            field(&plans[0], "rows_end"),
            field(&plans[0], "rows_total")
        ),
        (0, 40, 41)
    );
    assert_eq!(names[1], "attn_decode_f16");
    assert_eq!(field(&plans[1], "row0"), 40);
    // the decode row (300 keys) is split over the key range: decode kernel plus merge
    assert_eq!(names[2], "attn_merge");
    assert_eq!(names.len(), 3, "{names:?}");
    // rows of different sequences are never tiled together
    let lens = [20i64, 21, 22, 23, 24, 25];
    let mut tb = vec![0i64; 6 * 64];
    for r in 0..6 {
        tb[r * 64] = r as i64 * 2;
        tb[r * 64 + 1] = r as i64 * 2 + 1;
    }
    let plans = plan_attention(&kv, &params, &tb, &lens, &AttentionTuning::default()).unwrap();
    assert!(plans
        .iter()
        .all(|p| p.shader_name.starts_with("attn_decode") || p.shader_name == "attn_merge"));
    // wide values pick the wider accumulator variant
    let kv2 = kv_layout(2, kvh, 192, 256, bs, 512);
    let p2 = attn_params(heads, kvh, 192, 256, bs, 64);
    let lens = [30i64, 31, 32, 33];
    let mut tb = vec![0i64; 4 * 64];
    for r in 0..4 {
        tb[r * 64] = 0;
        tb[r * 64 + 1] = 1;
        tb[r * 64 + 2] = 2;
    }
    let plans = plan_attention(&kv2, &p2, &tb, &lens, &AttentionTuning::default()).unwrap();
    assert_eq!(plans[0].shader_name, "attn_prefill_q8_d256");
    assert_eq!(plans[0].groups, [1, 8, 1]);
}

#[test]
fn attention_rejects_bad_input() {
    let (heads, kvh, d, bs) = (8, 2, 64, 16);
    let kv = kv_layout(1, kvh, d, d, bs, 8);
    let params = attn_params(heads, kvh, d, d, bs, 4);
    let ok = tables(&[20], bs, 4);
    let t = AttentionTuning::default();
    assert!(plan_attention(&kv, &params, &ok, &[20], &t).is_ok());
    assert!(plan_attention(&kv, &params, &ok, &[0], &t).is_err());
    assert!(
        plan_attention(&kv, &params, &ok, &[65], &t).is_err(),
        "65 keys need 5 blocks, the stride is 4"
    );
    assert!(
        plan_attention(&kv, &params, &[0, 1, 99, 0], &[40], &t).is_err(),
        "block outside the pool"
    );
    assert!(
        plan_attention(&kv, &params, &[0, 1], &[20], &t).is_err(),
        "table size"
    );
    assert!(plan_attention(&kv, &attn_params(7, kvh, d, d, bs, 4), &ok, &[20], &t).is_err());
    assert!(
        plan_attention(&kv, &attn_params(heads, kvh, d, 32, bs, 4), &ok, &[20], &t).is_err(),
        "cache mismatch"
    );
    let mut zero = attn_params(heads, kvh, d, d, bs, 4);
    zero.window = Some(0);
    assert!(plan_attention(&kv, &zero, &ok, &[20], &t).is_err());
    let mut cap = attn_params(heads, kvh, d, d, bs, 4);
    cap.softcap = Some(0.0);
    assert!(plan_attention(&kv, &cap, &ok, &[20], &t).is_err());
    let mut short = attn_params(heads, kvh, d, d, bs, 4);
    short.alibi_slopes = Some(Arc::new(vec![0.1; 3]));
    assert!(plan_attention(&kv, &short, &ok, &[20], &t).is_err());
    let wide = kv_layout(1, kvh, 640, 640, bs, 8);
    assert!(matches!(
        plan_attention(
            &wide,
            &attn_params(heads, kvh, 640, 640, bs, 4),
            &ok,
            &[20],
            &t
        ),
        Err(BackendError::UnsupportedOperation(_))
    ));
}

#[test]
fn moe_plans() {
    let r = plan_moe_topk(5, 64, 8, true, 0.5).unwrap();
    assert_eq!(
        (
            r.groups,
            field(&r, "k"),
            field(&r, "normalize_scores"),
            ffield(&r, "scale")
        ),
        ([5, 1, 1], 8, 1, 0.5)
    );
    assert!(plan_moe_topk(1, 2048, 8, true, 1.0).is_err());
    assert!(plan_moe_topk(1, 4, 8, true, 1.0).is_err());
    let route = plan_moe_route(3, 128, 8).unwrap();
    assert_eq!(
        (route.bindings[1].name, route.bindings[2].name),
        ("indices", "weights")
    );
    let lin = MoeLinear {
        quant: QuantType::Q4_K,
        rows: 768,
        cols: 2048,
        slots: 70_000,
        x_div: 8,
        n_experts: 128,
        has_bias: true,
        access: ExpertAccess::Arena,
        planes: [1, 2, 3],
    };
    let p = plan_moe_linear(&lin).unwrap();
    assert_eq!(p.shader_name, "moe_linear_q4_k_arena");
    assert_eq!(p.groups, [768, 65535, 2]);
    assert_eq!(
        (
            field(&p, "nb"),
            field(&p, "x_div"),
            field(&p, "flags"),
            field(&p, "off2")
        ),
        (8, 8, 1, 2)
    );
    assert_eq!(p.bindings.len(), 7);
    let b = plan_moe_linear(&MoeLinear {
        access: ExpertAccess::BufferAddress,
        quant: QuantType::Mxfp4,
        has_bias: false,
        slots: 6,
        ..lin
    })
    .unwrap();
    assert_eq!(b.shader_name, "moe_linear_mxfp4_bda");
    assert_eq!(b.groups, [768, 6, 1]);
    assert_eq!(b.slot("weights").unwrap().source, SlotSource::Dummy);
    assert_eq!(b.slot("bias_table").unwrap().source, SlotSource::Dummy);
    assert!(plan_moe_linear(&MoeLinear { cols: 2000, ..lin }).is_err());
    assert!(plan_moe_linear(&MoeLinear {
        quant: QuantType::F16,
        ..lin
    })
    .is_err());
    assert!(moe_linear_supported(QuantType::Q6_K) && !moe_linear_supported(QuantType::F32));
    let gemm = plan_moe_expert_gemm(
        &MoeLinear {
            slots: 1,
            x_div: 1,
            ..lin
        },
        5,
        4,
    )
    .unwrap();
    assert_eq!(gemm.len(), 2);
    assert_eq!(
        (
            field(&gemm[0], "slots"),
            field(&gemm[0], "x_div"),
            field(&gemm[0], "flags")
        ),
        (20, 4, 0)
    );
    assert_eq!(gemm[1].shader_name, "moe_combine");
    assert_eq!(
        (
            field(&gemm[1], "tokens"),
            field(&gemm[1], "k"),
            field(&gemm[1], "hidden")
        ),
        (5, 4, 768)
    );
    assert_eq!(
        gemm[0].slot("y").unwrap().source,
        gemm[1].slot("y").unwrap().source
    );
    let c = plan_moe_combine(4, 8, 2048).unwrap();
    assert_eq!(c.groups, groups_1d(4 * 2048, 256));
}

#[test]
fn gather_scatter_plans() {
    let g = plan_gather_rows(&[5, 36, 0], Some(&[0.5, 1.0, 2.0]), 130).unwrap();
    assert_eq!(
        (g.shader_name, field(&g, "n"), field(&g, "has_weights")),
        ("moe_gather_rows", 3, 1)
    );
    assert_eq!(
        g.bindings.iter().map(|b| b.name).collect::<Vec<_>>(),
        ["dst", "src", "idx", "weights"]
    );
    assert_eq!(g.groups, groups_1d(3 * 130, 256));
    let s = plan_scatter_add_rows(&[1, 2], None, 4).unwrap();
    assert_eq!(
        (s.slot("dst").unwrap().access, field(&s, "has_weights")),
        (Access::ReadWrite, 0)
    );
    assert_eq!(s.slot("weights").unwrap().source, SlotSource::Dummy);
    assert!(plan_gather_rows(&[-1], None, 4).is_err());
    assert!(plan_gather_rows(&[1, 2], Some(&[1.0]), 4).is_err());
}

fn delta_shape(kh: usize, vh: usize, hk: usize, hv: usize, ksz: usize) -> DeltaShape {
    DeltaShape {
        k_heads: kh,
        v_heads: vh,
        k_dim: hk,
        v_dim: hv,
        conv_kernel: ksz,
        tiled_heads: true,
        beta_scale: 1.0,
        eps: 1e-6,
    }
}

#[test]
fn delta_plans() {
    let shape = delta_shape(4, 8, 128, 96, 4);
    assert!(delta_supported(&shape));
    assert!(!delta_supported(&delta_shape(4, 8, 100, 96, 4)));
    assert!(!delta_supported(&delta_shape(4, 6, 128, 96, 4)));
    assert!(!delta_supported(&delta_shape(4, 8, 128, 192, 4)));
    assert!(!delta_supported(&delta_shape(4, 8, 128, 96, 9)));
    let runs = [
        RunSpec {
            first_row: 0,
            rows: 40,
            reset: true,
        },
        RunSpec {
            first_row: 40,
            rows: 1,
            reset: false,
        },
        RunSpec {
            first_row: 41,
            rows: 0,
            reset: false,
        },
    ];
    let offset = shape.layer_state_bytes();
    let plans = plan_delta_layer(&shape, &runs, offset).unwrap();
    assert_eq!(plans.len(), 4);
    let conv_dim = shape.conv_dim();
    let c = &plans[0];
    assert_eq!(c.shader_name, "delta_conv");
    // 4 + 4 head blocks, 768 value channels in blocks of 128; 40 tokens in chunks of 16
    assert_eq!(c.groups, [8 + 6, 3, 1]);
    assert_eq!(
        (
            field(c, "conv_dim") as usize,
            field(c, "hk"),
            field(c, "reset")
        ),
        (conv_dim, 128, 1)
    );
    assert_eq!(
        field(c, "hist_off") as usize,
        (offset + shape.rec_bytes()) / 4
    );
    assert!((ffield(c, "qscale") - 128f32.powf(-0.5)).abs() < 1e-9);
    assert_eq!(c.slot("state").unwrap().source, SlotSource::RunState(0));
    let r = &plans[1];
    assert_eq!(r.shader_name, "delta_rec_hk128");
    assert_eq!(r.groups, [8, 1, 1]);
    assert_eq!(
        (
            field(r, "state_off") as usize,
            field(r, "tiled"),
            field(r, "hv")
        ),
        (offset / 4, 1, 96)
    );
    assert_eq!(
        plans[2].slot("state").unwrap().source,
        SlotSource::RunState(1)
    );
    assert_eq!(plans[3].groups, [8, 1, 1]);
    assert!(plan_delta_layer(&shape, &runs, 3).is_err());
    assert!(plan_delta_layer(&delta_shape(4, 8, 100, 96, 4), &runs, 0).is_err());
}

fn mamba2() -> Mamba2Shape {
    Mamba2Shape {
        inner: 512,
        state: 128,
        heads: 8,
        groups: 2,
        conv_kernel: 4,
        norm_groups: 2,
        step_min: 0.001,
        eps: 1e-5,
    }
}

#[test]
fn mamba_plans() {
    let s = mamba2();
    assert_eq!(
        (s.conv_dim(), s.head_dim(), s.proj_width()),
        (512 + 2 * 2 * 128, 64, 512 + 1024 + 8)
    );
    let (hist, rec) = s.state_floats();
    assert_eq!((hist, rec), (1024 * 3, 8 * 64 * 128));
    let runs = [
        RunSpec {
            first_row: 0,
            rows: 20,
            reset: true,
        },
        RunSpec {
            first_row: 20,
            rows: 2,
            reset: false,
        },
    ];
    let plans = plan_mamba2_mixer(&s, &runs, 22, 256).unwrap();
    assert_eq!(
        plans.iter().map(|p| p.shader_name).collect::<Vec<_>>(),
        [
            "mamba_conv",
            "mamba_scan2_ns128",
            "mamba_conv",
            "mamba_scan2_ns128",
            "mamba_gate_norm"
        ]
    );
    let conv = &plans[0];
    assert_eq!(conv.groups, [div_up(1024, 64) as u32, 2, 1]);
    assert_eq!(
        (
            field(conv, "in_stride") as usize,
            field(conv, "in_off"),
            field(conv, "hist_off")
        ),
        (s.proj_width(), 512, 64)
    );
    let scan = &plans[1];
    assert_eq!(scan.groups, [8, 1, 1]);
    assert_eq!(
        (
            field(scan, "state_off") as usize,
            field(scan, "proj_dt_off") as usize
        ),
        (64 + hist, 512 + 1024)
    );
    assert_eq!(ffield(scan, "step_min"), 0.001);
    assert_eq!(plans[4].groups, [2, 22, 1]);
    assert_eq!(
        conv.slot("mixed").unwrap().source,
        SlotSource::Scratch {
            id: 1,
            bytes: 22 * 1024 * 4
        }
    );
    assert_eq!(s.layer_state_bytes() % 256, 0);
    assert_eq!(
        plan_mamba2_scan(0, &runs[0], &Mamba2Shape { state: 48, ..s }, 1, 0, 0, None)
            .unwrap()
            .shader_name,
        "mamba_scan2_gen"
    );
    assert!(plan_mamba2_mixer(&Mamba2Shape { heads: 7, ..s }, &runs, 22, 0).is_err());
    assert!(plan_mamba2_mixer(&Mamba2Shape { state: 300, ..s }, &runs, 22, 0).is_err());

    let m1 = Mamba1Shape {
        inner: 256,
        state: 16,
        rank: 8,
        conv_kernel: 4,
        bc_rms_eps: Some(1e-5),
    };
    let scan1 = plan_mamba1_scan(0, &runs[0], &m1, 512, 256, 100).unwrap();
    assert_eq!(scan1.shader_name, "mamba_scan1_ns16");
    assert_eq!(
        (scan1.groups, field(&scan1, "split_stride")),
        ([4, 1, 1], 8 + 32)
    );
    let norms = plan_mamba1_bc_norms(&m1, 9).unwrap();
    assert_eq!(norms.len(), 3);
    assert_eq!(
        (
            field(&norms[1], "x_off"),
            field(&norms[1], "cols"),
            field(&norms[2], "x_off")
        ),
        (8, 16, 24)
    );
    assert!(plan_mamba1_bc_norms(
        &Mamba1Shape {
            bc_rms_eps: None,
            ..m1
        },
        9
    )
    .unwrap()
    .is_empty());
}

// ------------------------------------------------------------------------------------------------------------------------------------------
// Consistency of the plans with the shader sources
// ------------------------------------------------------------------------------------------------------------------------------------------

fn shader_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("shaders")
        .join("vulkan")
}

/// (output name, source file, defines) of `variants_ops.list`.
fn variants() -> Vec<(String, String, Vec<String>)> {
    let text =
        std::fs::read_to_string(shader_dir().join("variants_ops.list")).expect("variants_ops.list");
    text.lines()
        .filter_map(|l| {
            let l = l.split('#').next().unwrap_or("").trim();
            let mut parts = l.split_whitespace();
            let (out, src) = (parts.next()?, parts.next()?);
            Some((
                out.to_string(),
                src.to_string(),
                parts.map(str::to_string).collect(),
            ))
        })
        .collect()
}

/// Source file of the shader called `name`.
fn source_file(name: &str) -> String {
    if let Some((_, src, _)) = variants().into_iter().find(|(out, _, _)| out == name) {
        return src;
    }
    let file = format!("{name}.comp");
    assert!(shader_dir().join(&file).exists(), "no shader named {name}");
    file
}

/// (binding names, push names) of the `// FLLM-SHADER:` header.
fn header(file: &str) -> (Vec<String>, Vec<String>) {
    let text = std::fs::read_to_string(shader_dir().join(file)).unwrap();
    let line = text
        .lines()
        .find(|l| l.starts_with("// FLLM-SHADER:"))
        .unwrap_or_else(|| panic!("{file}: no FLLM-SHADER header"));
    let part = |key: &str| -> Vec<String> {
        let at = line
            .find(key)
            .unwrap_or_else(|| panic!("{file}: header without {key}"))
            + key.len();
        let rest = &line[at..];
        let rest = rest.split('|').next().unwrap();
        rest.split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    };
    (part("bindings:"), part("push:"))
}

/// Member names of the `layout(push_constant)` block.
fn push_block(file: &str) -> Vec<String> {
    let text = std::fs::read_to_string(shader_dir().join(file)).unwrap();
    let at = text
        .find("layout(push_constant) uniform P {")
        .unwrap_or_else(|| panic!("{file}: no push block"));
    let rest = &text[at..];
    let body = &rest[rest.find('{').unwrap() + 1..rest.find('}').unwrap()];
    body.split(';')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.split_whitespace().last().unwrap().to_string())
        .collect()
}

/// A plan of every kind and variant.
fn sample_plans() -> Vec<OpPlan> {
    let mut v = vec![
        plan_rms_norm(2, 64, 1e-6, true).unwrap(),
        plan_rms_norm(2, 64, 1e-6, false).unwrap(),
        plan_rms_norm_grouped(2, 64, 4, 1e-6, true).unwrap(),
        plan_rms_norm_grouped(2, 64, 4, 1e-6, false).unwrap(),
        plan_layer_norm(2, 64, 1e-6, true).unwrap(),
        plan_rope(&[0, 1], 4, 2, 64, 32, false, 8).unwrap(),
        plan_residual_add(10, 1.0).unwrap(),
        plan_sigmoid_gate_mul(10).unwrap(),
        plan_split_q_gate(2, 4, 64, false).unwrap(),
        plan_argmax(2, 100).unwrap(),
        plan_moe_topk(2, 64, 4, false, 1.0).unwrap(),
        plan_moe_route(2, 64, 4).unwrap(),
        plan_moe_combine(2, 4, 64).unwrap(),
        plan_gather_rows(&[0, 1], None, 8).unwrap(),
        plan_scatter_add_rows(&[0, 1], Some(&[1.0, 2.0]), 8).unwrap(),
    ];
    for mode in [
        GluMode::Swiglu,
        GluMode::Geglu,
        GluMode::SwigluOai {
            alpha: 1.0,
            limit: 2.0,
        },
        GluMode::Silu,
        GluMode::Gelu,
        GluMode::Relu2,
        GluMode::Relu,
        GluMode::GeluErf,
        GluMode::GeluErfGated,
        GluMode::ReluGated,
        GluMode::Relu2Gated,
        GluMode::Copy,
    ] {
        v.push(plan_glu(mode, 64).unwrap());
    }
    for softplus in [false, true] {
        v.push(super::plan_ewise(softplus, 64, 0.5).unwrap());
    }
    v.push(super::plan_scale(64, 0.5).unwrap());
    {
        use crate::backend::vulkan::ops::hc::{plan_hc_combine, plan_hc_mix, plan_rotate_heads};
        v.push(plan_rotate_heads(64, 16, 4).unwrap());
        use crate::backend::HcMode;
        for mode in [HcMode::Independent { magnitude: 2.0 }, HcMode::Sinkhorn { iters: 4 }, HcMode::Head, HcMode::Mean] {
            v.push(plan_hc_mix(3, 4, 64, mode, 1e-6).unwrap());
        }
        for has_comb in [false, true] {
            v.push(plan_hc_combine(3, 4, 64, has_comb).unwrap());
        }
    }
    {
        use crate::backend::vulkan::ops::dsa::{plan_dsa_attention, plan_dsa_select};
        let p = crate::backend::DsaParams { heads: 4, dim: 32, topk: 8, stride: 128, block_size: 16, scale: 0.25, head_scale: 0.5 };
        let table: Vec<i64> = (0..128i64).collect();
        for lengths in [vec![300i64, 5, 1000], vec![3i64, 5]] {
            let tables: Vec<i64> = lengths.iter().flat_map(|_| table.iter().copied()).collect();
            let slots: Vec<usize> = (0..lengths.len()).collect();
            v.extend(plan_dsa_select(&p, &slots, &tables, &lengths, 128 * 16).unwrap());
        }
        let tables: Vec<i64> = table.iter().chain(table.iter()).copied().collect();
        v.push(plan_dsa_attention(&kv_layout(1, 4, 192, 128, 16, 128), &attn_params(4, 4, 192, 128, 16, 128), 8, &tables).unwrap());
    }
    for beta in [None, Some(0.5)] {
        v.push(super::plan_head_gate(64, 8, beta).unwrap());
    }
    for part in [super::MlaPart::Q, super::MlaPart::K, super::MlaPart::V, super::MlaPart::Latent] {
        v.push(super::plan_mla_move(part, 3, 4, 16, 8, 16, 32).unwrap());
    }
    for quant in [
        QuantType::F32,
        QuantType::F16,
        QuantType::BF16,
        QuantType::Q4_0,
        QuantType::Q4_1,
        QuantType::Q5_0,
        QuantType::Q5_1,
        QuantType::Q8_0,
        QuantType::Q2_K,
        QuantType::Q3_K,
        QuantType::Q4_K,
        QuantType::Q5_K,
        QuantType::Q6_K,
        QuantType::IQ4_NL,
        QuantType::Mxfp4,
    ] {
        v.push(
            plan_embedding(
                &EmbeddingTable {
                    quant,
                    rows: 4,
                    cols: 512,
                    planes: [1, 2, 3],
                },
                &[0, 3],
            )
            .unwrap(),
        );
        if moe_linear_supported(quant) {
            for access in [ExpertAccess::Arena, ExpertAccess::BufferAddress] {
                for has_bias in [false, true] {
                    v.push(
                        plan_moe_linear(&MoeLinear {
                            quant,
                            rows: 8,
                            cols: 512,
                            slots: 3,
                            x_div: 1,
                            n_experts: 4,
                            has_bias,
                            access,
                            planes: [1, 2, 3],
                        })
                        .unwrap(),
                    );
                }
            }
        }
    }
    for kind in [1, 2] {
        let kv = kv_layout(kind, 2, 64, 64, 16, 32);
        v.extend(plan_kv_write(&kv, &[0, 5]).unwrap());
        for dv in [64usize, 160, 300] {
            let kv = kv_layout(kind, 2, 64, dv, 16, 256);
            let mut params = attn_params(8, 2, 64, dv, 16, 128);
            params.alibi_slopes = Some(Arc::new(vec![0.1; 8]));
            // a long row (split + merge), then a tiled run
            let mut lengths = vec![2000i64];
            lengths.extend(20..40);
            let mut t = vec![0i64; lengths.len() * 128];
            for r in 1..lengths.len() {
                for b in 0..128 {
                    t[r * 128 + b] = b as i64;
                }
            }
            for b in 0..125 {
                t[b] = 128 + b as i64;
            }
            v.extend(
                plan_attention(&kv, &params, &t, &lengths, &AttentionTuning::default()).unwrap(),
            );
            let wide = AttentionTuning { prefill_rows: PREFILL_ROWS_WIDE, ..AttentionTuning::default() };
            v.extend(plan_attention(&kv, &params, &t, &lengths, &wide).unwrap());
            // the cooperative-matrix prompt kernel (f16 cache, key width = value width 64)
            let coop = AttentionTuning { coop_prefill: true, ..wide };
            v.extend(plan_attention(&kv, &params, &t, &lengths, &coop).unwrap());
        }
        if kind == 1 {
            // ... and the same with width 128
            let kv = kv_layout(kind, 2, 128, 128, 16, 256);
            let params = attn_params(8, 2, 128, 128, 16, 128);
            let mut lengths = vec![2000i64];
            lengths.extend(20..40);
            let mut t = vec![0i64; lengths.len() * 128];
            for r in 1..lengths.len() {
                for b in 0..128 {
                    t[r * 128 + b] = b as i64;
                }
            }
            let coop = AttentionTuning { prefill_rows: PREFILL_ROWS_WIDE, coop_prefill: true, ..AttentionTuning::default() };
            v.extend(plan_attention(&kv, &params, &t, &lengths, &coop).unwrap());
        }
    }
    let shape = delta_shape(2, 4, 64, 64, 4);
    v.extend(
        plan_delta_layer(
            &shape,
            &[RunSpec {
                first_row: 0,
                rows: 3,
                reset: true,
            }],
            0,
        )
        .unwrap(),
    );
    for hk in [32, 64, 96, 128] {
        v.extend(
            plan_delta_layer_with(
                &delta_shape(2, 4, hk, 40, 4),
                &[RunSpec {
                    first_row: 0,
                    rows: 3,
                    reset: true,
                }],
                0,
                true,
            )
            .unwrap(),
        );
    }
    for hk in [32, 96, 128] {
        v.extend(
            plan_delta_layer(
                &delta_shape(2, 4, hk, 32, 4),
                &[RunSpec {
                    first_row: 0,
                    rows: 3,
                    reset: true,
                }],
                0,
            )
            .unwrap(),
        );
    }
    for state in [16, 32, 64, 128, 48] {
        let s = Mamba2Shape { state, ..mamba2() };
        v.extend(
            plan_mamba2_mixer(
                &s,
                &[RunSpec {
                    first_row: 0,
                    rows: 3,
                    reset: true,
                }],
                3,
                0,
            )
            .unwrap(),
        );
    }
    for state in [8, 16, 32, 24] {
        let m1 = Mamba1Shape {
            inner: 64,
            state,
            rank: 4,
            conv_kernel: 4,
            bc_rms_eps: Some(1e-5),
        };
        v.push(
            plan_mamba1_scan(
                0,
                &RunSpec {
                    first_row: 0,
                    rows: 2,
                    reset: false,
                },
                &m1,
                128,
                64,
                0,
            )
            .unwrap(),
        );
        v.extend(plan_mamba1_bc_norms(&m1, 2).unwrap());
    }
    v.push(
        plan_mamba_conv(
            0,
            &RunSpec {
                first_row: 0,
                rows: 2,
                reset: false,
            },
            64,
            4,
            128,
            0,
            0,
            false,
            None,
        )
        .unwrap(),
    );
    {
        use super::recurrent::{plan_rwkv6_wkv, plan_rwkv7_wkv, plan_rwkv_act, plan_rwkv_lerp, plan_rwkv_shift};
        let run = RunSpec { first_row: 2, rows: 3, reset: false };
        v.push(plan_rwkv_shift(1, &run, 128, 4096).unwrap());
        for has_delta in [false, true] {
            v.push(plan_rwkv_lerp(256, 128, has_delta).unwrap());
        }
        for (mode, gate) in [(0, false), (2, false), (4, true), (5, false)] {
            v.push(plan_rwkv_act(256, mode, gate).unwrap());
        }
        for head_size in [16usize, 32, 64, 128] {
            let shape = crate::backend::RwkvShape { hidden: 2 * head_size, head_size, group_eps: 1e-5, v7: true };
            v.push(plan_rwkv7_wkv(0, &run, &shape, 0, head_size != 32).unwrap());
            v.push(plan_rwkv6_wkv(0, &run, &shape, 256).unwrap());
        }
        assert!(plan_rwkv_act(0, 0, false).is_err() && plan_rwkv_act(4, 4, false).is_err() && plan_rwkv_act(4, 6, false).is_err());
        for head_dim in [16usize, 32, 64, 128] {
            for lower_bound in [None, Some(-5.0f32)] {
                let shape = crate::backend::KdaShape { heads: 2, head_dim, conv_kernel: 4, eps: 1e-6, lower_bound };
                v.extend(super::recurrent::plan_kda_layer(&shape, &[run], 0).unwrap());
            }
        }
        v.push(super::recurrent::plan_copy_cols(3, 64, 192, 64).unwrap());
        for has_bias in [false, true] {
            v.push(super::recurrent::plan_shortconv(1, &run, 256, 3, 128, has_bias).unwrap());
        }
        assert!(super::recurrent::plan_shortconv(0, &run, 256, 9, 0, false).is_err());
        assert!(super::recurrent::plan_copy_cols(3, 64, 100, 64).is_err());
        let odd = crate::backend::RwkvShape { hidden: 96, head_size: 48, group_eps: 1e-5, v7: true };
        assert!(plan_rwkv7_wkv(0, &run, &odd, 0, false).is_err() && !super::recurrent::rwkv_supported(&odd));
    }
    v.extend(
        plan_moe_expert_gemm(
            &MoeLinear {
                quant: QuantType::Q8_0,
                rows: 8,
                cols: 64,
                slots: 1,
                x_div: 1,
                n_experts: 4,
                has_bias: false,
                access: ExpertAccess::Arena,
                planes: [1, 2, 3],
            },
            2,
            2,
        )
        .unwrap(),
    );
    v
}

#[test]
fn plans_match_the_shader_headers() {
    let plans = sample_plans();
    let mut seen = std::collections::BTreeSet::new();
    for p in &plans {
        seen.insert(p.shader_name);
        let file = source_file(p.shader_name);
        let (bindings, push) = header(&file);
        let block = push_block(&file);
        assert_eq!(
            push, block,
            "{}: header push list differs from the push block of {file}",
            p.shader_name
        );
        assert_eq!(
            p.push_names, push,
            "{}: plan push fields differ from the shader",
            p.shader_name
        );
        assert_eq!(p.push_constants.len(), push.len() * 4, "{}", p.shader_name);
        assert!(p.push_constants.len() <= MAX_PUSH_BYTES);
        let kv_write = p.shader_name.starts_with("kv_write");
        let names: Vec<String> = p
            .bindings
            .iter()
            .map(|b| match (kv_write, b.name) {
                (true, "k" | "v") => "src".to_string(),
                (true, "key_pool" | "value_pool") => "pool".to_string(),
                _ => b.name.to_string(),
            })
            .collect();
        // gemm / route rename a few slots after the fact: compare counts for those
        let renamed = matches!(p.shader_name, "moe_topk" | "moe_combine")
            || p.shader_name.starts_with("moe_linear");
        if renamed {
            assert_eq!(names.len(), bindings.len(), "{}", p.shader_name);
        } else {
            assert_eq!(names, bindings, "{}: binding names", p.shader_name);
        }
        for (i, b) in p.bindings.iter().enumerate() {
            assert_eq!(b.binding as usize, i);
        }
        assert!(
            p.groups.iter().all(|&g| g <= MAX_GROUPS_PER_DIM),
            "{}: {:?}",
            p.shader_name,
            p.groups
        );
    }
    // every shader of the directory is reached by some plan, except the ones deliberately unused (none)
    let mut all: Vec<String> = variants().into_iter().map(|(out, _, _)| out).collect();
    for e in std::fs::read_dir(shader_dir()).unwrap().flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if let Some(stem) = name.strip_suffix(".comp") {
            let listed = variants().iter().any(|(_, src, _)| *src == name);
            if !listed
                && !stem.starts_with('_')
                && !stem.starts_with("scaffold")
                // the three EXL3 passes are recorded by the stream (`stream/linear.rs`) and checked against their modules in `backend/tests.rs`
                && !stem.starts_with("exl3_")
                && !stem.starts_with("mv_")
                && !stem.starts_with("mm_")
                && !stem.starts_with("quant_")
                && !stem.starts_with("lin_")
                // the matrix-unit benchmark of `FLLM_VK_BENCH_LINEAR=1` is not a projection
                && !stem.starts_with("bench_")
            {
                all.push(stem.to_string());
            }
        }
    }
    all.retain(|n| !n.starts_with("mv_") && !n.starts_with("mm_") && !n.starts_with("quant_"));
    let missing: Vec<&String> = all.iter().filter(|n| !seen.contains(n.as_str())).collect();
    assert!(missing.is_empty(), "shaders no plan produces: {missing:?}");
}

#[test]
fn every_shader_source_has_a_consistent_header() {
    let mut count = 0;
    for e in std::fs::read_dir(shader_dir()).unwrap().flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let ours = [
            "attn_", "norm_", "rope_", "act_", "kv_", "moe_", "delta_", "mamba_", "misc_", "embed",
        ]
        .iter()
        .any(|p| name.starts_with(p));
        if name.ends_with(".comp") && ours {
            let (bindings, push) = header(&name);
            assert_eq!(push, push_block(&name), "{name}");
            assert!(!bindings.is_empty());
            assert!(push.len() * 4 <= MAX_PUSH_BYTES, "{name}");
            count += 1;
        }
    }
    assert!(count >= 25, "{count} shaders found");
    for (out, src, defines) in variants() {
        assert!(shader_dir().join(&src).exists(), "{out}: no source {src}");
        assert!(defines.iter().all(|d| d.starts_with("-D")), "{out}");
    }
    let outs: Vec<String> = variants().into_iter().map(|(o, _, _)| o).collect();
    let unique: std::collections::BTreeSet<&String> = outs.iter().collect();
    assert_eq!(outs.len(), unique.len(), "duplicate variant names");
}
