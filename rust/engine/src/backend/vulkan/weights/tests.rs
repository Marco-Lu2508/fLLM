//! CPU tests of the Vulkan weight layouts and the dispatch planner (no GPU involved).

use super::layout::*;
use super::mirror::chunk;
use super::plan::*;
use crate::backend::QuantType;
use crate::cpu_selftest::{random_weights, Lcg};

/// GGUF type of the blocks of `quant` (the Q4ZP formats store the Q4Z blocks followed by a permutation table).
fn block_kind(quant: QuantType) -> u32 {
    let id = quant.to_gguf_type_id();
    if quant.is_q4zp() {
        id - 4
    } else {
        id
    }
}

fn sample(quant: QuantType, rows: usize, cols: usize, seed: u32) -> Vec<u8> {
    let mut rng = Lcg(seed.into());
    random_weights(quant.to_gguf_type_id(), rows, cols, &mut rng)
}

fn unit_vector(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn lossy_options() -> PlanOptions<'static> {
    PlanOptions {
        exact: false,
        ..PlanOptions::default()
    }
}

#[test]
fn every_supported_format_has_a_layout_with_aligned_planes() {
    for &quant in supported_formats() {
        let info = format_info(quant).expect("format info");
        let layout = DeviceLayout::new(quant, 7, info.qk * 3).expect("layout");
        assert_eq!(layout.blocks(), 21);
        for plane in 0..4 {
            assert_eq!(
                layout.plane_byte_offset(plane) % 16,
                0,
                "{quant:?} plane {plane}"
            );
        }
        assert_eq!(layout.total_bytes % 16, 0);
        // the planes together hold the same payload as the file (K-quant scales are widened: at most 20 % more)
        let file = 21 * info.src_block_bytes;
        assert!(
            layout.total_bytes >= file && layout.total_bytes <= file * 6 / 5 + 128,
            "{quant:?}: {} vs {file}",
            layout.total_bytes
        );
    }
}

#[test]
fn chunks_decode_to_the_reference_weights() {
    for &quant in supported_formats() {
        let info = format_info(quant).unwrap();
        let (rows, cols) = (5usize, info.qk * 2);
        let raw = sample(quant, rows, cols, 0x5eed_f00d);
        let packed = pack_weight(quant, &raw, rows, cols).expect("pack");
        let block_len = rows * cols / info.qk * info.src_block_bytes;
        let want = crate::compute::dequantize(&raw[..block_len], block_kind(quant), rows * cols)
            .expect("reference");
        let bpr = cols / info.qk;
        for r in 0..rows {
            for b in 0..bpr {
                for cin in 0..info.qk / 8 {
                    let c = chunk(&packed.layout, &packed.bytes, r * bpr + b, cin);
                    for i in 0..8 {
                        let got = c.ws * c.q[i] as f32 + c.wm;
                        let at = r * cols + b * info.qk + cin * 8 + i;
                        let tol = 1e-6 * (c.ws * c.q[i] as f32).abs().max(c.wm.abs()) + 1e-9;
                        assert!(
                            (got - want[at]).abs() <= tol,
                            "{quant:?} row {r} block {b} chunk {cin} value {i}: {got} vs {}",
                            want[at]
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn q4zp_returns_the_permutation_and_rejects_bad_tables() {
    let (rows, cols) = (3usize, 128usize);
    let raw = sample(QuantType::Q4ZP64, rows, cols, 7);
    let packed = pack_weight(QuantType::Q4ZP64, &raw, rows, cols).unwrap();
    let perm = packed.perm.expect("permutation");
    assert_eq!(perm.len(), cols);
    let mut sorted = perm.clone();
    sorted.sort_unstable();
    assert!(sorted.iter().enumerate().all(|(i, &p)| i as u32 == p));
    let mut bad = raw.clone();
    let n = bad.len();
    bad[n - 4..].copy_from_slice(&(cols as u32).to_le_bytes());
    assert!(pack_weight(QuantType::Q4ZP64, &bad, rows, cols).is_err());
    assert!(pack_weight(QuantType::Q4ZP64, &raw[..raw.len() - 4], rows, cols).is_err());
    assert!(pack_weight(QuantType::Q4Z64, &raw, rows, cols).is_err());
}

/// Dot product of every weight row with `x` computed chunk by chunk like `mv_scalar.comp` (blocks `ch / (QK / 8)`, chunk `ch % (QK / 8)`),
/// against the f64 dot product of the reference weights.
#[test]
fn row_dot_through_chunks_matches_the_reference() {
    for &quant in supported_formats() {
        let info = format_info(quant).unwrap();
        let (rows, cols) = (4usize, info.qk * 3);
        let raw = sample(quant, rows, cols, 99);
        let packed = pack_weight(quant, &raw, rows, cols).unwrap();
        // the reference holds the weights in original column order; Q4ZP shaders see x permuted by perm
        let weights = crate::compute::dequantize(&raw, quant.to_gguf_type_id(), rows * cols)
            .expect("reference");
        let x = unit_vector(cols, 3);
        let xs: Vec<f32> = match &packed.perm {
            Some(perm) => perm.iter().map(|&p| x[p as usize]).collect(),
            None => x.clone(),
        };
        let bpr = cols / info.qk;
        for r in 0..rows {
            let (mut got, mut mag) = (0.0f64, 0.0f64);
            for ch in 0..cols / 8 {
                let c = chunk(
                    &packed.layout,
                    &packed.bytes,
                    r * bpr + ch / (info.qk / 8),
                    ch % (info.qk / 8),
                );
                let mut dot = 0.0f32;
                let mut sum = 0.0f32;
                for i in 0..8 {
                    dot += c.q[i] as f32 * xs[ch * 8 + i];
                    sum += xs[ch * 8 + i];
                }
                got += (c.ws * dot + c.wm * sum) as f64;
            }
            let mut want = 0.0f64;
            for k in 0..cols {
                let t = weights[r * cols + k] as f64 * x[k] as f64;
                want += t;
                mag += t.abs();
            }
            assert!(
                (got - want).abs() <= 1e-5 * mag + 1e-6,
                "{quant:?} row {r}: {got} vs {want}"
            );
        }
    }
}

/// The integer-dot arithmetic of `mv_idot.comp` (int8 activations in blocks of 32, `ws * da * idot + wm * da * sum`) against the f64 dot product.
#[test]
fn integer_dot_arithmetic_matches_within_the_activation_rounding() {
    for &quant in supported_formats() {
        let info = format_info(quant).unwrap();
        let (rows, cols) = (4usize, info.qk * 3);
        let raw = sample(quant, rows, cols, 1234);
        let packed = pack_weight(quant, &raw, rows, cols).unwrap();
        let weights = crate::compute::dequantize(&raw, quant.to_gguf_type_id(), rows * cols)
            .expect("reference");
        let x = unit_vector(cols, 5);
        let xs: Vec<f32> = match &packed.perm {
            Some(perm) => perm.iter().map(|&p| x[p as usize]).collect(),
            None => x.clone(),
        };
        // quant_act_q8.comp
        let mut q = vec![0i32; cols];
        let mut d = vec![0f32; cols / 32];
        for (b, block) in xs.chunks(32).enumerate() {
            let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            d[b] = amax / 127.0;
            let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
            for (i, v) in block.iter().enumerate() {
                q[b * 32 + i] = ((v * inv).round_ties_even() as i32).clamp(-127, 127);
            }
        }
        let bpr = cols / info.qk;
        for r in 0..rows {
            let (mut got, mut mag, mut want) = (0.0f64, 0.0f64, 0.0f64);
            for ch in 0..cols / 8 {
                let c = chunk(
                    &packed.layout,
                    &packed.bytes,
                    r * bpr + ch / (info.qk / 8),
                    ch % (info.qk / 8),
                );
                let da = d[ch / 4];
                let idot: i32 = (0..8).map(|i| c.q[i] as i32 * q[ch * 8 + i]).sum();
                let asum: i32 = (0..8).map(|i| q[ch * 8 + i]).sum();
                got += (c.ws * da * idot as f32 + c.wm * da * asum as f32) as f64;
            }
            for k in 0..cols {
                let t = weights[r * cols + k] as f64 * x[k] as f64;
                want += t;
                mag += t.abs();
            }
            assert!(
                (got - want).abs() <= 0.02 * mag + 1e-6,
                "{quant:?} row {r}: {got} vs {want} (scale {mag})"
            );
        }
    }
}

/// Writes the packed weights and the reference dequantization of every format to `$FLLM_VK_DUMP_DIR/<tag>.bin` (no-op when the variable is
/// unset). `shaders/vulkan/emu/check_decoders.sh` and `run_kernels.sh` run the GLSL text as C++ against these files.
/// Header: eight u32 (rows, cols, qk, blocks per row, plane offsets 1..3, byte length), the device bytes, then `rows * cols` f32.
#[test]
fn dump_vectors_for_the_glsl_transpile_check() {
    let Ok(dir) = std::env::var("FLLM_VK_DUMP_DIR") else {
        return;
    };
    for &quant in supported_formats() {
        if quant.is_q4zp() {
            continue; // same blocks as the Q4Z format
        }
        let info = format_info(quant).unwrap();
        for (rows, cols, suffix) in [
            (9usize, info.qk * 3, ""),
            (70usize, (info.qk * 2).max(256), "_big"),
        ] {
            let raw = sample(quant, rows, cols, 0xc0ffee);
            let packed = pack_weight(quant, &raw, rows, cols).unwrap();
            let want = crate::compute::dequantize(&raw, block_kind(quant), rows * cols).unwrap();
            let l = &packed.layout;
            let mut out = Vec::new();
            for v in [
                rows as u32,
                cols as u32,
                info.qk as u32,
                l.blocks_per_row as u32,
                l.plane_word_offset[1],
                l.plane_word_offset[2],
                l.plane_word_offset[3],
                packed.bytes.len() as u32,
            ] {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out.extend_from_slice(&packed.bytes);
            for v in &want {
                out.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(format!("{dir}/{}{suffix}.bin", info.tag), out).unwrap();
        }
    }
}

#[test]
fn rows_per_slice_keeps_a_slice_within_the_range() {
    for &quant in supported_formats() {
        let info = format_info(quant).unwrap();
        let cols = info.qk * 16;
        let max_range = 1usize << 24;
        let rows = rows_per_slice(quant, cols, max_range).unwrap();
        assert!(rows >= 64 && rows.is_multiple_of(64));
        let layout = DeviceLayout::new(quant, rows, cols).unwrap();
        assert!(
            layout.total_bytes <= max_range,
            "{quant:?}: {} > {max_range}",
            layout.total_bytes
        );
    }
}

/// A weight uploaded as slices of rows decodes to the same values as the whole weight, and each slice knows its `row_base`.
#[test]
fn row_slices_decode_like_the_whole_weight() {
    for quant in [QuantType::Q6_K, QuantType::Q4ZP64, QuantType::Mxfp4] {
        let info = format_info(quant).unwrap();
        let (rows, cols) = (150usize, info.qk * 2);
        let raw = sample(quant, rows, cols, 21);
        let whole = pack_weight(quant, &raw, rows, cols).unwrap();
        let slices = pack_weight_slices(quant, &raw, rows, cols, 64).unwrap();
        assert_eq!(
            slices
                .iter()
                .map(|s| s.packed.layout.rows)
                .collect::<Vec<_>>(),
            [64, 64, 22]
        );
        assert_eq!(
            slices.iter().map(|s| s.row_base).collect::<Vec<_>>(),
            [0, 64, 128]
        );
        let bpr = cols / info.qk;
        for slice in &slices {
            assert_eq!(slice.packed.perm, whole.perm);
            for r in [0, slice.packed.layout.rows - 1] {
                for ch in 0..cols / 8 {
                    let (g, cin) = (r * bpr + ch / (info.qk / 8), ch % (info.qk / 8));
                    let a = chunk(&slice.packed.layout, &slice.packed.bytes, g, cin);
                    let g_whole = (slice.row_base + r) * bpr + ch / (info.qk / 8);
                    let b = chunk(&whole.layout, &whole.bytes, g_whole, cin);
                    assert_eq!(
                        (a.q, a.ws, a.wm),
                        (b.q, b.ws, b.wm),
                        "{quant:?} slice at {} row {r} chunk {ch}",
                        slice.row_base
                    );
                }
            }
        }
    }
    assert!(pack_weight_slices(QuantType::Q4_0, &[0u8; 18], 2, 32, 1).is_err());
}

#[test]
fn layout_rejects_bad_shapes() {
    assert!(DeviceLayout::new(QuantType::Q4_K, 4, 200).is_err());
    assert!(DeviceLayout::new(QuantType::Q4_K, 0, 256).is_err());
    assert!(DeviceLayout::new(QuantType::F32, 4, 256).is_err());
    assert!(DeviceLayout::new(QuantType::IQ2_XS, 4, 256).is_err());
    assert!(pack_weight(QuantType::Q4_0, &[0u8; 17], 1, 32).is_err());
}

fn caps_full() -> VkCapsLike {
    VkCapsLike {
        coop_matrix: true,
        int_dot: true,
        subgroup_size: 32,
        fp16: true,
        max_shared: 49152,
        vendor: 0x10de,
    }
}

fn le_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect()
}

#[test]
fn decode_on_a_basic_device_uses_the_scalar_shader() {
    let plan = plan_quantized_linear_with(
        QuantType::Q4_0,
        4096,
        4096,
        1,
        &VkCapsLike::default(),
        &lossy_options(),
    )
    .unwrap();
    assert_eq!(plan.shader_name, "mv_q4_0_s");
    assert_eq!(plan.variant, Variant::Scalar);
    assert_eq!(plan.kind, PlanKind::MatVec { columns: 1 });
    assert!(plan.activation_prepass.is_none());
    assert_eq!(plan.groups, [2048, 1, 1]);
    assert_eq!(plan.spec_constants[2], SpecConstant { id: 2, value: 1 });
    assert_eq!(plan.push_constants.len(), PUSH_BYTES);
    let w = le_words(&plan.push_constants);
    assert_eq!(&w[..4], &[4096, 4096, 1, 0]);
    assert_eq!(f32::from_bits(w[4]), 1.0);
    assert_eq!(w[5], 4096 / 32);
    assert_eq!(w[6], plan.layout.plane_word_offset[1]);
    assert_eq!(w[9], 4096);
    assert_eq!(
        plan.bindings.iter().map(|b| b.role).collect::<Vec<_>>(),
        [
            BindingRole::Weights,
            BindingRole::Activations,
            BindingRole::Output,
            BindingRole::Bias
        ]
    );
}

#[test]
fn decode_with_integer_dot_quantizes_the_activations_first() {
    let options = PlanOptions {
        has_bias: true,
        scale: 0.5,
        ..lossy_options()
    };
    let plan =
        plan_quantized_linear_with(QuantType::Q4_K, 1000, 2048, 3, &caps_full(), &options).unwrap();
    assert_eq!(plan.shader_name, "mv_q4_k_i");
    assert_eq!(plan.kind, PlanKind::MatVec { columns: 3 });
    let pre = plan.activation_prepass.as_ref().expect("prepass");
    assert_eq!(pre.shader_name, "quant_act_q8");
    assert_eq!(pre.kind, PrepassKind::QuantizeActivations);
    assert_eq!(pre.output_bytes, (3 * 2048 / 4 + 3 * 2048 / 32) * 4);
    assert_eq!(le_words(&pre.push_constants), [2048, 3, 3 * 2048 / 4]);
    assert_eq!(pre.groups, [(3 * 2048 / 32u32).div_ceil(128), 1, 1]);
    assert_eq!(plan.bindings[1].role, BindingRole::PrepassOutput);
    let w = le_words(&plan.push_constants);
    assert_eq!(w[3], 1, "bias flag");
    assert_eq!(f32::from_bits(w[4]), 0.5);
    assert_eq!(w[11], 3 * 2048 / 4, "xq_d");
}

#[test]
fn prefill_prefers_cooperative_matrices_then_integer_dot_then_f32() {
    let plan = plan_quantized_linear_with(
        QuantType::Q6_K,
        4096,
        4096,
        512,
        &caps_full(),
        &lossy_options(),
    )
    .unwrap();
    assert_eq!(
        (plan.shader_name.as_str(), plan.variant),
        ("mm_q6_k_c", Variant::CoopMat)
    );
    assert_eq!(plan.spec_constants, [SpecConstant { id: 0, value: 128 }]);
    assert_eq!(plan.groups, [64, 8, 1]);
    assert!(plan.activation_prepass.is_none());

    let no_coop = VkCapsLike {
        coop_matrix: false,
        ..caps_full()
    };
    let plan =
        plan_quantized_linear_with(QuantType::Q6_K, 4096, 4096, 512, &no_coop, &lossy_options())
            .unwrap();
    assert_eq!(plan.shader_name, "mm_q6_k_i");
    assert!(plan.activation_prepass.is_some());

    let plan = plan_quantized_linear_with(
        QuantType::Q6_K,
        4096,
        4096,
        512,
        &VkCapsLike::default(),
        &lossy_options(),
    )
    .unwrap();
    assert_eq!(plan.shader_name, "mm_q6_k_s");

    // a registry without the cooperative-matrix shader (old compiler) falls back
    let only_noncoop = |n: &str| n != "mm_q6_k_c";
    let options = PlanOptions {
        available: Some(&only_noncoop),
        ..lossy_options()
    };
    let plan = plan_quantized_linear_with(QuantType::Q6_K, 4096, 4096, 512, &caps_full(), &options)
        .unwrap();
    assert_eq!(plan.shader_name, "mm_q6_k_i");
}

#[test]
fn exact_mode_and_missing_shaders_are_handled() {
    let options = PlanOptions {
        exact: true,
        ..lossy_options()
    };
    let plan =
        plan_quantized_linear_with(QuantType::Mxfp4, 256, 256, 1, &caps_full(), &options).unwrap();
    assert_eq!(plan.shader_name, "mv_mxfp4_s");
    let none = |_: &str| false;
    let options = PlanOptions {
        available: Some(&none),
        ..lossy_options()
    };
    assert!(
        plan_quantized_linear_with(QuantType::Mxfp4, 256, 256, 1, &caps_full(), &options).is_err()
    );
    let options = PlanOptions {
        force: Some(Variant::CoopMat),
        ..lossy_options()
    };
    assert!(
        plan_quantized_linear_with(QuantType::Mxfp4, 256, 256, 1, &caps_full(), &options).is_err(),
        "coopmat has no matvec"
    );
    assert!(plan_quantized_linear_with(
        QuantType::Mxfp4,
        256,
        256,
        0,
        &caps_full(),
        &lossy_options()
    )
    .is_err());
    assert!(plan_quantized_linear_with(
        QuantType::F16,
        256,
        256,
        1,
        &caps_full(),
        &lossy_options()
    )
    .is_err());
}

#[test]
fn act_order_weights_permute_the_activations() {
    let scalar = plan_quantized_linear_with(
        QuantType::Q4ZP128,
        512,
        1024,
        1,
        &VkCapsLike::default(),
        &lossy_options(),
    )
    .unwrap();
    assert_eq!(scalar.shader_name, "mv_q4z128_s");
    let pre = scalar.activation_prepass.as_ref().unwrap();
    assert_eq!(
        (pre.shader_name.as_str(), pre.kind),
        ("quant_permute", PrepassKind::PermuteActivations)
    );
    assert_eq!(le_words(&pre.push_constants), [1024, 1024]);
    assert_eq!(pre.output_bytes, 4096);
    assert!(pre.bindings.iter().any(|b| b.role == PrepassRole::Perm));
    assert_eq!(scalar.bindings[1].role, BindingRole::PrepassOutput);

    let idot = plan_quantized_linear_with(
        QuantType::Q4ZP32,
        512,
        1024,
        4,
        &caps_full(),
        &lossy_options(),
    )
    .unwrap();
    assert_eq!(idot.shader_name, "mv_q4z32_i");
    assert_eq!(
        idot.activation_prepass.as_ref().unwrap().shader_name,
        "quant_act_q8_perm"
    );
    assert!(idot.activation_prepass.as_ref().unwrap().uses_perm);

    // plain Q4Z has no prepass in f32
    let plain = plan_quantized_linear_with(
        QuantType::Q4Z64,
        512,
        1024,
        1,
        &VkCapsLike::default(),
        &lossy_options(),
    )
    .unwrap();
    assert!(plain.activation_prepass.is_none());
}

#[test]
fn big_grids_are_split_over_x_and_y() {
    assert_eq!(split_groups(1), [1, 1, 1]);
    assert_eq!(split_groups(65535), [65535, 1, 1]);
    assert_eq!(split_groups(65536), [65535, 2, 1]);
    let plan = plan_quantized_linear_with(
        QuantType::Q6_K,
        152_064,
        2048,
        1,
        &caps_full(),
        &PlanOptions {
            force: Some(Variant::Scalar),
            ..lossy_options()
        },
    )
    .unwrap();
    assert_eq!(plan.groups, [65535, 2, 1], "76032 workgroups");
}

#[test]
fn matvec_tuning_respects_the_shared_memory_limit() {
    let small = VkCapsLike {
        max_shared: 4096,
        ..VkCapsLike::default()
    };
    let plan =
        plan_quantized_linear_with(QuantType::Q8_0, 256, 256, 8, &small, &lossy_options()).unwrap();
    let wg = plan.spec_constants[0].value;
    let per = plan.spec_constants[1].value;
    assert!(per * 8 * wg * 4 <= 4096, "wg {wg} rows {per}");
    assert!(wg.is_power_of_two());
}
