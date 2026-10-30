//! Tests of the prompt-path kernel `mm_q.comp`: its arithmetic (transcribed), the kernel selection and the tile choice.

use super::*;

/// Scale groups per K step of `mm_q.comp` (`NG` of the shader): the formats whose weight scale changes every 16 values.
fn mmq_scale_groups(quant: QuantType) -> usize {
    match quant {
        QuantType::Q2_K | QuantType::Q3_K | QuantType::Q6_K | QuantType::NvFp4 => 2,
        _ => 1,
    }
}

/// Transcription of the arithmetic of `mm_q.comp` for one output: per block of 32 values and scale group the int8 products are summed first (the
/// accumulating dot instruction), then `acc += isum * (ws * d_act) + wm * (d_act * sum(act))`. The scale and offset of a group are those of its first
/// chunk, so every chunk of the group must decode to the same pair (checked here for every format).
#[test]
fn group_arithmetic_matches_the_reference() {
    for &quant in supported_formats() {
        let info = format_info(quant).unwrap();
        let (rows, cols) = (5usize, info.qk.max(256) * 2);
        let raw = sample(quant, rows, cols, 777);
        let packed = pack_weight(quant, &raw, rows, cols).unwrap();
        let weights = crate::compute::dequantize(&raw, quant.to_gguf_type_id(), rows * cols).expect("reference");
        let x = unit_vector(cols, 11);
        let xs: Vec<f32> = match &packed.perm {
            Some(perm) => perm.iter().map(|&p| x[p as usize]).collect(),
            None => x.clone(),
        };
        let ng = mmq_scale_groups(quant);
        let chunks_per_group = 4 / ng;
        let bpr = cols / info.qk;
        for r in 0..rows {
            let (mut got, mut mag, mut want) = (0.0f64, 0.0f64, 0.0f64);
            for (b, block) in xs.chunks(32).enumerate() {
                // quant_act_q8.comp
                let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let d = amax / 127.0;
                let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
                let q: Vec<i32> = block.iter().map(|v| ((v * inv).round_ties_even() as i32).clamp(-127, 127)).collect();
                for g in 0..ng {
                    let mut isum = 0i32;
                    let mut asum = 0i32;
                    let mut lead: Option<(f32, f32)> = None;
                    for c in 0..chunks_per_group {
                        let ch = b * 4 + g * chunks_per_group + c;
                        let decoded = chunk(&packed.layout, &packed.bytes, r * bpr + ch / (info.qk / 8), ch % (info.qk / 8));
                        match lead {
                            None => lead = Some((decoded.ws, decoded.wm)),
                            Some((ws, wm)) => assert!(
                                ws == decoded.ws && wm == decoded.wm,
                                "{quant:?}: the scale changes inside a group of {} values (block {b}, chunk {c})",
                                chunks_per_group * 8
                            ),
                        }
                        let local = (g * chunks_per_group + c) * 8;
                        isum += (0..8).map(|i| decoded.q[i] as i32 * q[local + i]).sum::<i32>();
                        asum += (0..8).map(|i| q[local + i]).sum::<i32>();
                    }
                    let (ws, wm) = lead.unwrap();
                    got += (isum as f32 * (ws * d) + wm * (d * asum as f32)) as f64;
                }
            }
            for k in 0..cols {
                let t = weights[r * cols + k] as f64 * x[k] as f64;
                want += t;
                mag += t.abs();
            }
            assert!((got - want).abs() <= 0.02 * mag + 1e-6, "{quant:?} row {r}: {got} vs {want} (scale {mag})");
        }
    }
}

fn caps_no_coop() -> VkCapsLike {
    VkCapsLike {
        coop_matrix: false,
        ..caps_full()
    }
}

#[test]
fn prompt_batches_use_the_tiled_kernel() {
    let plan = plan_quantized_linear_with(QuantType::Q4_K, 2048, 2048, 505, &caps_no_coop(), &lossy_options()).unwrap();
    assert_eq!(plan.shader_name, "mmq_q4_k_i");
    assert_eq!(plan.variant, Variant::IntDot);
    let values: Vec<u32> = plan.spec_constants.iter().map(|c| c.value).collect();
    // 128 x 64 tile, 8 x 4 outputs per invocation
    assert_eq!(values, [256, 128, 64, 8, 1]);
    assert_eq!(plan.groups, [16, 8, 1]);
    assert!(plan.activation_prepass.is_some());
    // the first batch above the matrix-vector range already takes the tiled kernel (a few concurrent users, speculative drafts)
    let plan = plan_quantized_linear_with(QuantType::Q4_K, 2048, 2048, 16, &caps_no_coop(), &lossy_options()).unwrap();
    assert_eq!(plan.shader_name, "mmq_q4_k_i");
    let plan = plan_quantized_linear_with(QuantType::Q4_K, 2048, 2048, 9, &caps_no_coop(), &lossy_options()).unwrap();
    assert_eq!(plan.shader_name, "mmq_q4_k_i");
    // in the matrix-vector range the matrix-vector kernels serve
    let plan = plan_quantized_linear_with(QuantType::Q4_K, 2048, 2048, 4, &caps_no_coop(), &lossy_options()).unwrap();
    assert_eq!(plan.shader_name, "mv_q4_k_i");
    // a registry without the new kernel falls back
    let old_only = |n: &str| !n.starts_with("mmq_");
    let options = PlanOptions {
        available: Some(&old_only),
        ..lossy_options()
    };
    let plan = plan_quantized_linear_with(QuantType::Q4_K, 2048, 2048, 505, &caps_no_coop(), &options).unwrap();
    assert_eq!(plan.shader_name, "mm_q4_k_i");
    // exact mode and devices without the integer dot product never see it
    let plan = plan_quantized_linear_with(QuantType::Q4_K, 2048, 2048, 505, &VkCapsLike::default(), &lossy_options()).unwrap();
    assert_eq!(plan.shader_name, "mm_q4_k_s");
    let exact = PlanOptions {
        exact: true,
        ..lossy_options()
    };
    let plan = plan_quantized_linear_with(QuantType::Q4_K, 2048, 2048, 505, &caps_no_coop(), &exact).unwrap();
    assert_eq!(plan.shader_name, "mm_q4_k_s");
    // a forced variant keeps the kernel it asked for
    let forced = PlanOptions {
        force: Some(Variant::IntDot),
        ..lossy_options()
    };
    let plan = plan_quantized_linear_with(QuantType::Q4_K, 2048, 2048, 505, &caps_no_coop(), &forced).unwrap();
    assert_eq!(plan.shader_name, "mm_q4_k_i");
}

#[test]
fn tile_choice_follows_the_limits_and_the_knobs() {
    let caps = caps_full();
    let knobs = MmqKnobs::DEFAULT;
    assert_eq!(mmq_tile(&caps, 505, &knobs), Some((64, 64)));
    assert_eq!(mmq_tile(&caps, 9, &knobs), Some((64, 64)));
    assert_eq!(mmq_tile(&caps, 8, &knobs), None);
    assert_eq!(mmq_tile(&caps, 4096, &MmqKnobs { enabled: false, ..knobs }), None);
    assert_eq!(mmq_tile(&caps, 100, &MmqKnobs { min_batch: 200, ..knobs }), None);
    assert_eq!(mmq_tile(&caps, 1000, &MmqKnobs { bm: Some(128), bn: Some(64), ..knobs }), Some((128, 64)));
    assert_eq!(mmq_tile(&caps, 1000, &MmqKnobs { bm: Some(32), bn: Some(128), ..knobs }), Some((32, 128)));
    // a device with small workgroups or little shared memory gets a smaller tile (or none)
    let small_wg = VkCapsLike { max_workgroup: 128, ..caps };
    assert_eq!(mmq_tile(&small_wg, 1000, &knobs), Some((32, 64)));
    let tiny_wg = VkCapsLike { max_workgroup: 64, ..caps };
    assert_eq!(mmq_tile(&tiny_wg, 1000, &knobs), None);
    let small_shared = VkCapsLike { max_shared: 8192, ..caps };
    assert_eq!(mmq_tile(&small_shared, 1000, &knobs), None);
    for batch in [32u32, 100, 505, 2048] {
        for bm in [None, Some(32), Some(64), Some(128)] {
            for bn in [None, Some(64), Some(128)] {
                if let Some((m, n)) = mmq_tile(&caps, batch, &MmqKnobs { bm, bn, ..knobs }) {
                    assert!(m * n / 16 <= caps.max_workgroup && m * n / 16 >= 4 * m && m * n / 16 >= 2 * n, "{m}x{n}");
                    assert!(mmq_shared_bytes(m, n) <= caps.max_shared);
                }
            }
        }
    }
}

/// Opcodes used by a compiled module.
fn opcodes(name: &str) -> std::collections::BTreeSet<u32> {
    let words = crate::backend::vulkan::shaders::spirv(name).unwrap();
    let mut set = std::collections::BTreeSet::new();
    let mut i = 5;
    while i < words.len() {
        set.insert(words[i] & 0xffff);
        i += (words[i] >> 16).max(1) as usize;
    }
    set
}

/// Every glslang since 2022 knows GL_EXT_spirv_intrinsics, so the dot product must be the hardware instruction (OpSDot 4450 / OpSDotAccSat 4453)
/// and not the integer emulation, which costs several instructions per dot product.
#[test]
fn compiled_kernels_use_the_dot_product_instruction() {
    let ops = opcodes("mmq_q4_k_i");
    assert!(ops.contains(&4453), "mmq_q4_k_i has no OpSDotAccSat: opcodes {ops:?}");
    assert!(opcodes("mv_q4_k_i").contains(&4450));
    // the module declares what the instruction needs: capabilities DotProduct (6019) and DotProductInput4x8BitPacked (6018), the extension
    let words = crate::backend::vulkan::shaders::spirv("mmq_q4_k_i").unwrap();
    let (mut caps, mut extension) = (Vec::new(), false);
    let mut i = 5;
    while i < words.len() {
        let (op, count) = (words[i] & 0xffff, (words[i] >> 16).max(1) as usize);
        if op == 17 {
            caps.push(words[i + 1]);
        }
        if op == 10 {
            let text: Vec<u8> = words[i + 1..i + count].iter().flat_map(|w| w.to_le_bytes()).collect();
            extension |= text.starts_with(b"SPV_KHR_integer_dot_product");
        }
        i += count;
    }
    assert!(caps.contains(&6019) && caps.contains(&6018) && extension, "capabilities {caps:?}, extension declared: {extension}");
}
