//! Consistency checks of the Vulkan backend that need no device: every shader the planners can ask for exists in the SPIR-V registry, and
//! the bindings, push-constant sizes and specialization constants of every plan agree with what the compiled SPIR-V module declares.
//! (The SPIR-V is read by a small reflection below, so the check is against the compiled binary, not against the comments of the sources.)

use crate::backend::vulkan::ops::attention::{plan_attention, plan_kv_write, AttentionTuning, KvLayout};
use crate::backend::vulkan::ops::moe::{
    plan_gather_rows, plan_moe_combine, plan_moe_expert_gemm, plan_moe_linear, plan_moe_topk, plan_scatter_add_rows, ExpertAccess,
    MoeLinear,
};
use crate::backend::vulkan::ops::recurrent::{
    plan_delta_layer, plan_delta_layer_with, plan_mamba1_bc_norms, plan_mamba1_scan, plan_mamba2_mixer, plan_mamba_conv, Mamba1Shape, Mamba2Shape, RunSpec,
};
use crate::backend::vulkan::ops::{
    plan_argmax, plan_embedding, plan_glu, plan_layer_norm, plan_residual_add, plan_rms_norm, plan_rms_norm_grouped, plan_rope, plan_sigmoid_gate_mul,
    plan_split_q_gate, EmbeddingTable, GluMode, OpPlan, SlotSource,
};
use crate::backend::vulkan::shaders;
use crate::backend::vulkan::stream::plan_float_linear;
use crate::backend::vulkan::weights::{
    exl2_device_type, extended_formats, plan_quantized_linear_with, supported_formats, uses_perm, PlanOptions, VkCapsLike, MATVEC_MAX_BATCH, PUSH_BYTES,
};
use crate::backend::{AttentionParams, DeltaShape, QuantType};
use std::collections::{BTreeSet, HashMap, HashSet};

// ---------------------------------------------------------------------------------------------------------------------------------------------
// SPIR-V reflection
// ---------------------------------------------------------------------------------------------------------------------------------------------

#[derive(Debug, Default)]
struct ShaderInfo {
    /// Binding numbers of the storage buffers in descriptor set 0 (sorted).
    bindings: Vec<u32>,
    /// Descriptor sets used by any storage buffer.
    sets: BTreeSet<u32>,
    /// Size of the push-constant block (end of the last member), `None` without one.
    push_bytes: Option<u32>,
    /// Members of the push block that are not 32-bit scalars.
    push_non_scalar: bool,
    spec_ids: BTreeSet<u32>,
}

const OP_TYPE_INT: u32 = 21;
const OP_TYPE_FLOAT: u32 = 22;
const OP_TYPE_STRUCT: u32 = 30;
const OP_TYPE_POINTER: u32 = 32;
const OP_VARIABLE: u32 = 59;
const OP_DECORATE: u32 = 71;
const OP_MEMBER_DECORATE: u32 = 72;
const DEC_SPEC_ID: u32 = 1;
const DEC_BLOCK: u32 = 2;
const DEC_BUFFER_BLOCK: u32 = 3;
const DEC_BINDING: u32 = 33;
const DEC_DESCRIPTOR_SET: u32 = 34;
const DEC_OFFSET: u32 = 35;
const CLASS_UNIFORM: u32 = 2;
const CLASS_PUSH_CONSTANT: u32 = 9;
const CLASS_STORAGE_BUFFER: u32 = 12;

fn reflect(words: &[u32]) -> ShaderInfo {
    assert_eq!(words[0], 0x0723_0203, "not a SPIR-V module");
    let mut binding: HashMap<u32, u32> = HashMap::new();
    let mut set: HashMap<u32, u32> = HashMap::new();
    let mut block: HashSet<u32> = HashSet::new();
    let mut buffer_block: HashSet<u32> = HashSet::new();
    let mut member_offset: HashMap<(u32, u32), u32> = HashMap::new();
    let mut variables: Vec<(u32, u32, u32)> = Vec::new(); // (id, pointer type, class)
    let mut pointers: HashMap<u32, (u32, u32)> = HashMap::new(); // id -> (class, pointee)
    let mut structs: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut scalar32: HashSet<u32> = HashSet::new();
    let mut info = ShaderInfo::default();
    let mut i = 5;
    while i < words.len() {
        let count = (words[i] >> 16) as usize;
        let op = words[i] & 0xffff;
        assert!(count > 0 && i + count <= words.len(), "truncated SPIR-V instruction");
        let operands = &words[i + 1..i + count];
        match op {
            OP_DECORATE => match operands[1] {
                DEC_BINDING => {
                    binding.insert(operands[0], operands[2]);
                }
                DEC_DESCRIPTOR_SET => {
                    set.insert(operands[0], operands[2]);
                }
                DEC_SPEC_ID => {
                    info.spec_ids.insert(operands[2]);
                }
                DEC_BLOCK => {
                    block.insert(operands[0]);
                }
                DEC_BUFFER_BLOCK => {
                    buffer_block.insert(operands[0]);
                }
                _ => {}
            },
            OP_MEMBER_DECORATE if operands[2] == DEC_OFFSET => {
                member_offset.insert((operands[0], operands[1]), operands[3]);
            }
            OP_VARIABLE => variables.push((operands[1], operands[0], operands[2])),
            OP_TYPE_POINTER => {
                pointers.insert(operands[0], (operands[1], operands[2]));
            }
            OP_TYPE_STRUCT => {
                structs.insert(operands[0], operands[1..].to_vec());
            }
            OP_TYPE_INT | OP_TYPE_FLOAT if operands[1] == 32 => {
                scalar32.insert(operands[0]);
            }
            _ => {}
        }
        i += count;
    }
    for (id, pointer, class) in variables {
        let Some(&(_, pointee)) = pointers.get(&pointer) else { continue };
        match class {
            CLASS_STORAGE_BUFFER => {
                info.bindings.push(binding[&id]);
                info.sets.insert(set.get(&id).copied().unwrap_or(0));
            }
            CLASS_UNIFORM if buffer_block.contains(&pointee) => {
                info.bindings.push(binding[&id]);
                info.sets.insert(set.get(&id).copied().unwrap_or(0));
            }
            CLASS_PUSH_CONSTANT => {
                let members = &structs[&pointee];
                let mut end = 0;
                for (m, ty) in members.iter().enumerate() {
                    let offset = member_offset[&(pointee, m as u32)];
                    info.push_non_scalar |= !scalar32.contains(ty);
                    end = end.max(offset + 4);
                }
                info.push_bytes = Some(end);
                assert!(block.contains(&pointee), "push-constant struct without Block");
            }
            _ => {}
        }
    }
    info.bindings.sort_unstable();
    info
}

fn info_of(name: &str) -> ShaderInfo {
    let words = shaders::spirv(name).unwrap_or_else(|| panic!("the shader '{name}' is not in the SPIR-V registry"));
    reflect(&words)
}

/// The plan of an operation agrees with the module it names.
fn check_op_plan(plan: &OpPlan, seen: &mut BTreeSet<String>) {
    let name = plan.shader_name;
    seen.insert(name.to_owned());
    let info = info_of(name);
    let n = plan.bindings.len() as u32;
    // the optimiser drops a binding no code of the module reads; that is only acceptable for a slot the plan fills with the dummy buffer
    assert!(
        info.bindings.iter().all(|&b| b < n),
        "{name}: the module binds {:?}, the plan has {n} slots",
        info.bindings
    );
    for slot in &plan.bindings {
        if !info.bindings.contains(&slot.binding) {
            assert!(
                matches!(slot.source, SlotSource::Dummy),
                "{name}: slot {} '{}' is not in the module (bindings {:?}) but the plan binds a real buffer to it",
                slot.binding,
                slot.name,
                info.bindings
            );
        }
    }
    assert!(info.sets.iter().all(|&s| s == 0), "{name}: descriptor sets {:?}", info.sets);
    assert!(
        !info.push_non_scalar,
        "{name}: the push block has members that are not 32-bit scalars"
    );
    assert_eq!(
        info.push_bytes,
        Some(plan.push_constants.len() as u32),
        "{name}: push-constant size of the module vs the plan"
    );
    assert_eq!(plan.push_names.len() * 4, plan.push_constants.len(), "{name}: field names vs bytes");
    for (index, slot) in plan.bindings.iter().enumerate() {
        assert_eq!(slot.binding as usize, index, "{name}: slots must be numbered in order");
    }
    assert!(plan.groups.iter().all(|&g| g <= 65535), "{name}: grid {:?}", plan.groups);
    assert!(plan.groups[0] > 0, "{name}: empty grid");
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Operation plans
// ---------------------------------------------------------------------------------------------------------------------------------------------

fn kv_layout(kind: i32, kv_heads: usize, head_dim: usize, value_dim: usize) -> KvLayout {
    KvLayout {
        kind,
        kv_heads,
        head_dim,
        value_dim,
        block_size: 16,
        pool_blocks: 1024,
        key_base: 3,
        value_base: 5,
    }
}

fn attention_params(heads: usize, kv_heads: usize, dk: usize, dv: usize, fancy: bool) -> AttentionParams {
    AttentionParams {
        num_heads: heads,
        num_kv_heads: kv_heads,
        head_dim: dk,
        value_dim: dv,
        stride: 256,
        block_size: 16,
        scale: 0.125,
        softcap: fancy.then_some(30.0),
        window: fancy.then_some(128),
        chunk: None,
        sinks: fancy.then(|| std::sync::Arc::new(vec![0.5; heads])),
        alibi_slopes: fancy.then(|| std::sync::Arc::new(vec![0.25; heads])),
    }
}

fn all_op_plans() -> Vec<OpPlan> {
    let mut plans = Vec::new();
    for has_weight in [true, false] {
        plans.push(plan_rms_norm(4, 256, 1e-6, has_weight).unwrap());
    }
    plans.push(plan_rms_norm_grouped(4, 256, 4, 1e-6, true).unwrap());
    for has_bias in [true, false] {
        plans.push(plan_layer_norm(4, 256, 1e-5, has_bias).unwrap());
    }
    let positions: Vec<usize> = (0..5).collect();
    plans.push(plan_rope(&positions, 8, 2, 64, 64, true, 16).unwrap());
    plans.push(plan_rope(&positions, 8, 2, 64, 32, false, 16).unwrap());
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
        plans.push(
            plan_embedding(
                &EmbeddingTable {
                    quant,
                    rows: 100,
                    cols: 512,
                    planes: [1, 2, 3],
                },
                &[1, 2, 3],
            )
            .unwrap(),
        );
    }
    for mode in [
        GluMode::Swiglu,
        GluMode::Geglu,
        GluMode::SwigluOai { alpha: 1.702, limit: 7.0 },
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
        plans.push(plan_glu(mode, 1000).unwrap());
    }
    for softplus in [false, true] {
        plans.push(crate::backend::vulkan::ops::plan_ewise(softplus, 1000, 0.7).unwrap());
    }
    plans.push(crate::backend::vulkan::ops::plan_scale(1000, 0.7).unwrap());
    for beta in [None, Some(0.7f32)] {
        plans.push(crate::backend::vulkan::ops::plan_head_gate(1000, 8, beta).unwrap());
    }
    {
        use crate::backend::vulkan::ops::{plan_mla_move, MlaPart};
        for part in [MlaPart::Q, MlaPart::K, MlaPart::V, MlaPart::Latent] {
            plans.push(plan_mla_move(part, 3, 4, 16, 8, 16, 32).unwrap());
        }
    }
    plans.push(plan_residual_add(1000, 0.5).unwrap());
    plans.push(plan_sigmoid_gate_mul(1000).unwrap());
    for per_head in [true, false] {
        plans.push(plan_split_q_gate(3, 4, 64, per_head).unwrap());
    }
    plans.push(plan_argmax(3, 1000).unwrap());

    for kind in [1, 2] {
        for (kv_heads, dk, dv) in [(2, 64, 64), (2, 80, 64), (1, 128, 128)] {
            plans.extend(plan_kv_write(&kv_layout(kind, kv_heads, dk, dv), &[0, 5, 9]).unwrap());
        }
    }
    let wide = AttentionTuning { prefill_rows: crate::backend::vulkan::ops::attention::PREFILL_ROWS_WIDE, ..AttentionTuning::default() };
    for tuning in [AttentionTuning::default(), wide, AttentionTuning { coop_prefill: true, ..wide }] {
    for kind in [1, 2] {
        for (heads, kv_heads, dk, dv) in [
            (8, 2, 64, 64),
            (4, 4, 128, 128),
            (4, 1, 192, 128),
            (8, 2, 256, 256),
            (2, 1, 512, 512),
        ] {
            for fancy in [false, true] {
                let layout = kv_layout(kind, kv_heads, dk, dv);
                let params = attention_params(heads, kv_heads, dk, dv, fancy);
                let table: Vec<i64> = (0..params.stride as i64).collect();
                // decode: a few rows of different lengths, one of them long enough to be split over the key range
                let lengths = [300i64, 5, 4000];
                let tables: Vec<i64> = lengths.iter().flat_map(|_| table.iter().copied()).collect();
                plans.extend(plan_attention(&layout, &params, &tables, &lengths, &tuning).unwrap());
                // prompt: twenty rows of one sequence
                let lengths: Vec<i64> = (1..=20).collect();
                let tables: Vec<i64> = lengths.iter().flat_map(|_| table.iter().copied()).collect();
                plans.extend(plan_attention(&layout, &params, &tables, &lengths, &tuning).unwrap());
            }
        }
    }
    }

    plans.push(plan_moe_topk(3, 64, 8, true, 1.0).unwrap());
    plans.push(plan_moe_topk(3, 64, 8, false, 2.5).unwrap());
    for quant in [
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
        for has_bias in [false, true] {
            let call = MoeLinear {
                quant,
                rows: 96,
                cols: 512,
                slots: 12,
                x_div: 4,
                n_experts: 8,
                has_bias,
                access: ExpertAccess::Arena,
                planes: [1, 2, 3],
            };
            plans.push(plan_moe_linear(&call).unwrap());
            if has_bias {
                plans.extend(plan_moe_expert_gemm(&call, 3, 4).unwrap());
            }
        }
    }
    plans.push(plan_moe_combine(3, 4, 256).unwrap());
    plans.push(plan_gather_rows(&[3, 1, 2], Some(&[0.5, 0.25, 1.0]), 64).unwrap());
    plans.push(plan_gather_rows(&[3, 1, 2], None, 64).unwrap());
    plans.push(plan_scatter_add_rows(&[3, 1, 2], Some(&[0.5, 0.25, 1.0]), 64).unwrap());
    plans.push(plan_scatter_add_rows(&[3, 1, 2], None, 64).unwrap());

    for hk in [32, 64, 96, 128] {
        let shape = DeltaShape {
            k_heads: 2,
            v_heads: 4,
            k_dim: hk,
            v_dim: 64,
            conv_kernel: 4,
            tiled_heads: false,
            beta_scale: 1.0,
            eps: 1e-6,
        };
        let runs = [
            RunSpec {
                first_row: 0,
                rows: 10,
                reset: true,
            },
            RunSpec {
                first_row: 10,
                rows: 1,
                reset: false,
            },
        ];
        plans.extend(plan_delta_layer(&shape, &runs, 256).unwrap());
        plans.extend(plan_delta_layer_with(&shape, &runs, 256, true).unwrap());
    }
    for state in [16, 32, 64, 128, 48] {
        let shape = Mamba2Shape {
            inner: 256,
            state,
            heads: 8,
            groups: 2,
            conv_kernel: 4,
            norm_groups: 2,
            step_min: 0.0,
            eps: 1e-5,
        };
        let runs = [RunSpec {
            first_row: 0,
            rows: 6,
            reset: true,
        }];
        plans.extend(plan_mamba2_mixer(&shape, &runs, 6, 0).unwrap());
    }
    for state in [8, 16, 32, 48] {
        let shape = Mamba1Shape {
            inner: 256,
            state,
            rank: 16,
            conv_kernel: 4,
            bc_rms_eps: Some(1e-5),
        };
        let run = RunSpec {
            first_row: 0,
            rows: 6,
            reset: true,
        };
        plans.push(plan_mamba_conv(0, &run, 256, 4, 512, 0, 0, true, None).unwrap());
        plans.push(plan_mamba1_scan(0, &run, &shape, 512, 256, 1024).unwrap());
        plans.extend(plan_mamba1_bc_norms(&shape, 6).unwrap());
    }
    plans
}

#[test]
fn every_operation_plan_matches_its_compiled_module() {
    let mut seen = BTreeSet::new();
    let plans = all_op_plans();
    assert!(plans.len() > 200, "only {} plans enumerated", plans.len());
    for plan in &plans {
        check_op_plan(plan, &mut seen);
    }
    assert!(seen.len() > 60);
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Projections
// ---------------------------------------------------------------------------------------------------------------------------------------------

fn capability_profiles() -> Vec<VkCapsLike> {
    let mut profiles = Vec::new();
    for (coop_matrix, int_dot) in [(false, false), (false, true), (true, false), (true, true)] {
        for subgroup_size in [32, 64] {
            profiles.push(VkCapsLike {
                coop_matrix,
                int_dot,
                subgroup_size,
                fp16: true,
                max_shared: 49152,
                max_workgroup: 1024,
                vendor: 0x10de,
            });
        }
    }
    profiles.push(VkCapsLike {
        coop_matrix: false,
        int_dot: false,
        subgroup_size: 32,
        fp16: false,
        max_shared: 16384,
        max_workgroup: 1024,
        vendor: 0,
    });
    profiles
}

/// Every format a projection can be planned for: the planar integer formats, bitsandbytes / block-wise FP8 / MLX, and an EXL2 device type.
fn planned_formats() -> Vec<QuantType> {
    let mut all = supported_formats().to_vec();
    all.extend(extended_formats());
    all.push(exl2_device_type(64).unwrap());
    all
}

#[test]
fn every_projection_plan_matches_its_compiled_module() {
    let mut seen = BTreeSet::new();
    let always = |_: &str| true;
    let mut count = 0;
    for quant in planned_formats() {
        for caps in capability_profiles() {
            for batch in [1usize, 2, 5, MATVEC_MAX_BATCH as usize, MATVEC_MAX_BATCH as usize + 1, 33, 256] {
                for (exact, has_bias) in [(false, false), (false, true), (true, false)] {
                    let options = PlanOptions {
                        has_bias,
                        scale: 0.75,
                        row_base: 64,
                        out_stride: Some(1000),
                        exact,
                        force: None,
                        available: Some(&always),
                    };
                    let plan = plan_quantized_linear_with(quant, 70, 1024, batch, &caps, &options).unwrap();
                    count += 1;
                    let name = plan.shader_name.as_str();
                    seen.insert(name.to_owned());
                    let info = info_of(name);
                    assert_eq!(info.bindings, vec![0, 1, 2, 3], "{name}: bindings");
                    assert_eq!(plan.bindings.len(), 4);
                    assert_eq!(info.push_bytes, Some(PUSH_BYTES as u32), "{name}: push-constant size");
                    assert_eq!(plan.push_constants.len(), PUSH_BYTES, "{name}");
                    assert!(!info.push_non_scalar);
                    for c in &plan.spec_constants {
                        assert!(
                            info.spec_ids.contains(&c.id),
                            "{name}: the module has no specialization constant {}",
                            c.id
                        );
                    }
                    assert!(plan.groups.iter().all(|&g| g > 0 && g <= 65535), "{name}: {:?}", plan.groups);
                    if let Some(pp) = &plan.activation_prepass {
                        seen.insert(pp.shader_name.clone());
                        let pinfo = info_of(&pp.shader_name);
                        assert_eq!(
                            pinfo.bindings,
                            (0..pp.bindings.len() as u32).collect::<Vec<_>>(),
                            "{}: bindings",
                            pp.shader_name
                        );
                        assert_eq!(
                            pinfo.push_bytes,
                            Some(pp.push_constants.len() as u32),
                            "{}: push-constant size",
                            pp.shader_name
                        );
                        assert!(
                            pp.groups.iter().all(|&g| g > 0 && g <= 65535),
                            "{}: {:?}",
                            pp.shader_name,
                            pp.groups
                        );
                        assert_eq!(pp.uses_perm, uses_perm(quant));
                    } else {
                        assert!(!uses_perm(quant), "weights with a column permutation always need a prepass");
                    }
                }
            }
        }
    }
    assert!(count > 1000);
    // the weakest device (no optional feature) can run every format with the scalar shaders alone
    let weak = VkCapsLike {
        coop_matrix: false,
        int_dot: false,
        subgroup_size: 32,
        fp16: false,
        max_shared: 16384,
        max_workgroup: 1024,
        vendor: 0,
    };
    for quant in planned_formats() {
        for batch in [1usize, 9, 100] {
            let plan = plan_quantized_linear_with(
                quant,
                128,
                1024,
                batch,
                &weak,
                &PlanOptions {
                    exact: false,
                    ..PlanOptions::default()
                },
            )
            .unwrap();
            assert!(plan.shader_name.ends_with("_s"), "{}", plan.shader_name);
        }
    }
}

/// The three EXL3 shaders: bindings, push-constant sizes and the specialization constant of the tile width agree with the plan constants of the stream.
#[test]
fn exl3_passes_match_their_compiled_modules() {
    use crate::backend::vulkan::weights::exl3::{MM_PUSH_BYTES, POST_PUSH_BYTES, PRE_PUSH_BYTES};
    for (name, bindings, push) in [("exl3_pre", 3u32, PRE_PUSH_BYTES), ("exl3_mm", 3, MM_PUSH_BYTES), ("exl3_post", 4, POST_PUSH_BYTES)] {
        let info = info_of(name);
        assert_eq!(info.bindings, (0..bindings).collect::<Vec<_>>(), "{name}: bindings");
        assert_eq!(info.push_bytes, Some(push), "{name}: push-constant size");
        assert!(!info.push_non_scalar, "{name}");
        assert_eq!(info.spec_ids.iter().copied().collect::<Vec<_>>(), if name == "exl3_mm" { vec![0] } else { vec![] }, "{name}: specialization constants");
    }
}

#[test]
fn float_projection_plans_match_their_compiled_modules() {
    for quant in [QuantType::F32, QuantType::F16, QuantType::BF16] {
        let plan = plan_float_linear(quant, 100, 128, 3, 0, 100, true, 1.0).unwrap();
        let info = info_of(plan.shader);
        assert_eq!(info.bindings, vec![0, 1, 2, 3], "{}", plan.shader);
        assert_eq!(info.push_bytes, Some(plan.push.len() as u32), "{}", plan.shader);
        assert!(!info.push_non_scalar);
        assert!(info.spec_ids.is_empty());
    }
}

/// Every module of the registry is reachable from a plan (or is one of the known exceptions).
#[test]
fn no_shader_is_orphaned() {
    let mut seen = BTreeSet::new();
    for plan in &all_op_plans() {
        seen.insert(plan.shader_name.to_owned());
    }
    let always = |_: &str| true;
    for quant in planned_formats() {
        for caps in capability_profiles() {
            for batch in [1usize, 4, 16, 64] {
                // the `mm_<fmt>_i` kernels serve the batches above the matrix-vector range when the tiled `mmq_<fmt>_i` kernel is not available
                let no_tiled = |n: &str| !n.starts_with("mmq_");
                let availability: [&dyn Fn(&str) -> bool; 2] = [&always, &no_tiled];
                for available in availability {
                    let options = PlanOptions {
                        exact: false,
                        available: Some(available),
                        ..PlanOptions::default()
                    };
                    let plan = plan_quantized_linear_with(quant, 64, 1024, batch, &caps, &options).unwrap();
                    seen.insert(plan.shader_name.clone());
                    if let Some(pp) = plan.activation_prepass {
                        seen.insert(pp.shader_name);
                    }
                }
            }
        }
    }
    for quant in [QuantType::F32, QuantType::F16, QuantType::BF16] {
        seen.insert(plan_float_linear(quant, 8, 8, 1, 0, 8, false, 1.0).unwrap().shader.to_owned());
    }
    // the three passes of an EXL3 product are recorded by the stream, not planned by `plan_quantized_linear`
    seen.extend(["exl3_pre", "exl3_mm", "exl3_post"].map(str::to_owned));
    let registry: BTreeSet<String> = shaders::SHADERS.iter().map(|(n, _)| (*n).to_owned()).collect();
    let orphans: Vec<&String> = registry.iter().filter(|n| !seen.contains(*n)).collect();
    // the buffer-address variants need bufferDeviceAddress, which the core does not enable; scaffold_fill is the scaffold's test shader
    let allowed = |n: &str| n.ends_with("_bda") || n == "scaffold_fill" || n.starts_with("bench_mma_");
    let unexpected: Vec<&&String> = orphans.iter().filter(|n| !allowed(n)).collect();
    assert!(unexpected.is_empty(), "shaders that no plan ever uses: {unexpected:?}");
    let missing: Vec<&String> = seen.iter().filter(|n| !registry.contains(*n)).collect();
    assert!(missing.is_empty(), "plans that name shaders missing from the registry: {missing:?}");
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// The float-weight shader, run on the CPU
// ---------------------------------------------------------------------------------------------------------------------------------------------

/// Lane-by-lane transcription of `lin_float.comp` (workgroup of 128 lanes, one weight row, groups of 8 activation rows).
fn lin_float_mirror(format: QuantType, weights: &[u32], x: &[f32], bias: Option<&[f32]>, p: [u32; 8], groups: [u32; 3]) -> Vec<f32> {
    let [rows, cols, batch, flags, scale_bits, out_stride, row_base, _] = p;
    let scale = f32::from_bits(scale_bits);
    let wval = |i: u32| -> f32 {
        match format {
            QuantType::F32 => f32::from_bits(weights[i as usize]),
            QuantType::F16 => {
                let half = (weights[(i >> 1) as usize] >> ((i & 1) * 16)) & 0xffff;
                half::f16::from_bits(half as u16).to_f32()
            }
            _ => f32::from_bits(((weights[(i >> 1) as usize] >> ((i & 1) * 16)) & 0xffff) << 16),
        }
    };
    let mut out = vec![f32::NAN; (batch * out_stride + row_base + rows) as usize + 8];
    for gz in 0..groups[2] {
        for gy in 0..groups[1] {
            for gx in 0..groups[0] {
                let row = gx + gy * groups[0];
                if row >= rows {
                    continue;
                }
                let b0 = gz * 8;
                let nb = 8.min(batch - b0);
                let mut red = [[0.0f32; 128]; 8];
                for t in 0..128u32 {
                    let mut acc = [0.0f32; 8];
                    let mut k = t;
                    while k < cols {
                        let wv = wval(row * cols + k);
                        for c in 0..nb {
                            acc[c as usize] += wv * x[((b0 + c) * cols + k) as usize];
                        }
                        k += 128;
                    }
                    for (lane, value) in red.iter_mut().zip(acc) {
                        lane[t as usize] = value;
                    }
                }
                let mut s = 64;
                while s > 0 {
                    for t in 0..s {
                        for lane in red.iter_mut() {
                            lane[t] += lane[t + s];
                        }
                    }
                    s >>= 1;
                }
                for t in 0..nb {
                    let mut v = red[t as usize][0] * scale;
                    if flags & 1 != 0 {
                        v += bias.unwrap()[(row_base + row) as usize];
                    }
                    out[((b0 + t) * out_stride + row_base + row) as usize] = v;
                }
            }
        }
    }
    out
}

#[test]
fn float_shader_indexing_matches_a_plain_product() {
    let mut state = 12345u64;
    let mut next = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    for format in [QuantType::F32, QuantType::F16, QuantType::BF16] {
        for (rows, cols, batch, row_base, out_stride) in [
            (5usize, 33usize, 1usize, 0usize, 5usize),
            (3, 200, 19, 4, 11),
            (7, 7, 8, 0, 7),
            (2, 129, 9, 3, 6),
        ] {
            let w: Vec<f32> = (0..rows * cols).map(|_| next()).collect();
            let x: Vec<f32> = (0..batch * cols).map(|_| next()).collect();
            let bias: Vec<f32> = (0..row_base + rows).map(|_| next()).collect();
            // the device image of the weights
            let (words, w_ref): (Vec<u32>, Vec<f32>) = match format {
                QuantType::F32 => (w.iter().map(|v| v.to_bits()).collect(), w.clone()),
                QuantType::F16 => {
                    let halves: Vec<u16> = w.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
                    let refs = halves.iter().map(|&h| half::f16::from_bits(h).to_f32()).collect();
                    (
                        halves
                            .chunks(2)
                            .map(|c| u32::from(c[0]) | (u32::from(*c.get(1).unwrap_or(&0)) << 16))
                            .collect(),
                        refs,
                    )
                }
                _ => {
                    let halves: Vec<u16> = w.iter().map(|&v| (v.to_bits() >> 16) as u16).collect();
                    let refs = halves.iter().map(|&h| f32::from_bits(u32::from(h) << 16)).collect();
                    (
                        halves
                            .chunks(2)
                            .map(|c| u32::from(c[0]) | (u32::from(*c.get(1).unwrap_or(&0)) << 16))
                            .collect(),
                        refs,
                    )
                }
            };
            let plan = plan_float_linear(format, rows, cols, batch, row_base, out_stride, true, 0.5).unwrap();
            let p: [u32; 8] = std::array::from_fn(|i| u32::from_le_bytes(plan.push[i * 4..i * 4 + 4].try_into().unwrap()));
            let got = lin_float_mirror(format, &words, &x, Some(&bias), p, plan.groups);
            for b in 0..batch {
                for r in 0..rows {
                    let want: f64 = (0..cols)
                        .map(|k| f64::from(w_ref[r * cols + k]) * f64::from(x[b * cols + k]))
                        .sum::<f64>()
                        * 0.5
                        + f64::from(bias[row_base + r]);
                    let have = got[b * out_stride + row_base + r];
                    assert!(
                        (f64::from(have) - want).abs() < 1e-4 * (1.0 + want.abs()),
                        "{format:?} {rows}x{cols}x{batch} b{b} r{r}: {have} vs {want}"
                    );
                }
            }
        }
    }
}
