//! Inventory of the Metal kernels the backend can ask for, and its check against the embedded MSL sources.
//!
//! The planners of the shader packages (`weights::plan_quantized_linear`, `ops::*`, `ops::moe_plan`, `ops::recurrent_plan`) name the kernels the
//! stream dispatches. [`inventory`] runs them over a representative set of shapes and collects every kernel name; [`missing_kernels`] compares
//! that list with the kernels the MSL sources define (found with the source scanner of `pipelines/source.rs`: `kernel void NAME(`,
//! `[[kernel]] void NAME(` and `[[host_name("NAME")]]`). A kernel that is planned but not defined would only show up as a pipeline error in the
//! middle of a generation; this check shows it before. The same logic exists as `scripts/check_metal_kernels.py` for hosts without Rust.
//!
//! No Metal objects are used, so the logic runs on any host through `p07-harness`; the unit test that reads the embedded sources
//! (`shaders::SOURCES`) is compiled on Apple targets only.

use crate::backend::metal::ops::attention::{
    self, AttentionTuning, KvLayout, DECODE_E, PREFILL_FRAGMENTS,
};
use crate::backend::metal::ops::embedding::{self, EmbeddingTable, EMBED_MXFP4_PLANAR};
use crate::backend::metal::ops::moe_plan::{self, ExpertAccess, MoeLinear};
use crate::backend::metal::ops::recurrent_plan::{
    self, Mamba1Shape, Mamba2Shape, MixedBuffer, RunSpec,
};
use crate::backend::metal::ops::{self, GluMode, OpPlan};
use crate::backend::metal::pipelines::source::scan_kernel_names;
use crate::backend::metal::weights::{self, MetalCapsLike};
use crate::backend::{AttentionParams, BackendError, DeltaShape, MlaShape, QuantType};
use std::collections::{BTreeMap, BTreeSet};

/// A kernel some planner asks for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedKernel {
    pub name: String,
    /// Where the name comes from (planner function), for the message of a missing kernel.
    pub origin: &'static str,
}

/// Result of [`inventory`]: the kernels, and the planner calls that unexpectedly failed.
#[derive(Clone, Debug, Default)]
pub struct Inventory {
    pub kernels: Vec<PlannedKernel>,
    pub problems: Vec<String>,
}

/// Defined in the sources but deliberately never planned: smoke-test kernels of the scaffold, and the `o_*` kernel templates of `common_ops.metal`
/// users, which are only reachable through their `[[host_name]]` instantiations (the scanner sees the template definition as a kernel too).
pub const KNOWN_UNPLANNED: &[&str] = &[
    "scaffold_fill",
    "o_act_glu",
    "o_attn_decode",
    "o_attn_prefill",
    "o_embed_float",
    "o_embed_quant",
    "o_layer_norm",
    "o_rms_norm",
];

/// Every format the engine knows (the planners are asked about all of them; the unsupported ones answer with an error or `None`).
const ALL_QUANTS: &[QuantType] = &[
    QuantType::F32,
    QuantType::F16,
    QuantType::BF16,
    QuantType::Q4_0,
    QuantType::Q4_1,
    QuantType::Q5_0,
    QuantType::Q5_1,
    QuantType::Q8_0,
    QuantType::Q8_1,
    QuantType::Q2_K,
    QuantType::Q3_K,
    QuantType::Q4_K,
    QuantType::Q5_K,
    QuantType::Q6_K,
    QuantType::Q8_K,
    QuantType::IQ1_S,
    QuantType::IQ1_M,
    QuantType::IQ2_XXS,
    QuantType::IQ2_XS,
    QuantType::IQ2_S,
    QuantType::IQ3_XXS,
    QuantType::IQ3_S,
    QuantType::IQ4_NL,
    QuantType::IQ4_XS,
    QuantType::Fp8E4m3,
    QuantType::Fp8E5m2,
    QuantType::NvFp4,
    QuantType::Mxfp4,
    QuantType::Q4Z32,
    QuantType::Q4Z64,
    QuantType::Q4Z128,
    QuantType::Q4ZP32,
    QuantType::Q4ZP64,
    QuantType::Q4ZP128,
    QuantType::Nvfp4Marlin,
    QuantType::Q4_K_Marlin,
    QuantType::Fp8Marlin,
];

struct Collector {
    found: BTreeMap<String, &'static str>,
    problems: Vec<String>,
}

impl Collector {
    fn add(&mut self, name: &str, origin: &'static str) {
        self.found.entry(name.to_owned()).or_insert(origin);
    }

    fn plan(&mut self, label: &'static str, result: Result<OpPlan, BackendError>) {
        match result {
            Ok(plan) => self.add(&plan.kernel_name, label),
            Err(e) => self.problems.push(format!("{label}: {e}")),
        }
    }

    fn plans(&mut self, label: &'static str, result: Result<Vec<OpPlan>, BackendError>) {
        match result {
            Ok(plans) if plans.is_empty() => self.problems.push(format!("{label}: no plan")),
            Ok(plans) => {
                for plan in &plans {
                    self.add(&plan.kernel_name, label);
                }
            }
            Err(e) => self.problems.push(format!("{label}: {e}")),
        }
    }
}

/// All kernels the planners name, over a representative set of shapes (every format, batch class, head width, state size, ...).
pub fn inventory() -> Inventory {
    let mut c = Collector {
        found: BTreeMap::new(),
        problems: Vec::new(),
    };
    quantized_linear(&mut c);
    basic_ops(&mut c);
    embeddings(&mut c);
    attention_kernels(&mut c);
    moe_kernels(&mut c);
    recurrent_kernels(&mut c);
    Inventory {
        kernels: c
            .found
            .into_iter()
            .map(|(name, origin)| PlannedKernel { name, origin })
            .collect(),
        problems: c.problems,
    }
}

fn quantized_linear(c: &mut Collector) {
    let plain = MetalCapsLike::default();
    let exact = MetalCapsLike {
        exact: true,
        ..plain
    };
    let no_matrix = MetalCapsLike {
        simdgroup_matrix: false,
        ..plain
    };
    for quant in weights::supported_formats() {
        for caps in [&plain, &exact, &no_matrix] {
            for batch in [1usize, 2, 4, 5, 8, 9, 33] {
                // 256 columns: a whole number of blocks of every format
                match weights::plan_quantized_linear(quant, 64, 256, batch, caps) {
                    Ok(plan) => {
                        c.add(&plan.kernel_name, "weights::plan_quantized_linear");
                        if let Some(pre) = &plan.activation_prepass {
                            c.add(pre.kernel_name, "weights::plan_quantized_linear (prepass)");
                        }
                    }
                    Err(e) => c.problems.push(format!(
                        "plan_quantized_linear({quant:?}, batch {batch}): {e}"
                    )),
                }
            }
        }
    }
    for name in weights::all_kernel_names() {
        c.add(&name, "weights::all_kernel_names");
    }
}

fn basic_ops(c: &mut Collector) {
    c.plan("ops::plan_rms_norm", ops::plan_rms_norm(2, 64, 1e-6, true));
    c.plan("ops::plan_rms_norm", ops::plan_rms_norm(2, 63, 1e-6, false));
    c.plan(
        "ops::plan_layer_norm",
        ops::plan_layer_norm(2, 64, 1e-5, true),
    );
    c.plan(
        "ops::plan_layer_norm",
        ops::plan_layer_norm(2, 63, 1e-5, false),
    );
    c.plan(
        "ops::plan_rope",
        ops::plan_rope(&[0, 1], 4, 2, 64, 64, true, 8),
    );
    for mode in [
        GluMode::Swiglu,
        GluMode::Geglu,
        GluMode::SwigluOai {
            alpha: 1.702,
            limit: 7.0,
        },
        GluMode::Silu,
        GluMode::Gelu,
        GluMode::Relu2,
        GluMode::Relu,
        GluMode::GeluErf,
        GluMode::GegluErf,
        GluMode::Reglu,
        GluMode::Relu2Glu,
    ] {
        c.plan("ops::plan_glu", ops::plan_glu(mode, 64));
    }
    c.plan("arch_plan::plan_clamp", ops::arch_plan::plan_clamp(64, 6.0));
    c.plan(
        "arch_plan::plan_softplus_gate_mul",
        ops::arch_plan::plan_softplus_gate_mul(64, 1.0),
    );
    c.plan(
        "arch_plan::plan_mla_latent",
        ops::arch_plan::plan_mla_latent(2, 16, 24),
    );
    c.plan(
        "arch_plan::plan_mla_assemble",
        ops::arch_plan::plan_mla_assemble(
            2,
            &MlaShape { heads: 2, nope: 16, rope: 8, value: 16, kv_lora: 16 },
        ),
    );
    c.plan("ops::plan_residual_add", ops::plan_residual_add(64, 1.0));
    c.plan(
        "ops::plan_sigmoid_gate_mul",
        ops::plan_sigmoid_gate_mul(64),
    );
    c.plan(
        "ops::plan_split_q_gate",
        ops::plan_split_q_gate(2, 4, 64, true),
    );
    c.plan("ops::plan_argmax", ops::plan_argmax(2, 1000));
}

fn embeddings(c: &mut Collector) {
    for &quant in ALL_QUANTS {
        if embedding::embed_kernel(quant).is_none() {
            continue;
        }
        c.plan(
            "embedding::plan_embedding",
            embedding::plan_embedding(&EmbeddingTable::gguf(quant, 4, 256), &[0, 3]),
        );
    }
    c.plan(
        "embedding::plan_embedding (planar MXFP4)",
        embedding::plan_embedding(&EmbeddingTable::mxfp4_planar(4, 256), &[1]),
    );
    c.add(EMBED_MXFP4_PLANAR, "embedding::EMBED_MXFP4_PLANAR");
}

fn kv_layout(kind: i32) -> KvLayout {
    KvLayout {
        kind,
        kv_heads: 1,
        head_dim: 64,
        value_dim: 64,
        block_size: 16,
        pool_blocks: 64,
        key_base: 0,
        value_base: 0,
    }
}

fn attention_kernels(c: &mut Collector) {
    for kind in [1, 2] {
        c.plans(
            "attention::plan_kv_write",
            attention::plan_kv_write(&kv_layout(kind), &[0, 5]),
        );
    }
    let q8 = [false, true];
    for q in q8 {
        for e in DECODE_E {
            match attention::decode_kernel(q, e) {
                Some(name) => c.add(name, "attention::decode_kernel"),
                None => c.problems.push(format!("decode_kernel({q}, {e}) is missing")),
            }
        }
        for frags in PREFILL_FRAGMENTS {
            match attention::prefill_kernel(q, frags) {
                Some(name) => c.add(name, "attention::prefill_kernel"),
                None => c
                    .problems
                    .push(format!("prefill_kernel({q}, {frags:?}) is missing")),
            }
        }
    }
    // end to end: a long single row (decode kernel + merge) and a prompt tile (prompt kernel) for both cache kinds
    let params = AttentionParams {
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 64,
        value_dim: 64,
        stride: 64,
        block_size: 16,
        scale: 0.125,
        softcap: None,
        window: None,
        chunk: None,
        sinks: None,
        alibi_slopes: None,
    };
    let table: Vec<i64> = (0..64).collect();
    for kind in [1, 2] {
        let layout = kv_layout(kind);
        let tuning = AttentionTuning::default();
        c.plans(
            "attention::plan_attention (decode)",
            attention::plan_attention(&layout, &params, &table, &[1000], &tuning),
        );
        let mut tables = Vec::new();
        let mut lengths = Vec::new();
        for r in 0..32i64 {
            tables.extend_from_slice(&table);
            lengths.push(r + 1);
        }
        c.plans(
            "attention::plan_attention (prompt)",
            attention::plan_attention(&layout, &params, &tables, &lengths, &tuning),
        );
    }
}

fn moe_kernels(c: &mut Collector) {
    c.plan("moe_plan::plan_moe_topk", moe_plan::plan_moe_topk(2, 8, 2, true, 1.0));
    c.plan("moe_plan::plan_moe_route", moe_plan::plan_moe_route(2, 8, 2));
    c.plan("moe_plan::plan_moe_combine", moe_plan::plan_moe_combine(2, 2, 64));
    c.plan(
        "moe_plan::plan_gather_rows",
        moe_plan::plan_gather_rows(&[0, 1], None, 4),
    );
    c.plan(
        "moe_plan::plan_scatter_add_rows",
        moe_plan::plan_scatter_add_rows(&[0, 1], Some(&[1.0, 0.5]), 4),
    );
    for &quant in ALL_QUANTS {
        if moe_plan::moe_format(quant).is_none() {
            continue;
        }
        for access in [ExpertAccess::Address, ExpertAccess::Arena] {
            let lin = MoeLinear {
                quant,
                rows: 8,
                cols: 256,
                slots: 4,
                x_div: 2,
                n_experts: 8,
                has_bias: access == ExpertAccess::Arena,
                access,
            };
            c.plan("moe_plan::plan_moe_linear", moe_plan::plan_moe_linear(&lin));
            if access == ExpertAccess::Address {
                c.plans(
                    "moe_plan::plan_moe_expert_gemm",
                    moe_plan::plan_moe_expert_gemm(&lin, 2, 2),
                );
            }
        }
    }
}

fn recurrent_kernels(c: &mut Collector) {
    c.plan(
        "recurrent_plan::plan_split_q_gate",
        recurrent_plan::plan_split_q_gate(2, 4, 64, true),
    );
    c.plan(
        "recurrent_plan::plan_sigmoid_gate_mul",
        recurrent_plan::plan_sigmoid_gate_mul(64),
    );
    let run = RunSpec {
        first_row: 0,
        rows: 3,
        reset: true,
    };
    for hk in [32usize, 64, 96, 128] {
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
        c.plans(
            "recurrent_plan::plan_delta_layer",
            recurrent_plan::plan_delta_layer(&shape, &[run], 0),
        );
    }
    for state in [16usize, 32, 64, 128, 48] {
        let shape = Mamba2Shape {
            inner: 128,
            state,
            heads: 4,
            groups: 1,
            conv_kernel: 4,
            norm_groups: 1,
            step_min: 0.0,
            eps: 1e-6,
        };
        c.plans(
            "recurrent_plan::plan_mamba2_mixer",
            recurrent_plan::plan_mamba2_mixer(&shape, &[run], 3, 0),
        );
        c.plan(
            "recurrent_plan::plan_mamba2_scan",
            recurrent_plan::plan_mamba2_scan(0, &run, &shape, shape.proj_width(), 0, 0, MixedBuffer::Caller),
        );
    }
    for state in [8usize, 16, 32, 24] {
        let shape = Mamba1Shape {
            inner: 64,
            state,
            rank: 4,
            conv_kernel: 4,
            bc_rms_eps: Some(1e-6),
        };
        c.plan(
            "recurrent_plan::plan_mamba1_scan",
            recurrent_plan::plan_mamba1_scan(0, &run, &shape, 128, 64, 0),
        );
        match recurrent_plan::plan_mamba1_bc_norm(&shape, 3) {
            Ok(Some(plan)) => c.add(&plan.kernel_name, "recurrent_plan::plan_mamba1_bc_norm"),
            Ok(None) => c.problems.push("plan_mamba1_bc_norm planned nothing".into()),
            Err(e) => c.problems.push(format!("plan_mamba1_bc_norm: {e}")),
        }
    }
    c.plan(
        "recurrent_plan::plan_mamba_conv",
        recurrent_plan::plan_mamba_conv(0, &run, 256, 4, 512, 0, 0, true, MixedBuffer::Caller),
    );
}

/// The kernel names the MSL `sources` (file name, text) define.
pub fn defined_kernels(sources: &[(&str, &str)]) -> BTreeSet<String> {
    sources
        .iter()
        .flat_map(|(_, text)| scan_kernel_names(text))
        .collect()
}

/// `"name (origin)"` for every planned kernel that no source defines.
pub fn missing_kernels(planned: &[PlannedKernel], defined: &BTreeSet<String>) -> Vec<String> {
    planned
        .iter()
        .filter(|k| !defined.contains(&k.name))
        .map(|k| format!("{} ({})", k.name, k.origin))
        .collect()
}

/// Defined kernels that no planner names: leftovers or forgotten wiring. Template definitions (`..._t`, no `[[host_name]]`) and the
/// [`KNOWN_UNPLANNED`] ones are not reported.
pub fn unplanned_kernels(planned: &[PlannedKernel], defined: &BTreeSet<String>) -> Vec<String> {
    let names: BTreeSet<&str> = planned.iter().map(|k| k.name.as_str()).collect();
    defined
        .iter()
        .filter(|d| !names.contains(d.as_str()) && !d.ends_with("_t") && !KNOWN_UNPLANNED.contains(&d.as_str()))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_planners_run_over_the_whole_inventory() {
        let inv = inventory();
        assert!(inv.problems.is_empty(), "planner calls failed: {:#?}", inv.problems);
        // a sanity floor (the inventory held 197 kernels when this was written): quantized kernels of 23 formats, attention, MoE, DeltaNet, Mamba ...
        assert!(inv.kernels.len() > 180, "only {} kernels", inv.kernels.len());
    }

    #[test]
    fn the_logic_finds_missing_and_unplanned_kernels() {
        let sources = [
            (
                "a.metal",
                "kernel void fllm_one(device float *x [[buffer(0)]]) {}\n\
                 template [[host_name(\"fllm_two\")]] [[kernel]] decltype(t<1>) t<1>;\n",
            ),
            ("b.metal", "[[kernel]] void fllm_three() {}\nkernel void helper_t() {}\n"),
        ];
        let defined = defined_kernels(&sources);
        for name in ["fllm_one", "fllm_two", "fllm_three"] {
            assert!(defined.contains(name), "{name} not found in {defined:?}");
        }
        let planned = vec![
            PlannedKernel {
                name: "fllm_one".into(),
                origin: "x",
            },
            PlannedKernel {
                name: "fllm_four".into(),
                origin: "planner y",
            },
        ];
        assert_eq!(
            missing_kernels(&planned, &defined),
            vec!["fllm_four (planner y)".to_owned()]
        );
        // fllm_two and fllm_three are defined but not planned; helper_t is a template and not reported
        assert_eq!(
            unplanned_kernels(&planned, &defined),
            vec!["fllm_three".to_owned(), "fllm_two".to_owned()]
        );
    }

    /// The names the planners produce must all exist as kernels in the embedded sources, and the sources must not hold kernels nobody plans.
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    #[test]
    fn every_planned_kernel_is_defined_in_the_embedded_sources() {
        let inv = inventory();
        assert!(inv.problems.is_empty(), "{:#?}", inv.problems);
        let defined = defined_kernels(crate::backend::metal::shaders::SOURCES);
        let missing = missing_kernels(&inv.kernels, &defined);
        assert!(missing.is_empty(), "planned but not defined: {missing:#?}");
        let unplanned = unplanned_kernels(&inv.kernels, &defined);
        assert!(unplanned.is_empty(), "defined but never planned: {unplanned:#?}");
    }
}
