//! Unit tests of the plans (package p06b): grid and constant arithmetic, and agreement of every plan with the MSL source it drives (kernel name,
//! buffer indices, parameter struct fields and size), so the Rust side cannot drift from the kernels.

use super::attention::*;
use super::embedding::*;
use super::*;
use crate::backend::{AttentionParams, QuantType};

// ---------------------------------------------------------------------------------------------------------------------------------------------
// MSL sources
// ---------------------------------------------------------------------------------------------------------------------------------------------

const COMMON: &str = include_str!("../../../../metal/common_ops.metal");
pub(super) const KERNEL_FILES: &[(&str, &str)] = &[
    (
        "norm_rms.metal",
        include_str!("../../../../metal/norm_rms.metal"),
    ),
    (
        "norm_layer.metal",
        include_str!("../../../../metal/norm_layer.metal"),
    ),
    (
        "rope_apply.metal",
        include_str!("../../../../metal/rope_apply.metal"),
    ),
    (
        "act_glu.metal",
        include_str!("../../../../metal/act_glu.metal"),
    ),
    (
        "act_sigmoid_mul.metal",
        include_str!("../../../../metal/act_sigmoid_mul.metal"),
    ),
    (
        "misc_residual.metal",
        include_str!("../../../../metal/misc_residual.metal"),
    ),
    (
        "misc_split_q_gate.metal",
        include_str!("../../../../metal/misc_split_q_gate.metal"),
    ),
    (
        "misc_argmax.metal",
        include_str!("../../../../metal/misc_argmax.metal"),
    ),
    (
        "kv_write.metal",
        include_str!("../../../../metal/kv_write.metal"),
    ),
    (
        "embed_float.metal",
        include_str!("../../../../metal/embed_float.metal"),
    ),
    (
        "embed_quant.metal",
        include_str!("../../../../metal/embed_quant.metal"),
    ),
    (
        "attn_decode.metal",
        include_str!("../../../../metal/attn_decode.metal"),
    ),
    (
        "attn_prefill.metal",
        include_str!("../../../../metal/attn_prefill.metal"),
    ),
];

/// Source text without `//` comments (line structure kept).
fn strip_comments(src: &str) -> String {
    src.lines()
        .map(|l| match l.find("//") {
            Some(i) => &l[..i],
            None => l,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Names of all `[[host_name("...")]]` entries and plain `kernel void NAME(` definitions of a source.
fn kernel_names(src: &str) -> Vec<String> {
    let text = strip_comments(src);
    let mut names = Vec::new();
    let mut rest = text.as_str();
    while let Some(i) = rest.find("host_name(\"") {
        rest = &rest[i + "host_name(\"".len()..];
        let end = rest.find('"').expect("closing quote");
        names.push(rest[..end].to_string());
    }
    for line in text.lines() {
        if let Some(i) = line.find("kernel void ") {
            let after = &line[i + "kernel void ".len()..];
            if let Some(j) = after.find('(') {
                names.push(after[..j].trim().to_string());
            }
        }
    }
    names
}

/// The signature text (between the parentheses) of the kernel function that serves `kernel_name`.
fn signature(kernel_name: &str) -> Option<String> {
    for (_, src) in KERNEL_FILES {
        let text = strip_comments(src);
        let key = format!("host_name(\"{kernel_name}\")");
        let func = if let Some(i) = text.find(&key) {
            // `... decltype(FUNC<args>) FUNC<args>;`
            let after = &text[i..];
            let d = after.find("decltype(")?;
            let tail = &after[d + "decltype(".len()..];
            let end = tail.find(['<', ')'])?;
            tail[..end].trim().to_string()
        } else if text.contains(&format!("kernel void {kernel_name}(")) {
            kernel_name.to_string()
        } else {
            continue;
        };
        let def = format!("void {func}(");
        let at = text.find(&def)?;
        let tail = &text[at + def.len()..];
        let mut depth = 1;
        for (k, ch) in tail.char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(tail[..k].to_string());
                    }
                }
                _ => {}
            }
        }
    }
    None
}

/// `[[buffer(N)]]` indices of a signature in order of appearance.
fn buffer_indices(sig: &str) -> Vec<u32> {
    let mut out = Vec::new();
    let mut rest = sig;
    while let Some(i) = rest.find("[[buffer(") {
        rest = &rest[i + "[[buffer(".len()..];
        let end = rest.find(')').unwrap();
        out.push(rest[..end].trim().parse().expect("buffer index"));
    }
    out
}

/// Name of the parameter struct: the type of the `constant <T> &p` parameter.
fn params_struct(sig: &str) -> String {
    let i = sig.find("constant ").expect("constant parameter");
    let tail = &sig[i + "constant ".len()..];
    tail[..tail.find('&').unwrap()].trim().to_string()
}

/// Members `(type, name)` of a struct of `common_ops.metal`.
fn struct_fields(name: &str) -> Vec<(String, String)> {
    let text = strip_comments(COMMON);
    let head = format!("struct {name} {{");
    let at = text
        .find(&head)
        .unwrap_or_else(|| panic!("struct {name} not found"));
    let body = &text[at + head.len()..];
    let body = &body[..body.find("};").unwrap()];
    body.split(';')
        .map(str::trim)
        .filter(|m| !m.is_empty())
        .map(|m| {
            let mut parts = m.split_whitespace();
            let ty = parts.next().unwrap().to_string();
            let nm = parts.next().unwrap().to_string();
            (ty, nm)
        })
        .collect()
}

/// Size of an MSL struct of 4-byte scalars and `ulong`s (natural alignment).
fn struct_size(fields: &[(String, String)]) -> usize {
    let mut off = 0usize;
    let mut align = 4usize;
    for (ty, _) in fields {
        let sz = match ty.as_str() {
            "uint" | "int" | "float" => 4,
            "ulong" => 8,
            other => panic!("unexpected member type {other}"),
        };
        align = align.max(sz);
        off = off.next_multiple_of(sz) + sz;
    }
    off.next_multiple_of(align)
}

thread_local! {
    /// Field names of the parameter struct of the plans built on this thread, by kernel name (recorded by `ops::plan`).
    static FIELDS: std::cell::RefCell<std::collections::HashMap<String, Vec<&'static str>>> = std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Called by `ops::plan`: the names of the parameter fields a plan of `kernel` wrote.
pub(super) fn remember_fields(kernel: &str, names: &[&'static str]) {
    FIELDS.with(|f| {
        f.borrow_mut().insert(kernel.to_string(), names.to_vec());
    });
}

fn scalar_size(ty: &str) -> usize {
    match ty {
        "uint" | "int" | "float" => 4,
        "ulong" => 8,
        other => panic!("unexpected member type {other}"),
    }
}

/// Check one plan against the source of its kernel.
pub(super) fn check_plan(plan: &OpPlan) {
    let name = plan.kernel_name.as_str();
    let sig = signature(name).unwrap_or_else(|| panic!("kernel {name} not found in the sources"));
    // buffer 0 = constants struct, buffers 1..=n = the slots
    let mut got = buffer_indices(&sig);
    got.sort_unstable();
    let expect: Vec<u32> = (0..=plan.slots.len() as u32).collect();
    assert_eq!(got, expect, "{name}: buffer indices of the signature");
    for (i, b) in plan.slots.iter().enumerate() {
        assert_eq!(b.index as usize, i + 1, "{name}: slot {} index", b.name);
    }
    let st = params_struct(&sig);
    assert!(
        strip_ws(&sig).contains(&format!(
            "constant{st}&p[[buffer({CONSTANTS_BUFFER_INDEX})]]"
        )),
        "{name}: the constants struct must be bound at buffer {CONSTANTS_BUFFER_INDEX}"
    );
    let fields = struct_fields(&st);
    let names: Vec<&str> = fields.iter().map(|(_, n)| n.as_str()).collect();
    let plan_names = FIELDS
        .with(|f| f.borrow().get(name).cloned())
        .unwrap_or_else(|| panic!("{name}: no recorded fields"));
    assert_eq!(names, plan_names, "{name}: fields of {st}");
    assert_eq!(
        struct_size(&fields),
        plan.constants.len(),
        "{name}: bytes of {st}"
    );
    assert!(plan.constants.len() <= MAX_PARAM_BYTES);
    let threads: u64 = plan
        .threads_per_group
        .iter()
        .map(|&t| u64::from(t))
        .product();
    assert!(
        (1..=1024).contains(&threads),
        "{name}: {threads} threads per group"
    );
}

fn strip_ws(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

/// Offset and size of a field of the plan's constants struct, from the MSL definition.
fn field_at(plan: &OpPlan, field: &str) -> (usize, usize) {
    let sig = signature(&plan.kernel_name).expect("kernel");
    let mut off = 0usize;
    for (ty, n) in struct_fields(&params_struct(&sig)) {
        let sz = scalar_size(&ty);
        off = off.next_multiple_of(sz);
        if n == field {
            return (off, sz);
        }
        off += sz;
    }
    panic!("no field {field}");
}

fn word(plan: &OpPlan, field: &str) -> u32 {
    let (off, sz) = field_at(plan, field);
    assert_eq!(sz, 4, "{field} is not a 32-bit field");
    u32::from_le_bytes(plan.constants[off..off + 4].try_into().unwrap())
}

fn wide(plan: &OpPlan, field: &str) -> u64 {
    let (off, sz) = field_at(plan, field);
    assert_eq!(sz, 8, "{field} is not a 64-bit field");
    u64::from_le_bytes(plan.constants[off..off + 8].try_into().unwrap())
}
// ---------------------------------------------------------------------------------------------------------------------------------------------
// Tables against the sources
// ---------------------------------------------------------------------------------------------------------------------------------------------

fn host_names(file: &str) -> Vec<String> {
    let (_, src) = KERNEL_FILES.iter().find(|(n, _)| *n == file).unwrap();
    let mut v = kernel_names(src);
    v.sort();
    v
}

#[test]
fn decode_table_matches_the_source() {
    let mut from_rust: Vec<String> = Vec::new();
    for q8 in [false, true] {
        for e in DECODE_E {
            from_rust.push(decode_kernel(q8, e).unwrap().to_string());
        }
    }
    from_rust.push("fllm_attn_merge_f32".into());
    from_rust.sort();
    assert_eq!(from_rust, host_names("attn_decode.metal"));
}

#[test]
fn prefill_table_matches_the_source() {
    let mut from_rust: Vec<String> = Vec::new();
    for q8 in [false, true] {
        for f in PREFILL_FRAGMENTS {
            from_rust.push(prefill_kernel(q8, f).unwrap().to_string());
        }
    }
    from_rust.sort();
    assert_eq!(from_rust, host_names("attn_prefill.metal"));
}

#[test]
fn embedding_table_matches_the_sources() {
    let all = [
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
    ];
    let mut from_rust: Vec<String> = all
        .iter()
        .map(|&q| embed_kernel(q).unwrap().0.to_string())
        .collect();
    from_rust.push(EMBED_MXFP4_PLANAR.to_string());
    from_rust.sort();
    let mut from_src = host_names("embed_float.metal");
    from_src.extend(host_names("embed_quant.metal"));
    from_src.sort();
    assert_eq!(from_rust, from_src);
    // block geometry equals the engine's QuantType and the table `o_fmt_bytes` of common_ops.metal
    let text = strip_comments(COMMON);
    let fmt_bytes_src = &text[text.find("constexpr uint o_fmt_bytes").unwrap()..];
    let fmt_bytes_src = &fmt_bytes_src[..fmt_bytes_src.find("\n}").unwrap()];
    for &q in &all[3..] {
        let (_, qk, bytes) = embed_kernel(q).unwrap();
        assert_eq!(qk, q.block_size(), "{q:?} block size");
        assert_eq!(bytes, q.block_bytes(), "{q:?} block bytes");
        let tag = format!("{q:?}").to_uppercase();
        let needle = format!("fmt == O_FMT_{tag} ? ");
        match fmt_bytes_src.find(&needle) {
            Some(i) => {
                let v: String = fmt_bytes_src[i + needle.len()..]
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect();
                assert_eq!(
                    v.parse::<usize>().unwrap(),
                    bytes,
                    "{q:?} bytes in o_fmt_bytes"
                );
            }
            None => assert_eq!(
                q,
                QuantType::Mxfp4,
                "{q:?} is the fall-through entry (17 bytes)"
            ),
        }
    }
    assert!(fmt_bytes_src.contains(": 17u;"));
}

#[test]
fn every_source_kernel_is_reachable_from_a_plan_table() {
    // all kernel names of the sources, minus the template definitions that have no host_name
    let mut all: Vec<String> = KERNEL_FILES
        .iter()
        .flat_map(|(_, s)| kernel_names(s))
        .collect();
    all.sort();
    let mut planned: Vec<String> = vec![
        "fllm_rms_norm_f32",
        "fllm_rms_norm_f32x4",
        "fllm_layer_norm_f32",
        "fllm_layer_norm_f32x4",
        "fllm_rope_f32",
        "fllm_act_swiglu",
        "fllm_act_geglu",
        "fllm_act_swiglu_oai",
        "fllm_act_silu",
        "fllm_act_gelu",
        "fllm_act_relu2",
        "fllm_act_relu",
        "fllm_act_gelu_erf",
        "fllm_act_geglu_erf",
        "fllm_act_reglu",
        "fllm_act_relu2glu",
        "fllm_sigmoid_mul_f32",
        "fllm_residual_add_f32",
        "fllm_split_q_gate_f32",
        "fllm_argmax_f32",
        "fllm_kv_write_f16",
        "fllm_kv_write_q8",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    planned.extend(host_names("attn_decode.metal"));
    planned.extend(host_names("attn_prefill.metal"));
    planned.extend(host_names("embed_float.metal"));
    planned.extend(host_names("embed_quant.metal"));
    planned.sort();
    planned.dedup();
    all.dedup();
    assert_eq!(all, planned);
    for n in &all {
        assert!(
            n.starts_with("fllm_"),
            "kernel {n} violates the naming convention"
        );
    }
}

#[test]
fn common_ops_symbols_are_prefixed() {
    let text = strip_comments(COMMON);
    for line in text.lines() {
        let l = line.trim_start();
        let name = if let Some(r) = l.strip_prefix("struct ") {
            r.split_whitespace().next()
        } else if let Some(r) = l.strip_prefix("inline ") {
            r.split_whitespace()
                .nth(1)
                .map(|s| s.split('(').next().unwrap())
        } else if let Some(r) = l.strip_prefix("constexpr ") {
            r.split_whitespace()
                .nth(1)
                .map(|s| s.split('(').next().unwrap())
        } else if let Some(r) = l.strip_prefix("constant ") {
            r.split_whitespace()
                .nth(1)
                .map(|s| s.split(['[', ' ', '=']).next().unwrap())
        } else if let Some(r) = l.strip_prefix("#define ") {
            r.split_whitespace()
                .next()
                .map(|s| s.split('(').next().unwrap())
        } else {
            None
        };
        if let Some(n) = name {
            if matches!(n, "FLLM_OPS_COMMON_METAL" | "FLLM_HAS_SIMDGROUP_MATRIX") {
                continue;
            }
            assert!(
                n.starts_with("o_") || n.starts_with("O_"),
                "symbol {n} of common_ops.metal lacks the o_ prefix"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Parameter bytes
// ---------------------------------------------------------------------------------------------------------------------------------------------

#[test]
fn params_builder_layout() {
    let (bytes, names) = Params::new()
        .u64("a", 1 << 40)
        .u("b", 7)
        .f("c", 1.5)
        .finish()
        .unwrap();
    assert_eq!(names, ["a", "b", "c"]);
    assert_eq!(bytes.len(), 16);
    assert_eq!(u64::from_le_bytes(bytes[..8].try_into().unwrap()), 1 << 40);
    assert_eq!(f32::from_le_bytes(bytes[12..16].try_into().unwrap()), 1.5);
    // a 64-bit field pads the struct to a multiple of 8 bytes
    let (bytes, names) = Params::new().u64("a", 1).u("b", 7).finish().unwrap();
    assert_eq!(names, ["a", "b", "_pad"]);
    assert_eq!(bytes.len(), 16);
    // no 64-bit field: no padding
    let (bytes, names) = Params::new()
        .u("a", 1)
        .u("b", 2)
        .u("c", 3)
        .finish()
        .unwrap();
    assert_eq!((bytes.len(), names.len()), (12, 3));
    // a 64-bit field at an unaligned offset is an error
    assert!(Params::new().u("a", 1).u64("b", 2).finish().is_err());
    // values that do not fit 32 bits are errors, not truncations
    assert!(Params::new().u("a", 1 << 33).finish().is_err());
    assert!(Params::new().i("a", 1 << 33).finish().is_err());
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Simple operations
// ---------------------------------------------------------------------------------------------------------------------------------------------

#[test]
fn rms_norm_plans() {
    for cols in [1usize, 3, 4, 64, 100, 128, 2048, 4096, 8192, 8193] {
        for has_weight in [true, false] {
            let p = plan_rms_norm(5, cols, 1e-6, has_weight).unwrap();
            check_plan(&p);
            let vec4 = cols.is_multiple_of(4);
            assert_eq!(
                p.kernel_name,
                if vec4 {
                    "fllm_rms_norm_f32x4"
                } else {
                    "fllm_rms_norm_f32"
                }
            );
            assert_eq!(p.threadgroups, [5, 1, 1]);
            let t = p.threads_per_group[0] as usize;
            assert!(
                t.is_multiple_of(32) && (32..=512).contains(&t),
                "cols {cols}: {t} threads"
            );
            // every thread loops at most ceil(items / t) times; the group is not larger than needed (beyond the 32 minimum)
            let items = if vec4 { cols / 4 } else { cols };
            assert!(t < items + 32 || t == 32 || t == 512);
            assert_eq!(
                p.slot("weight").unwrap().source,
                if has_weight {
                    SlotSource::Buffer
                } else {
                    SlotSource::Dummy
                }
            );
            assert_eq!(word(&p, "flags"), u32::from(has_weight));
            assert_eq!(
                f32::from_le_bytes(p.constants[24..28].try_into().unwrap()),
                1e-6
            );
        }
    }
    assert!(plan_rms_norm(0, 8, 1e-6, true).is_err());
    assert!(plan_rms_norm(1, 0, 1e-6, true).is_err());
    // strided slice of a wider row
    let layout = RowLayout {
        x_stride: 300,
        x_off: 7,
        o_stride: 64,
        o_off: 0,
    };
    let p = plan_rms_norm_strided(2, 64, 1e-5, true, layout).unwrap();
    check_plan(&p);
    assert_eq!(
        (
            word(&p, "x_stride"),
            word(&p, "x_off"),
            word(&p, "o_stride"),
            word(&p, "o_off")
        ),
        (300, 7, 64, 0)
    );
}

#[test]
fn layer_norm_plans() {
    for cols in [7usize, 8, 1000, 1024] {
        for has_bias in [true, false] {
            let p = plan_layer_norm(3, cols, 1e-5, has_bias).unwrap();
            check_plan(&p);
            assert_eq!(
                p.slot("bias").unwrap().source,
                if has_bias {
                    SlotSource::Buffer
                } else {
                    SlotSource::Dummy
                }
            );
            assert_eq!(p.buffer_count(), 4);
        }
    }
}

#[test]
fn rope_plans() {
    let pos: Vec<usize> = (10..22).collect();
    for (dim, rope_dim, neox) in [
        (128usize, 128usize, true),
        (128, 64, false),
        (64, 64, true),
        (80, 16, false),
    ] {
        let p = plan_rope(&pos, 32, 8, dim, rope_dim, neox, 4096).unwrap();
        check_plan(&p);
        let half = rope_dim / 2;
        let tx = p.threads_per_group[0] as usize;
        assert!(tx.is_multiple_of(32) && tx >= half.min(256));
        assert_eq!(p.threadgroups, [half.div_ceil(tx) as u32, 40, 12]);
        assert_eq!(word(&p, "half_dim") as usize, half);
        assert_eq!(word(&p, "neox"), u32::from(neox));
        match &p.slot("positions").unwrap().source {
            SlotSource::Host(b) => assert_eq!(b.len(), pos.len() * 4),
            other => panic!("{other:?}"),
        }
    }
    assert!(plan_rope(&pos, 4, 4, 64, 65, true, 100).is_err()); // odd rope width
    assert!(plan_rope(&pos, 4, 4, 64, 128, true, 100).is_err()); // wider than the head
    assert!(plan_rope(&pos, 4, 4, 64, 64, true, 20).is_err()); // table too short
    assert!(plan_rope(&[], 4, 4, 64, 64, true, 20).is_err());
}

#[test]
fn activation_plans() {
    let modes = [
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
    ];
    for mode in modes {
        let p = plan_glu(mode, 11008).unwrap();
        check_plan(&p);
        assert_eq!(
            p.slot("up").unwrap().source,
            if mode.gated() {
                SlotSource::Buffer
            } else {
                SlotSource::Dummy
            }
        );
        assert_eq!(p.threadgroups[1], 1);
        let p = plan_glu_strided(
            mode,
            17,
            300,
            GluLayout {
                gate_stride: 1200,
                gate_off: 0,
                up_stride: 1200,
                up_off: 600,
                out_stride: 300,
            },
        )
        .unwrap();
        check_plan(&p);
        assert_eq!(p.threadgroups, [2, 17, 1]);
        assert_eq!(p.threads_per_group, [256, 1, 1]);
        assert_eq!(word(&p, "up_off"), 600);
        if let GluMode::SwigluOai { alpha, limit } = mode {
            let a = f32::from_le_bytes(p.constants[28..32].try_into().unwrap());
            let l = f32::from_le_bytes(p.constants[32..36].try_into().unwrap());
            assert_eq!((a, l), (alpha, limit));
        }
    }
    assert!(plan_glu(GluMode::Swiglu, 0).is_err());
}

#[test]
fn elementwise_plans() {
    let p = plan_residual_add(4096 * 7 + 1, 0.5).unwrap();
    check_plan(&p);
    assert_eq!(
        p.threadgroups,
        [(4096 * 7 + 1usize).div_ceil(256) as u32, 1, 1]
    );
    assert!(p.total_threads() > 4096 * 7);
    assert_eq!(
        f32::from_le_bytes(p.constants[4..8].try_into().unwrap()),
        0.5
    );
    let p = plan_sigmoid_gate_mul(1000).unwrap();
    check_plan(&p);
    for per_head in [true, false] {
        let p = plan_split_q_gate(9, 16, 128, per_head).unwrap();
        check_plan(&p);
        assert_eq!(p.threadgroups, [1, 16, 9]);
        assert_eq!(word(&p, "per_head"), u32::from(per_head));
    }
    let p = plan_split_q_gate(2, 3, 300, true).unwrap();
    assert_eq!((p.threads_per_group[0], p.threadgroups[0]), (256, 2));
    assert!(plan_residual_add(0, 1.0).is_err());
    assert!(plan_sigmoid_gate_mul(0).is_err());
    assert!(plan_split_q_gate(0, 1, 1, true).is_err());
}

#[test]
fn argmax_plans() {
    for vocab in [5usize, 32, 100, 4096, 151_936] {
        let p = plan_argmax(3, vocab).unwrap();
        check_plan(&p);
        let t = p.threads_per_group[0];
        assert!(
            t.is_power_of_two() && (32..=512).contains(&t),
            "vocab {vocab}: {t} threads"
        );
        assert_eq!(p.threadgroups, [3, 1, 1]);
    }
    assert!(plan_argmax(0, 10).is_err());
}

#[test]
fn argmax_threads_are_a_power_of_two() {
    // the kernel halves the active threads; a non power of two would lose lanes
    assert_eq!(plan_argmax(1, 33).unwrap().threads_per_group[0], 64);
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------------------------------------------------------------------------

#[test]
fn embedding_planar_mxfp4_plan() {
    let table = EmbeddingTable::mxfp4_planar(1000, 2048);
    let p = plan_embedding(&table, &[1, 999]).unwrap();
    check_plan(&p);
    assert_eq!(p.kernel_name, EMBED_MXFP4_PLANAR);
    // plane 1 starts behind the 16 code bytes of every block
    assert_eq!(wide(&p, "plane1") as usize, 1000 * 2048 / 32 * 16);
    assert_eq!(word(&p, "row_bytes"), 2048 / 32 * 16);
    // only MXFP4 has a planar layout in the Metal backend
    let bad = EmbeddingTable {
        layout: TableLayout::Planar,
        ..EmbeddingTable::gguf(QuantType::Q4_K, 10, 256)
    };
    assert!(plan_embedding(&bad, &[1]).is_err());
}

#[test]
fn embedding_plans() {
    let all = [
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
    ];
    for q in all {
        let table = EmbeddingTable::gguf(q, 1000, 2048);
        let p = plan_embedding(&table, &[0, 5, 999]).unwrap();
        check_plan(&p);
        assert_eq!(p.threadgroups, [8, 3, 1]);
        assert_eq!(word(&p, "row_bytes") as usize, table.row_bytes().unwrap());
        assert_eq!(p.kernel_name, embed_kernel(q).unwrap().0);
    }
    let q = EmbeddingTable::gguf(QuantType::Q4_K, 10, 2048);
    assert_eq!(q.row_bytes(), Some(2048 / 256 * 144));
    // not a whole number of blocks, out-of-range ids, unsupported formats, no ids
    assert!(plan_embedding(&EmbeddingTable::gguf(QuantType::Q6_K, 10, 300), &[1]).is_err());
    assert!(plan_embedding(&q, &[10]).is_err());
    assert!(plan_embedding(&q, &[-1]).is_err());
    assert!(plan_embedding(&q, &[]).is_err());
    for u in [
        QuantType::NvFp4,
        QuantType::IQ4_XS,
        QuantType::Q8_K,
        QuantType::Q4Z32,
        QuantType::Fp8E4m3,
        QuantType::IQ2_S,
    ] {
        let e = plan_embedding(&EmbeddingTable::gguf(u, 4, 256), &[1]).unwrap_err();
        assert!(matches!(e, BackendError::UnsupportedQuant(_)), "{u:?}");
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// KV write and attention
// ---------------------------------------------------------------------------------------------------------------------------------------------

fn layout(
    kind: i32,
    kv_heads: usize,
    dk: usize,
    dv: usize,
    block: usize,
    blocks: usize,
    layer: usize,
) -> KvLayout {
    let (wk, wv) =
        crate::backend::DeviceKvCache::geometry(4, blocks, block, kv_heads, dk, dv, kind).unwrap();
    KvLayout {
        kind,
        kv_heads,
        head_dim: dk,
        value_dim: dv,
        block_size: block,
        pool_blocks: blocks,
        key_base: layer * wk,
        value_base: layer * wv,
    }
}

#[test]
fn kv_write_plans() {
    for (kind, dk, dv) in [
        (1, 128usize, 128usize),
        (1, 80, 64),
        (2, 128, 128),
        (2, 80, 80),
        (2, 192, 128),
        (2, 64, 64),
    ] {
        let kv = layout(kind, 4, dk, dv, 16, 8, 2);
        let slots = [3usize, 100, 17];
        let plans = plan_kv_write(&kv, &slots).unwrap();
        assert_eq!(plans.len(), 2);
        for (i, p) in plans.iter().enumerate() {
            check_plan(p);
            let dim = if i == 0 { dk } else { dv };
            assert_eq!(
                p.kernel_name,
                if kind == 1 {
                    "fllm_kv_write_f16"
                } else {
                    "fllm_kv_write_q8"
                }
            );
            assert_eq!(word(p, "kv") as usize, 4 * dim);
            let base = wide(p, "base") as usize;
            assert_eq!(
                base,
                2 * if i == 0 {
                    kv.key_words()
                } else {
                    kv.value_words()
                }
            );
            if kind == 2 {
                let group = word(p, "group") as usize;
                assert_eq!(group, if dim.is_multiple_of(32) { 32 } else { dim });
                assert_eq!(p.threadgroups, [(4 * dim / group) as u32, 3, 1]);
                assert_eq!(p.threads_per_group, [32, 1, 1]);
                assert_eq!(
                    wide(p, "scale_base") as usize,
                    (kv.slots() * 4 * dim).div_ceil(2)
                );
            } else {
                assert_eq!(wide(p, "scale_base"), 0);
                assert_eq!(p.threadgroups[1], 3);
                assert!(p.total_threads() >= (3 * 4 * dim) as u64);
            }
        }
    }
    let kv = layout(1, 2, 64, 64, 16, 4, 0);
    assert!(plan_kv_write(&kv, &[]).is_err());
    assert!(plan_kv_write(&kv, &[64]).is_err()); // slot == pool slots
    let mut bad = kv;
    bad.kind = 3;
    assert!(plan_kv_write(&bad, &[0]).is_err());
}

fn attn_params(
    heads: usize,
    kv_heads: usize,
    dk: usize,
    dv: usize,
    block: usize,
    stride: usize,
) -> AttentionParams {
    AttentionParams {
        num_heads: heads,
        num_kv_heads: kv_heads,
        head_dim: dk,
        value_dim: dv,
        stride,
        block_size: block,
        scale: (dk as f32).powf(-0.5),
        softcap: None,
        window: None,
        chunk: None,
        sinks: None,
        alibi_slopes: None,
    }
}

/// Block tables of `lengths.len()` rows that all read the blocks 0.. of one sequence (as a prefill does).
fn shared_tables(lengths: &[i64], block: usize, stride: usize) -> Vec<i64> {
    let mut t = Vec::new();
    for _ in lengths {
        t.extend((0..stride as i64).map(|b| b % 4));
    }
    let _ = block;
    t
}

#[test]
fn decode_plan_splits_the_key_range() {
    let kv = layout(1, 8, 128, 128, 16, 4096 / 16, 1);
    let params = attn_params(32, 8, 128, 128, 16, 256);
    let lengths = [4096i64];
    let tables = shared_tables(&lengths, 16, 256);
    let tuning = AttentionTuning {
        use_prefill_kernel: false,
        ..AttentionTuning::default()
    };
    let plans = plan_attention(&kv, &params, &tables, &lengths, &tuning).unwrap();
    assert_eq!(plans.len(), 2, "decode + merge");
    for p in &plans {
        check_plan(p);
    }
    let d = &plans[0];
    assert_eq!(d.kernel_name, "fllm_attn_decode_f16_e4");
    let nsplit = word(d, "nsplit") as usize;
    assert!(nsplit > 1 && nsplit <= 64);
    assert_eq!(d.threadgroups, [nsplit as u32, 32, 1]);
    assert_eq!(d.threads_per_group, [DECODE_THREADS, 1, 1]);
    assert!(32 * nsplit >= 256 && 4096 / nsplit >= 128);
    let m = &plans[1];
    assert_eq!(m.kernel_name, "fllm_attn_merge_f32");
    assert_eq!(m.threadgroups, [32, 1, 1]);
    assert_eq!(word(m, "nsplit") as usize, nsplit);
    // scratch for the partial results is sized for (m, l, dv accumulators) per record and shared between the two plans
    let need = (32 * nsplit * (2 + 128)) * 4;
    match (
        &d.slot("part").unwrap().source,
        &m.slot("part").unwrap().source,
    ) {
        (SlotSource::Scratch { id: 0, bytes: a }, SlotSource::Scratch { id: 0, bytes: b }) => {
            assert!(*a == need && *b == need)
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(wide(d, "k_base") as usize, kv.key_base);
    assert_eq!(wide(d, "v_base") as usize, kv.value_base);

    // short context: no split, no merge
    let lengths = [20i64];
    let tables = shared_tables(&lengths, 16, 256);
    let plans = plan_attention(&kv, &params, &tables, &lengths, &tuning).unwrap();
    assert_eq!(plans.len(), 1);
    assert_eq!(word(&plans[0], "nsplit"), 1);
}

#[test]
fn decode_kernel_choice_follows_the_head_width() {
    for (dk, dv, e) in [
        (64usize, 64usize, 2usize),
        (80, 80, 3),
        (96, 96, 3),
        (128, 128, 4),
        (192, 128, 6),
        (256, 256, 8),
        (576, 512, 18),
        (32, 32, 1),
        (160, 160, 5),
        (100, 100, 4),
    ] {
        assert_eq!(decode_e(dk, dv), Some(e), "{dk}/{dv}");
        for kind in [1, 2] {
            let kv = layout(kind, 2, dk, dv, 16, 8, 0);
            let params = attn_params(4, 2, dk, dv, 16, 8);
            let tuning = AttentionTuning {
                use_prefill_kernel: false,
                ..AttentionTuning::default()
            };
            let plans =
                plan_attention(&kv, &params, &shared_tables(&[40], 16, 8), &[40], &tuning).unwrap();
            check_plan(&plans[0]);
            assert_eq!(plans[0].kernel_name, decode_kernel(kind == 2, e).unwrap());
            if kind == 2 {
                assert_eq!(
                    word(&plans[0], "k_gshift"),
                    if dk.is_multiple_of(32) { 5 } else { 0 }
                );
                assert_eq!(
                    word(&plans[0], "v_gshift"),
                    if dv.is_multiple_of(32) { 5 } else { 0 }
                );
                assert_eq!(
                    wide(&plans[0], "k_scale") as usize,
                    (kv.slots() * 2 * dk).div_ceil(2)
                );
            }
        }
    }
    assert_eq!(decode_e(577, 64), None);
    let kv = layout(1, 1, 640, 64, 16, 4, 0);
    let params = attn_params(1, 1, 640, 64, 16, 4);
    assert!(plan_attention(
        &kv,
        &params,
        &shared_tables(&[8], 16, 4),
        &[8],
        &AttentionTuning::default()
    )
    .is_err());
}

#[test]
fn prefill_segments() {
    let kv = layout(1, 8, 128, 128, 16, 64, 0);
    let params = attn_params(32, 8, 128, 128, 16, 64);
    // 72 prompt rows of one sequence (lengths 1..=72, one shared block table): two full tiles and a partial one (8 rows) in ONE dispatch
    let lengths: Vec<i64> = (1..=72).collect();
    let tables = shared_tables(&lengths, 16, 64);
    let plans =
        plan_attention(&kv, &params, &tables, &lengths, &AttentionTuning::default()).unwrap();
    assert_eq!(plans.len(), 1);
    let p = &plans[0];
    check_plan(p);
    assert_eq!(p.kernel_name, "fllm_attn_prefill_f16_k16v16");
    assert_eq!(p.threadgroups, [3, 32, 1]);
    assert_eq!(p.threads_per_group, [PREFILL_THREADS, 1, 1]);
    assert_eq!((word(p, "row0"), word(p, "rows_end")), (0, 72));
    assert_eq!(p.buffer_count(), 6);

    // a tail of fewer than prefill_min_rows rows goes to the decode kernel
    let plans = plan_attention(
        &kv,
        &params,
        &tables[..70 * 64],
        &lengths[..70],
        &AttentionTuning::default(),
    )
    .unwrap();
    assert_eq!(plans.len(), 2);
    assert_eq!(plans[0].kernel_name, "fllm_attn_prefill_f16_k16v16");
    assert_eq!(
        (word(&plans[0], "row0"), word(&plans[0], "rows_end")),
        (0, 64)
    );
    assert!(plans[1].kernel_name.starts_with("fllm_attn_decode"));
    assert_eq!(
        (word(&plans[1], "row0"), word(&plans[1], "rows_end")),
        (64, 70)
    );

    // 3 rows: below prefill_min_rows -> decode
    let plans = plan_attention(
        &kv,
        &params,
        &shared_tables(&[5, 6, 7], 16, 64),
        &[5, 6, 7],
        &AttentionTuning::default(),
    )
    .unwrap();
    assert!(plans
        .iter()
        .all(|p| p.kernel_name.starts_with("fllm_attn_decode")
            || p.kernel_name == "fllm_attn_merge_f32"));

    // a device without simdgroup_matrix never gets the tiled kernel
    let no_mm = AttentionTuning {
        simdgroup_matrix: false,
        ..AttentionTuning::default()
    };
    let plans = plan_attention(&kv, &params, &tables, &lengths, &no_mm).unwrap();
    assert!(plans.iter().all(|p| !p.kernel_name.contains("prefill")));

    // different sequences in one batch: runs split where the block tables stop being prefixes of each other
    let mut lens: Vec<i64> = (1..=12).collect();
    lens.extend(1..=12);
    let mut tabs = Vec::new();
    for r in 0..24 {
        let seq = if r < 12 { 0 } else { 1 };
        tabs.extend((0..64i64).map(|b| seq * 32 + b % 32));
    }
    let plans = plan_attention(&kv, &params, &tabs, &lens, &AttentionTuning::default()).unwrap();
    let prefill: Vec<&OpPlan> = plans
        .iter()
        .filter(|p| p.kernel_name.contains("prefill"))
        .collect();
    assert_eq!(prefill.len(), 2, "one dispatch per sequence");
    assert_eq!(
        (word(prefill[0], "row0"), word(prefill[0], "rows_end")),
        (0, 12)
    );
    assert_eq!(
        (word(prefill[1], "row0"), word(prefill[1], "rows_end")),
        (12, 24)
    );

    // head widths without a tile instantiation (MLA 576/512) use the decode kernel
    let kv = layout(1, 1, 576, 512, 16, 8, 0);
    let params = attn_params(8, 1, 576, 512, 16, 8);
    let lengths: Vec<i64> = (1..=40).collect();
    let plans = plan_attention(
        &kv,
        &params,
        &shared_tables(&lengths, 16, 8),
        &lengths,
        &AttentionTuning::default(),
    )
    .unwrap();
    assert!(plans.iter().all(|p| !p.kernel_name.contains("prefill")));
}

#[test]
fn prefill_fragments_cover_the_widths() {
    for (dk, dv, expect) in [
        (64usize, 64usize, (8usize, 8usize)),
        (80, 80, (10, 10)),
        (96, 96, (12, 12)),
        (112, 112, (14, 14)),
        (128, 128, (16, 16)),
        (192, 128, (24, 16)),
        (256, 256, (32, 32)),
        (72, 72, (10, 10)),
        (192, 192, (24, 24)),
        (128, 64, (16, 16)),
    ] {
        assert_eq!(prefill_fragments(dk, dv), Some(expect), "{dk}/{dv}");
    }
    assert_eq!(prefill_fragments(264, 64), None);
    assert_eq!(prefill_fragments(64, 264), None);
}

#[test]
fn attention_feature_flags_reach_the_params() {
    let kv = layout(2, 2, 64, 64, 16, 8, 3);
    let mut params = attn_params(8, 2, 64, 64, 16, 8);
    params.softcap = Some(30.0);
    params.window = Some(32);
    params.sinks = Some(Arc::new(vec![0.5; 8]));
    params.alibi_slopes = Some(Arc::new(vec![0.25; 8]));
    let lengths = [70i64, 71];
    let tables: Vec<i64> = (0..16).map(|b| b % 8).collect();
    let tuning = AttentionTuning {
        use_prefill_kernel: false,
        ..AttentionTuning::default()
    };
    let plans = plan_attention(&kv, &params, &tables, &lengths, &tuning).unwrap();
    let d = &plans[0];
    check_plan(d);
    assert_eq!(word(d, "flags"), 3);
    assert_eq!(word(d, "window"), 32);
    assert_eq!(word(d, "chunk"), 0);
    assert_eq!(word(d, "rows_total"), 2);
    let softcap = f32::from_le_bytes(
        d.constants[32 + 6 * 4 + 4..32 + 6 * 4 + 8]
            .try_into()
            .unwrap(),
    );
    assert_eq!(softcap, 30.0);
    match &d.slot("bias").unwrap().source {
        SlotSource::Host(b) => {
            assert_eq!(b.len(), 16 * 4);
            assert_eq!(f32::from_le_bytes(b[0..4].try_into().unwrap()), 0.5);
            assert_eq!(
                f32::from_le_bytes(b[8 * 4..8 * 4 + 4].try_into().unwrap()),
                0.25
            );
        }
        other => panic!("{other:?}"),
    }
    match &d.slot("meta").unwrap().source {
        SlotSource::Host(b) => assert_eq!(b.len(), (2 + 16) * 4),
        other => panic!("{other:?}"),
    }
    // a window shortens the key range and with it the splitting
    assert_eq!(word(d, "nsplit"), 1);
}

#[test]
fn attention_rejects_bad_batches() {
    let kv = layout(1, 2, 64, 64, 16, 8, 0);
    let params = attn_params(4, 2, 64, 64, 16, 8);
    let ok_tables: Vec<i64> = (0..8).collect();
    let t = AttentionTuning::default();
    assert!(plan_attention(&kv, &params, &ok_tables, &[40], &t).is_ok());
    assert!(plan_attention(&kv, &params, &ok_tables, &[], &t).is_err()); // no rows
    assert!(plan_attention(&kv, &params, &ok_tables, &[0], &t).is_err()); // empty row
    assert!(plan_attention(&kv, &params, &ok_tables, &[8 * 16 + 1], &t).is_err()); // longer than the table
    assert!(plan_attention(&kv, &params, &[0, 1, 2, 3, 4, 5, 6, 8], &[128], &t).is_err()); // block outside the pool
    assert!(plan_attention(&kv, &params, &[0, 1, 2], &[40], &t).is_err()); // table size
    let mut p = params.clone();
    p.num_heads = 3;
    assert!(plan_attention(&kv, &p, &ok_tables, &[40], &t).is_err()); // heads % kv_heads
    let mut p = params.clone();
    p.window = Some(0);
    assert!(plan_attention(&kv, &p, &ok_tables, &[40], &t).is_err());
    let mut p = params.clone();
    p.softcap = Some(0.0);
    assert!(plan_attention(&kv, &p, &ok_tables, &[40], &t).is_err());
    let mut p = params.clone();
    p.sinks = Some(Arc::new(vec![0.0; 2]));
    assert!(plan_attention(&kv, &p, &ok_tables, &[40], &t).is_err());
    let mut p = params.clone();
    p.block_size = 32;
    assert!(plan_attention(&kv, &p, &ok_tables, &[40], &t).is_err()); // does not match the cache
    let narrow_simd = AttentionTuning {
        simd_width: 64,
        ..AttentionTuning::default()
    };
    assert!(matches!(
        plan_attention(&kv, &params, &ok_tables, &[40], &narrow_simd),
        Err(BackendError::UnsupportedOperation(_))
    ));
}

#[test]
fn many_rows_make_many_groups_and_few_splits() {
    // a continuous batch of 64 decode rows with 8 heads each already fills the device: no splitting
    let kv = layout(1, 2, 128, 128, 16, 64, 0);
    let params = attn_params(8, 2, 128, 128, 16, 64);
    let lengths = vec![900i64; 64];
    let mut tables = Vec::new();
    for r in 0..64usize {
        tables.extend((0..64i64).map(|b| (b + r as i64) % 64));
    }
    let plans = plan_attention(
        &kv,
        &params,
        &tables,
        &lengths,
        &AttentionTuning {
            use_prefill_kernel: false,
            ..AttentionTuning::default()
        },
    )
    .unwrap();
    assert_eq!(plans.len(), 1);
    assert_eq!(plans[0].threadgroups, [1, 8, 64]);
}

#[test]
fn threadgroup_limits_steer_the_kernel_choice() {
    let kv = layout(1, 2, 64, 64, 16, 8, 0);
    let params = attn_params(4, 2, 64, 64, 16, 8);
    // a pipeline that allows only 128 threads cannot run the decode kernel (256), but the prompt kernel (128) is fine
    let small = AttentionTuning {
        max_threads: 128,
        ..AttentionTuning::default()
    };
    let lengths: Vec<i64> = (1..=40).collect();
    let plans = plan_attention(
        &kv,
        &params,
        &shared_tables(&lengths, 16, 8),
        &lengths,
        &small,
    )
    .unwrap();
    assert!(plans.iter().all(|p| p.kernel_name.contains("prefill")));
    assert!(matches!(
        plan_attention(&kv, &params, &shared_tables(&[40], 16, 8), &[40], &small),
        Err(BackendError::UnsupportedOperation(_))
    ));
    // below 128 neither runs; with a limit of 64 even a prompt batch is refused
    let tiny = AttentionTuning {
        max_threads: 64,
        ..AttentionTuning::default()
    };
    assert!(plan_attention(
        &kv,
        &params,
        &shared_tables(&lengths, 16, 8),
        &lengths,
        &tiny
    )
    .is_err());
}
