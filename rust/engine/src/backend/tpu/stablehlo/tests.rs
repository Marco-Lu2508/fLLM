//! Tests of the StableHLO emitter and of the structure check: formatting of constants and types, the shape rules of the builder, a
//! program built by hand, and programs corrupted in the ways an emitter bug would.

use super::validate::check;
use super::*;

fn sample() -> (Func, Value) {
    let mut f = Func::new("main");
    let x = f.param(&[2, 3], DType::F32);
    let w = f.param(&[4, 3], DType::F32);
    let b = f.param(&[4], DType::F32);
    let y = f.dot_general(
        &x,
        &w,
        &DotDims {
            lhs_contract: vec![1],
            rhs_contract: vec![1],
            ..Default::default()
        },
        DType::F32,
        Precision::Highest,
    );
    let biased = f.add_bias(&y, &b);
    let m = f.reduce(&biased, Reducer::Max, &[1]);
    let mb = f.unreduce(&m, biased.dims(), &[1]);
    let shifted = f.sub(&biased, &mb);
    let e = f.exp(&shifted);
    (f, e)
}

#[test]
fn float_literals_always_carry_a_point_or_exponent() {
    assert_eq!(float_literal(0.0), "0.0");
    assert_eq!(float_literal(1.0), "1.0");
    assert_eq!(float_literal(-2.0), "-2.0");
    assert_eq!(float_literal(0.5), "0.5");
    assert_eq!(float_literal(1e-6), "1.0e-6");
    assert_eq!(float_literal(2.5e-8), "2.5e-8");
    assert_eq!(float_literal(1.0e20), "1.0e20");
}

#[test]
fn tensor_types_print_like_mlir() {
    assert_eq!(
        TensorType::new(&[2, 3], DType::F32).to_string(),
        "tensor<2x3xf32>"
    );
    assert_eq!(TensorType::new(&[], DType::I32).to_string(), "tensor<i32>");
    assert_eq!(
        TensorType::new(&[7], DType::Bf16).to_string(),
        "tensor<7xbf16>"
    );
    assert_eq!(
        TensorType::new(&[7], DType::Pred).to_string(),
        "tensor<7xi1>"
    );
    assert_eq!(TensorType::new(&[2, 3], DType::U8).elements(), 6);
}

#[test]
fn a_hand_built_program_passes_the_check() {
    let (f, e) = sample();
    let text = module("sample", &f, &[&e]);
    let report = check(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    assert_eq!(report.functions, 1);
    assert_eq!(report.params, 3);
    assert!(report.ops >= 8, "{report:?}");
    assert!(text.contains("\"stablehlo.dot_general\""));
    assert!(text.contains("precision HIGHEST"));
    assert!(text.contains("\"stablehlo.reduce\"") && text.contains("^bb0("));
}

#[test]
fn donation_attribute_is_written_on_the_parameter() {
    let mut f = Func::new("main");
    let x = f.param(&[8], DType::F32);
    let one = f.constant(1.0, &[8], DType::F32);
    let y = f.add(&x, &one);
    f.set_param_attr(&x, "tf.aliasing_output = 0 : i32".into());
    let text = module("donate", &f, &[&y]);
    assert!(
        text.contains("%arg0: tensor<8xf32> {tf.aliasing_output = 0 : i32}"),
        "{text}"
    );
    check(&text).expect("valid");
}

#[test]
fn bitcast_follows_the_shape_rules() {
    let mut f = Func::new("main");
    let x = f.param(&[4], DType::U32);
    let as_float = f.bitcast_convert(&x, DType::F32);
    assert_eq!(as_float.dims(), [4]);
    let bytes = f.bitcast_convert(&x, DType::U8);
    assert_eq!(bytes.dims(), [4, 4]);
    let back = f.bitcast_convert(&bytes, DType::U32);
    assert_eq!(back.dims(), [4]);
    let halves = f.bitcast_convert(&x, DType::F16);
    assert_eq!(halves.dims(), [4, 2]);
    let text = module("bitcast", &f, &[&as_float, &back, &halves]);
    check(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
}

#[test]
#[should_panic(expected = "operand types differ")]
fn binary_operations_reject_mismatched_types() {
    let mut f = Func::new("main");
    let a = f.param(&[2], DType::F32);
    let b = f.param(&[3], DType::F32);
    f.add(&a, &b);
}

#[test]
#[should_panic(expected = "different element count")]
fn reshape_rejects_a_different_element_count() {
    let mut f = Func::new("main");
    let a = f.param(&[2, 3], DType::F32);
    f.reshape(&a, &[7]);
}

#[test]
#[should_panic(expected = "contracting dimension")]
fn dot_general_rejects_mismatched_contraction() {
    let mut f = Func::new("main");
    let a = f.param(&[2, 3], DType::F32);
    let b = f.param(&[4, 5], DType::F32);
    f.dot_general(
        &a,
        &b,
        &DotDims {
            lhs_contract: vec![1],
            rhs_contract: vec![1],
            ..Default::default()
        },
        DType::F32,
        Precision::Default,
    );
}

#[test]
fn gather_and_scatter_shapes() {
    let mut f = Func::new("main");
    let table = f.param(&[10, 4], DType::F32);
    let ids = f.param(&[3], DType::I32);
    let rows = f.gather_rows(&table, &ids);
    assert_eq!(rows.dims(), [3, 4]);
    let back = f.scatter_rows(&table, &ids, &rows, true, false);
    assert_eq!(back.dims(), [10, 4]);
    let text = module("rows", &f, &[&rows, &back]);
    check(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
}

#[test]
fn symbols_are_valid_mlir_names() {
    assert_eq!(symbol("rms_norm"), "rms_norm");
    assert_eq!(symbol("a-b c"), "a_b_c");
    assert_eq!(symbol("9lives"), "m_9lives");
    assert_eq!(symbol(""), "m_");
}

fn working() -> String {
    let (f, e) = sample();
    module("sample", &f, &[&e])
}

#[test]
fn check_rejects_unbalanced_brackets() {
    let text = working().replacen("tensor<2x3xf32>", "tensor<2x3xf32", 1);
    assert!(check(&text).is_err());
    let text = working().replacen("(%arg0,", "((%arg0,", 1);
    assert!(check(&text).is_err());
}

#[test]
fn check_rejects_use_before_definition() {
    let text = working().replace("(%v1)", "(%v99)");
    // the sample has no `(%v1)` operand list with one entry under that name; corrupt a real operand instead
    let text = if text == working() {
        working().replacen(
            "\"stablehlo.exponential\"(%v",
            "\"stablehlo.exponential\"(%w",
            1,
        )
    } else {
        text
    };
    let error = check(&text).expect_err("an undefined operand must be rejected");
    assert!(error.contains("before it is defined"), "{error}");
}

#[test]
fn check_rejects_wrong_operand_types() {
    let text = working().replacen(
        "(tensor<2x3xf32>, tensor<4x3xf32>)",
        "(tensor<2x4xf32>, tensor<4x3xf32>)",
        1,
    );
    let error = check(&text).expect_err("a wrong operand type must be rejected");
    assert!(error.contains("expects"), "{error}");
}

#[test]
fn check_rejects_wrong_result_shapes() {
    let text = working().replacen("-> tensor<2x4xf32>", "-> tensor<2x5xf32>", 1);
    assert!(check(&text).is_err());
}

#[test]
fn check_rejects_double_definitions_and_wrong_returns() {
    let text = working().replacen("%v2 = ", "%v1 = ", 1);
    assert!(check(&text).is_err());
    let text = working().replacen("func.return", "func.return %arg0, ", 1);
    assert!(check(&text).is_err());
}

#[test]
fn check_rejects_text_outside_a_module() {
    assert!(check("func.func public @main() -> () {\n  func.return\n}\n").is_err());
    assert!(check("").is_err());
}
