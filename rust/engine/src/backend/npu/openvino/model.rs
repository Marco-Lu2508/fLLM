//! The graph the NPU runs for one projection: `y = x * W^T` with static shapes and the weights as constants of the model.
//!
//! ```text
//!   x: f16 [batch, cols] ------------------------------+
//!                                                       v
//!   codes: i4/i8 [rows, groups, group] -> Convert(f16) -> Multiply(scales f16 [rows, groups, 1]) -> Reshape [rows, cols]
//!                                                                                                       |
//!                                                                          MatMul(transpose_b) <--------+
//!                                                                               |
//!                                                                      Result: y f16 [batch, rows]
//! ```
//!
//! This is the weight-decompression pattern that NNCF writes for symmetric weight compression (and the pattern the NPU compiler
//! recognizes to keep the weights compressed in the compiled blob). Channel-wise weights use `[rows, cols]` codes and `[rows, 1]`
//! scales without the Reshape; FP16 weights are one constant. The bias, the per-tensor scale and the conversions between f32 and f16
//! stay on the host: the model's inputs and outputs are f16, the precision the NPU computes in.
//!
//! Verification level: L0 (structure checked by unit tests; the files have been read by the OpenVINO IR reader only in the
//! parser check of the test plan, never compiled for an NPU).

use super::ir::{Elem, IrBuilder};
use super::quant::{NpuFormat, NpuWeight};

/// An IR ready for `read_model`.
pub struct LinearIr {
    pub xml: String,
    pub bin: Vec<u8>,
    /// Rows of the (static) input.
    pub batch: usize,
    /// Output features.
    pub rows: usize,
    /// Input features.
    pub cols: usize,
}

/// Build the graph of one projection for a fixed number of input rows.
pub fn build_linear_ir(weight: &NpuWeight, batch: usize, name: &str) -> Result<LinearIr, String> {
    if batch == 0 {
        return Err("batch must be positive".into());
    }
    let (rows, cols) = (weight.rows, weight.cols);
    let mut b = IrBuilder::new(name);
    let x = b.parameter("x", Elem::F16, &[batch, cols]);
    let w = match weight.format {
        NpuFormat::Float16 => b.constant("weights", Elem::F16, &[rows, cols], &weight.codes)?,
        NpuFormat::Int4Sym | NpuFormat::Int8Sym => {
            let code_type = if weight.format == NpuFormat::Int4Sym {
                Elem::I4
            } else {
                Elem::I8
            };
            let groups = weight.groups();
            if groups == 0 || groups * weight.group != cols || weight.scales.len() != rows * groups
            {
                return Err("inconsistent scales for the weight matrix".into());
            }
            let scale_bytes: Vec<u8> = weight.scales.iter().flat_map(|s| s.to_le_bytes()).collect();
            if groups == 1 {
                // channel-wise: codes [rows, cols], scales [rows, 1]
                let codes = b.constant("weights", code_type, &[rows, cols], &weight.codes)?;
                let converted = b.op(
                    "Convert",
                    "weights_f16",
                    &[("destination_type", "f16")],
                    &[&codes],
                    Elem::F16,
                    &[rows, cols],
                );
                let scales = b.constant("scales", Elem::F16, &[rows, 1], &scale_bytes)?;
                b.op(
                    "Multiply",
                    "weights_scaled",
                    &[("auto_broadcast", "numpy")],
                    &[&converted, &scales],
                    Elem::F16,
                    &[rows, cols],
                )
            } else {
                let codes = b.constant(
                    "weights",
                    code_type,
                    &[rows, groups, weight.group],
                    &weight.codes,
                )?;
                let converted = b.op(
                    "Convert",
                    "weights_f16",
                    &[("destination_type", "f16")],
                    &[&codes],
                    Elem::F16,
                    &[rows, groups, weight.group],
                );
                let scales = b.constant("scales", Elem::F16, &[rows, groups, 1], &scale_bytes)?;
                let scaled = b.op(
                    "Multiply",
                    "weights_scaled",
                    &[("auto_broadcast", "numpy")],
                    &[&converted, &scales],
                    Elem::F16,
                    &[rows, groups, weight.group],
                );
                let shape: Vec<u8> = [rows as i64, cols as i64]
                    .iter()
                    .flat_map(|d| d.to_le_bytes())
                    .collect();
                let target = b.constant("weights_shape", Elem::I64, &[2], &shape)?;
                b.op(
                    "Reshape",
                    "weights_matrix",
                    &[("special_zero", "false")],
                    &[&scaled, &target],
                    Elem::F16,
                    &[rows, cols],
                )
            }
        }
    };
    let y = b.op(
        "MatMul",
        "linear",
        &[("transpose_a", "false"), ("transpose_b", "true")],
        &[&x, &w],
        Elem::F16,
        &[batch, rows],
    );
    b.result("y", &y);
    let (xml, bin) = b.finish();
    Ok(LinearIr {
        xml,
        bin,
        batch,
        rows,
        cols,
    })
}

#[cfg(test)]
mod tests {
    use super::super::ir::test_xml::parse;
    use super::super::quant::{convert, QuantPolicy};
    use super::*;
    use crate::backend::QuantType;

    fn q4_0(rows: usize, cols: usize) -> NpuWeight {
        let mut raw = Vec::new();
        for i in 0..rows * cols / 32 {
            raw.extend_from_slice(&half::f16::from_f32(0.05 + i as f32 * 0.001).to_le_bytes());
            raw.extend((0..16).map(|j| (i * 7 + j * 13) as u8));
        }
        convert(QuantType::Q4_0, &raw, rows, cols, &QuantPolicy::default()).unwrap()
    }

    fn types(xml: &str) -> Vec<String> {
        parse(xml)
            .unwrap()
            .into_iter()
            .filter(|e| e.tag == "layer")
            .map(|e| e.attrs["type"].clone())
            .collect()
    }

    #[test]
    fn grouped_int4_graph_has_the_decompression_pattern() {
        let w = q4_0(8, 64);
        let ir = build_linear_ir(&w, 16, "lin").unwrap();
        assert_eq!(
            types(&ir.xml),
            [
                "Parameter",
                "Const",
                "Convert",
                "Const",
                "Multiply",
                "Const",
                "Reshape",
                "MatMul",
                "Result"
            ]
        );
        let elements = parse(&ir.xml).unwrap();
        let weights = elements
            .iter()
            .find(|e| {
                e.tag == "data" && e.attrs.get("element_type").map(String::as_str) == Some("i4")
            })
            .unwrap();
        assert_eq!(weights.attrs["shape"], "8,2,32");
        assert_eq!(weights.attrs["size"], (8 * 64 / 2).to_string());
        assert!(elements.iter().any(|e| e.tag == "data"
            && e.attrs.get("destination_type").map(String::as_str) == Some("f16")));
        let mm = elements
            .iter()
            .find(|e| e.tag == "data" && e.attrs.contains_key("transpose_b"))
            .unwrap();
        assert_eq!(
            (
                mm.attrs["transpose_a"].as_str(),
                mm.attrs["transpose_b"].as_str()
            ),
            ("false", "true")
        );
        // the Parameter is [batch, cols]
        let x = elements
            .iter()
            .find(|e| {
                e.tag == "data" && e.attrs.contains_key("shape") && !e.attrs.contains_key("offset")
            })
            .unwrap();
        assert_eq!(x.attrs["shape"], "16,64");
        assert_eq!(x.attrs["element_type"], "f16");
    }

    #[test]
    fn channel_wise_graph_skips_the_reshape() {
        let mut w = q4_0(4, 32);
        // one group of 32 over 32 columns = channel-wise
        assert_eq!(w.groups(), 1);
        let ir = build_linear_ir(&w, 4, "cw").unwrap();
        assert_eq!(
            types(&ir.xml),
            [
                "Parameter",
                "Const",
                "Convert",
                "Const",
                "Multiply",
                "MatMul",
                "Result"
            ]
        );
        w.scales.pop();
        assert!(
            build_linear_ir(&w, 4, "bad").is_err(),
            "scale count is checked"
        );
    }

    #[test]
    fn float_and_int8_graphs() {
        let values: Vec<u8> = (0..24)
            .flat_map(|i| (i as f32 * 0.5).to_le_bytes())
            .collect();
        let f = convert(QuantType::F32, &values, 4, 6, &QuantPolicy::default()).unwrap();
        assert_eq!(
            types(&build_linear_ir(&f, 2, "f").unwrap().xml),
            ["Parameter", "Const", "MatMul", "Result"]
        );
        let mut q8 = Vec::new();
        for _ in 0..4 {
            q8.extend_from_slice(&half::f16::from_f32(0.1).to_le_bytes());
            q8.extend((0..32).map(|j| j as u8));
        }
        let w8 = convert(QuantType::Q8_0, &q8, 2, 64, &QuantPolicy::default()).unwrap();
        let ir = build_linear_ir(&w8, 8, "q8").unwrap();
        assert!(ir.xml.contains("element_type=\"i8\""));
        assert!(!ir.bin.is_empty());
        assert!(build_linear_ir(&w8, 0, "zero").is_err());
    }

    #[test]
    fn constants_fit_in_the_bin() {
        let w = q4_0(16, 128);
        let ir = build_linear_ir(&w, 32, "fit").unwrap();
        let elements = parse(&ir.xml).unwrap();
        let mut end = 0;
        for e in elements
            .iter()
            .filter(|e| e.tag == "data" && e.attrs.contains_key("offset"))
        {
            let offset: usize = e.attrs["offset"].parse().unwrap();
            let size: usize = e.attrs["size"].parse().unwrap();
            assert!(offset >= end);
            end = offset + size;
        }
        assert_eq!(
            end,
            ir.bin.len(),
            "the last constant ends the bin (no trailing bytes)"
        );
    }
}
