//! Text form of a minimal MIL program for one linear layer (package 11, Apple Neural Engine study).
//!
//! MIL is the intermediate language of Core ML ML Programs. This module prints the program that [`super::mlpackage`] serializes as
//! protobuf, in the textual syntax `coremltools` uses for `print(prog)`, so that a human can read what the package contains and compare
//! it with the output of `coremltools` on a Mac. The text is informational: Core ML cannot compile it directly, and the syntax printed
//! here is written from the documented examples, not checked against a MIL parser (L0). Pure Rust, no hardware.

use std::fmt::Write as _;

/// The MIL opset name used in `func main<ios16>`; the protobuf form names the same opset "CoreML6" (iOS 16 / macOS 13).
pub const OPSET_TEXT: &str = "ios16";
/// Path under which the package's weight blob is referenced (`@model_path` is the package's `Data/com.apple.CoreML` directory).
pub const WEIGHT_BLOB_PATH: &str = "@model_path/weights/weight.bin";

fn shape_text(dims: &[usize]) -> String {
    let inner: Vec<String> = dims.iter().map(usize::to_string).collect();
    format!("[{}]", inner.join(", "))
}

/// MIL text of `y = linear(x, weight[, bias])` with `x: [m, k]`, `weight: [n, k]`, `bias: [n]`, all fp16. `weight_offset` and
/// `bias_offset` are the offsets of the two blobs' metadata records in the weight file.
pub fn linear_program_text(
    m: usize,
    k: usize,
    n: usize,
    weight_offset: u64,
    bias_offset: Option<u64>,
) -> String {
    let (x, w, b, y) = (
        shape_text(&[m, k]),
        shape_text(&[n, k]),
        shape_text(&[n]),
        shape_text(&[m, n]),
    );
    let mut s = String::new();
    let _ = writeln!(s, "program(1.0)");
    let _ = writeln!(s, "{{");
    let _ = writeln!(s, "    func main<{OPSET_TEXT}>(tensor<fp16, {x}> x) {{");
    let _ = writeln!(
        s,
        "        tensor<fp16, {w}> weight = const()[name = string(\"weight\"), val = tensor<fp16, {w}>(BLOBFILE(path = string(\"{WEIGHT_BLOB_PATH}\"), offset = uint64({weight_offset})))];"
    );
    if let Some(offset) = bias_offset {
        let _ = writeln!(
            s,
            "        tensor<fp16, {b}> bias = const()[name = string(\"bias\"), val = tensor<fp16, {b}>(BLOBFILE(path = string(\"{WEIGHT_BLOB_PATH}\"), offset = uint64({offset})))];"
        );
        let _ = writeln!(s, "        tensor<fp16, {y}> y = linear(x = x, weight = weight, bias = bias)[name = string(\"linear_0\")];");
    } else {
        let _ = writeln!(s, "        tensor<fp16, {y}> y = linear(x = x, weight = weight)[name = string(\"linear_0\")];");
    }
    let _ = writeln!(s, "    }} -> (y);");
    let _ = writeln!(s, "}}");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prints_a_linear_without_bias() {
        let text = linear_program_text(1, 64, 32, 64, None);
        assert!(text.starts_with("program(1.0)\n{\n"));
        assert!(text.contains("func main<ios16>(tensor<fp16, [1, 64]> x)"));
        assert!(text.contains("tensor<fp16, [32, 64]> weight = const()"));
        assert!(text.contains("offset = uint64(64)"));
        assert!(text.contains("tensor<fp16, [1, 32]> y = linear(x = x, weight = weight)"));
        assert!(text.trim_end().ends_with("} -> (y);\n}"));
        assert!(!text.contains("bias"));
    }

    #[test]
    fn prints_a_linear_with_bias() {
        let text = linear_program_text(4, 8, 16, 64, Some(256));
        assert!(text.contains("tensor<fp16, [16]> bias = const()"));
        assert!(text.contains("offset = uint64(256)"));
        assert!(text.contains("linear(x = x, weight = weight, bias = bias)"));
    }
}
