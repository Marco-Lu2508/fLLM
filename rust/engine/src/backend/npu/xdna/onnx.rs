//! Minimal ONNX `ModelProto` emitter for a single `MatMul` (package 11, AMD XDNA study).
//!
//! `y[m, n] = x[m, k] * w[k, n]` with the weight stored as an initializer. This is the smallest model that can be handed to ONNX Runtime
//! (`CreateSessionFromArray`) to see whether the VitisAI execution provider accepts and places a linear layer; it is the "single MatMul
//! model" of the XDNA study, not an LLM path. Field numbers are those of `onnx/onnx.proto3` (checked against the file at the time of
//! writing): ModelProto{ir_version=1, producer_name=2, graph=7, opset_import=8}, GraphProto{node=1, name=2, initializer=5, input=11,
//! output=12}, NodeProto{input=1, output=2, name=3, op_type=4}, TensorProto{dims=1, data_type=2, name=8, raw_data=9},
//! ValueInfoProto{name=1, type=2}, TypeProto{tensor_type=1}, TypeProto.Tensor{elem_type=1, shape=2}, TensorShapeProto{dim=1},
//! Dimension{dim_value=1}, OperatorSetIdProto{domain=1, version=2}.
//! Pure Rust, no hardware.

use super::pb::Message;

/// `TensorProto.DataType` values used here.
pub const ONNX_FLOAT: i32 = 1;
pub const ONNX_FLOAT16: i32 = 10;
pub const ONNX_BFLOAT16: i32 = 16;

/// ONNX IR version 8 (ONNX 1.10) and operator set 17: the version AMD recommends for Ryzen AI ("models with ONNX opset 17 are recommended").
const IR_VERSION: i64 = 8;
const OPSET: i64 = 17;

fn shape_proto(dims: &[i64]) -> Message {
    let mut shape = Message::new();
    for &d in dims {
        let mut dim = Message::new();
        dim.int(1, d);
        shape.message(1, &dim);
    }
    shape
}

fn value_info(name: &str, elem_type: i32, dims: &[i64]) -> Message {
    let mut tensor = Message::new();
    tensor
        .int(1, i64::from(elem_type))
        .message(2, &shape_proto(dims));
    let mut ty = Message::new();
    ty.message(1, &tensor);
    let mut info = Message::new();
    info.string(1, name).message(2, &ty);
    info
}

fn tensor_proto(name: &str, elem_type: i32, dims: &[i64], raw: &[u8]) -> Message {
    let mut t = Message::new();
    for &d in dims {
        t.int(1, d);
    }
    t.int(2, i64::from(elem_type)).string(8, name).bytes(9, raw);
    t
}

/// Bytes of a serialized ONNX model computing `y = x . w` with `x: [m, k]` (graph input `"x"`), `w: [k, n]` (initializer `"w"`, given as
/// row-major little-endian raw data of element type `elem_type`) and `y: [m, n]` (graph output `"y"`).
pub fn matmul_model(
    m: usize,
    k: usize,
    n: usize,
    elem_type: i32,
    weight_raw: &[u8],
) -> Result<Vec<u8>, String> {
    if m == 0 || k == 0 || n == 0 {
        return Err("matmul_model: dimensions must be positive".to_owned());
    }
    let elem_size = match elem_type {
        ONNX_FLOAT => 4,
        ONNX_FLOAT16 | ONNX_BFLOAT16 => 2,
        other => return Err(format!("matmul_model: unsupported element type {other}")),
    };
    if weight_raw.len() != k * n * elem_size {
        return Err(format!(
            "matmul_model: weight has {} bytes, expected {}",
            weight_raw.len(),
            k * n * elem_size
        ));
    }
    let (m, k, n) = (m as i64, k as i64, n as i64);

    let mut node = Message::new();
    node.string(1, "x")
        .string(1, "w")
        .string(2, "y")
        .string(3, "matmul0")
        .string(4, "MatMul");

    let mut graph = Message::new();
    graph.message(1, &node).string(2, "fllm_linear");
    graph.message(5, &tensor_proto("w", elem_type, &[k, n], weight_raw));
    graph.message(11, &value_info("x", elem_type, &[m, k]));
    graph.message(12, &value_info("y", elem_type, &[m, n]));

    let mut opset = Message::new();
    opset.string(1, "").int(2, OPSET);

    let mut model = Message::new();
    model
        .int(1, IR_VERSION)
        .string(2, "fllm")
        .message(7, &graph)
        .message(8, &opset);
    Ok(model.into_bytes())
}

/// Convenience for the float32 case of [`matmul_model`].
pub fn matmul_model_f32(m: usize, k: usize, n: usize, weight: &[f32]) -> Result<Vec<u8>, String> {
    let mut raw = Vec::with_capacity(weight.len() * 4);
    for v in weight {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    matmul_model(m, k, n, ONNX_FLOAT, &raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_weight_size() {
        assert!(matmul_model_f32(1, 2, 3, &[0.0; 5]).is_err());
        assert!(matmul_model_f32(0, 2, 3, &[0.0; 6]).is_err());
        assert!(matmul_model(1, 2, 3, 99, &[0; 6]).is_err());
    }

    #[test]
    fn model_starts_with_ir_version_and_contains_the_names() {
        let bytes = matmul_model_f32(2, 3, 4, &[0.5; 12]).unwrap();
        // field 1 (ir_version) = 8
        assert_eq!(&bytes[..2], &[0x08, 0x08]);
        let text = String::from_utf8_lossy(&bytes);
        for needle in ["MatMul", "matmul0", "fllm_linear", "fllm"] {
            assert!(text.contains(needle), "missing {needle}");
        }
        // the 12 weights of 4 bytes each are embedded verbatim
        let raw: Vec<u8> = (0..12).flat_map(|_| 0.5f32.to_le_bytes()).collect();
        assert!(bytes.windows(raw.len()).any(|w| w == raw.as_slice()));
    }

    #[test]
    fn value_info_encodes_shape_dims_in_order() {
        let info = value_info("x", ONNX_FLOAT, &[2, 3]).into_bytes();
        // name "x", then type { tensor_type { elem_type 1, shape { dim{2}, dim{3} } } }
        assert_eq!(
            info,
            [
                0x0a, 0x01, b'x', 0x12, 0x0e, 0x0a, 0x0c, 0x08, 0x01, 0x12, 0x08, 0x0a, 0x02, 0x08,
                0x02, 0x0a, 0x02, 0x08, 0x03
            ]
        );
    }
}
