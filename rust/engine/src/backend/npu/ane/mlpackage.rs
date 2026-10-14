//! Writer for a minimal Core ML `.mlpackage` holding one fp16 linear layer as an ML Program (package 11, Apple Neural Engine study).
//!
//! An `.mlpackage` is a directory:
//!
//! ```text
//! Manifest.json
//! Data/com.apple.CoreML/model.mlmodel        the CoreML.Specification.Model protobuf (with the MIL Program)
//! Data/com.apple.CoreML/weights/weight.bin   the weights in the MIL blob storage format (version 2)
//! ```
//!
//! All field numbers are those of the `.proto` files of `apple/coremltools` (BSD-3-Clause; checked at the time of writing): `Model.proto`
//! (Model{specificationVersion=1, description=2, mlProgram=502}, ModelDescription{input=1, output=10}, FeatureDescription{name=1, type=3}),
//! `FeatureTypes.proto` (FeatureType{multiArrayType=5}, ArrayFeatureType{shape=1, dataType=2}; FLOAT16=65552), `MIL.proto` (Program{version=1,
//! functions=2}, Function{inputs=1, opset=2, block_specializations=3}, Block{outputs=2, operations=3}, Operation{type=1, inputs=2, outputs=3,
//! attributes=5}, Argument{arguments=1}, Binding{name=1}, NamedValueType{name=1, type=2}, ValueType{tensorType=1}, TensorType{dataType=1,
//! rank=2, dimensions=3}, Dimension{constant=1}, ConstantDimension{size=1}, Value{type=2, immediateValue=3, blobFileValue=5},
//! ImmediateValue{tensor=1}, TensorValue{strings=4}, RepeatedStrings{values=1}, BlobFileValue{fileName=1, offset=2}; DataType FLOAT16=10,
//! STRING=2) and the blob format of `MILBlob/Blob/StorageFormat.hpp` (64-byte header, 64-byte metadata per blob, 64-byte alignment,
//! sentinel 0xDEADBEEF, dtype Float16=1). The result is untested against Core ML (L0); the unit tests check the byte-level structure.
//! Pure Rust (std::fs only), no hardware.

use crate::backend::npu::xdna::pb::Message;
use std::path::Path;

/// `Model.specificationVersion` 7: iOS 16 / macOS 13 (ML Programs with float16 multi-arrays).
pub const SPECIFICATION_VERSION: i64 = 7;
/// The protobuf name of the iOS 16 opset (`ios16` in the MIL text).
pub const OPSET: &str = "CoreML6";
/// `ArrayFeatureType.ArrayDataType.FLOAT16`.
pub const ARRAY_FLOAT16: u64 = 65552;
/// `MILSpec.DataType`: STRING = 2, FLOAT16 = 10.
const MIL_STRING: u64 = 2;
const MIL_FLOAT16: u64 = 10;
/// `BlobDataType::Float16`.
const BLOB_FLOAT16: u32 = 1;
const BLOB_SENTINEL: u32 = 0xDEAD_BEEF;
const BLOB_ALIGNMENT: usize = 64;
/// Path of the weights as the program refers to it.
pub const WEIGHT_FILE_NAME: &str = "@model_path/weights/weight.bin";

const MANIFEST_MODEL_ID: &str = "8A5D1E54-52B0-4D0E-A1C1-7F0D3B7A2E10";
const MANIFEST_WEIGHTS_ID: &str = "C3F6B0AE-93C4-4B57-9E6E-2D1B6F4A8C21";

/// Shape and fp16 data of the linear layer: `y = x . weight^T (+ bias)` with `x: [m, k]`, `weight: [n, k]`, `bias: [n]`.
pub struct LinearSpec<'a> {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    /// Row-major fp16 bit patterns, `n * k` values.
    pub weight: &'a [u16],
    /// fp16 bit patterns, `n` values.
    pub bias: Option<&'a [u16]>,
}

impl LinearSpec<'_> {
    fn validate(&self) -> Result<(), String> {
        if self.m == 0 || self.k == 0 || self.n == 0 {
            return Err("linear: dimensions must be positive".to_owned());
        }
        if self.weight.len() != self.n * self.k {
            return Err(format!(
                "linear: weight has {} values, expected {}",
                self.weight.len(),
                self.n * self.k
            ));
        }
        if let Some(bias) = self.bias {
            if bias.len() != self.n {
                return Err(format!(
                    "linear: bias has {} values, expected {}",
                    bias.len(),
                    self.n
                ));
            }
        }
        Ok(())
    }
}

fn dims_u64(dims: &[usize]) -> Vec<u64> {
    dims.iter().map(|&d| d as u64).collect()
}

/// `MILSpec.TensorType`.
fn tensor_type(data_type: u64, dims: &[usize]) -> Message {
    let mut t = Message::new();
    t.varint(1, data_type).int(2, dims.len() as i64);
    for d in dims_u64(dims) {
        let mut constant = Message::new();
        constant.varint(1, d);
        let mut dimension = Message::new();
        dimension.message(1, &constant);
        t.message(3, &dimension);
    }
    t
}

/// `MILSpec.ValueType{tensorType}`.
fn value_type(data_type: u64, dims: &[usize]) -> Message {
    let mut v = Message::new();
    v.message(1, &tensor_type(data_type, dims));
    v
}

/// `MILSpec.NamedValueType`.
fn named_value_type(name: &str, ty: &Message) -> Message {
    let mut n = Message::new();
    n.string(1, name).message(2, ty);
    n
}

/// A scalar string constant as `MILSpec.Value` (used for the `name` attributes).
fn string_value(text: &str) -> Message {
    let mut strings = Message::new();
    strings.string(1, text);
    let mut tensor = Message::new();
    tensor.message(4, &strings);
    let mut immediate = Message::new();
    immediate.message(1, &tensor);
    let mut v = Message::new();
    v.message(2, &value_type(MIL_STRING, &[]))
        .message(3, &immediate);
    v
}

/// An fp16 tensor stored in the weight blob file, as `MILSpec.Value{blobFileValue}`.
fn blob_value(dims: &[usize], offset: u64) -> Message {
    let mut blob = Message::new();
    blob.string(1, WEIGHT_FILE_NAME).varint(2, offset);
    let mut v = Message::new();
    v.message(2, &value_type(MIL_FLOAT16, dims))
        .message(5, &blob);
    v
}

/// `MILSpec.Argument` binding the given variable names.
fn argument(names: &[&str]) -> Message {
    let mut a = Message::new();
    for name in names {
        let mut binding = Message::new();
        binding.string(1, name);
        a.message(1, &binding);
    }
    a
}

fn const_operation(name: &str, dims: &[usize], blob_offset: u64) -> Message {
    let mut op = Message::new();
    op.string(1, "const")
        .message(3, &named_value_type(name, &value_type(MIL_FLOAT16, dims)));
    op.map_entry(5, "name", &string_value(name)).map_entry(
        5,
        "val",
        &blob_value(dims, blob_offset),
    );
    op
}

/// The `MILSpec.Program` of the layer. `weight_offset` / `bias_offset` are the blob metadata offsets from [`build_weight_blob`].
pub fn linear_program(
    spec: &LinearSpec<'_>,
    weight_offset: u64,
    bias_offset: Option<u64>,
) -> Result<Message, String> {
    spec.validate()?;
    let mut operations = vec![const_operation("weight", &[spec.n, spec.k], weight_offset)];
    if let Some(offset) = bias_offset {
        operations.push(const_operation("bias", &[spec.n], offset));
    }
    let mut linear = Message::new();
    linear.string(1, "linear");
    linear
        .map_entry(2, "x", &argument(&["x"]))
        .map_entry(2, "weight", &argument(&["weight"]));
    if bias_offset.is_some() {
        linear.map_entry(2, "bias", &argument(&["bias"]));
    }
    linear.message(
        3,
        &named_value_type("y", &value_type(MIL_FLOAT16, &[spec.m, spec.n])),
    );
    linear.map_entry(5, "name", &string_value("linear_0"));
    operations.push(linear);

    let mut block = Message::new();
    block.string(2, "y");
    for op in &operations {
        block.message(3, op);
    }
    let mut function = Message::new();
    function.message(
        1,
        &named_value_type("x", &value_type(MIL_FLOAT16, &[spec.m, spec.k])),
    );
    function.string(2, OPSET).map_entry(3, OPSET, &block);
    let mut program = Message::new();
    program.int(1, 1).map_entry(2, "main", &function);
    Ok(program)
}

fn feature_description(name: &str, dims: &[usize]) -> Message {
    let mut array = Message::new();
    array
        .packed_ints(1, &dims.iter().map(|&d| d as i64).collect::<Vec<_>>())
        .varint(2, ARRAY_FLOAT16);
    let mut feature_type = Message::new();
    feature_type.message(5, &array);
    let mut description = Message::new();
    description.string(1, name).message(3, &feature_type);
    description
}

/// The serialized `CoreML.Specification.Model` (the content of `model.mlmodel`).
pub fn linear_model(
    spec: &LinearSpec<'_>,
    weight_offset: u64,
    bias_offset: Option<u64>,
) -> Result<Vec<u8>, String> {
    let program = linear_program(spec, weight_offset, bias_offset)?;
    let mut description = Message::new();
    description.message(1, &feature_description("x", &[spec.m, spec.k]));
    description.message(10, &feature_description("y", &[spec.m, spec.n]));
    let mut model = Message::new();
    model
        .int(1, SPECIFICATION_VERSION)
        .message(2, &description)
        .message(502, &program);
    Ok(model.into_bytes())
}

/// Weight file in the MIL blob storage format: 64-byte header, then per blob a 64-byte metadata record followed by the (64-byte aligned)
/// data. Returns the file and the offset of each blob's metadata record (the value `BlobFileValue.offset` refers to).
pub fn build_weight_blob(blobs: &[&[u16]]) -> (Vec<u8>, Vec<u64>) {
    let align = |v: usize| v.div_ceil(BLOB_ALIGNMENT) * BLOB_ALIGNMENT;
    let mut file = vec![0u8; BLOB_ALIGNMENT];
    file[0..4].copy_from_slice(&(blobs.len() as u32).to_le_bytes());
    file[4..8].copy_from_slice(&2u32.to_le_bytes());
    let mut offsets = Vec::with_capacity(blobs.len());
    for blob in blobs {
        let meta_at = file.len();
        let size = blob.len() * 2;
        let data_at = meta_at + BLOB_ALIGNMENT;
        let mut meta = [0u8; BLOB_ALIGNMENT];
        meta[0..4].copy_from_slice(&BLOB_SENTINEL.to_le_bytes());
        meta[4..8].copy_from_slice(&BLOB_FLOAT16.to_le_bytes());
        meta[8..16].copy_from_slice(&(size as u64).to_le_bytes());
        meta[16..24].copy_from_slice(&(data_at as u64).to_le_bytes());
        // padding_size_in_bits and the four reserved words stay zero
        file.extend_from_slice(&meta);
        for v in *blob {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.resize(align(file.len()), 0);
        offsets.push(meta_at as u64);
    }
    (file, offsets)
}

/// `Manifest.json` of the package.
pub fn manifest_json() -> String {
    format!(
        "{{\n    \"fileFormatVersion\": \"1.0.0\",\n    \"itemInfoEntries\": {{\n        \"{MANIFEST_MODEL_ID}\": {{\n            \"author\": \"com.apple.CoreML\",\n            \"description\": \"CoreML Model Specification\",\n            \"name\": \"model.mlmodel\",\n            \"path\": \"com.apple.CoreML/model.mlmodel\"\n        }},\n        \"{MANIFEST_WEIGHTS_ID}\": {{\n            \"author\": \"com.apple.CoreML\",\n            \"description\": \"CoreML Model Weights\",\n            \"name\": \"weights\",\n            \"path\": \"com.apple.CoreML/weights\"\n        }}\n    }},\n    \"rootModelIdentifier\": \"{MANIFEST_MODEL_ID}\"\n}}\n"
    )
}

/// Writes the complete package into `dir` (created if missing). Returns the MIL text of the program for logging.
pub fn write_linear_package(dir: &Path, spec: &LinearSpec<'_>) -> Result<String, String> {
    spec.validate()?;
    let mut blobs: Vec<&[u16]> = vec![spec.weight];
    if let Some(bias) = spec.bias {
        blobs.push(bias);
    }
    let (weights, offsets) = build_weight_blob(&blobs);
    let bias_offset = spec.bias.map(|_| offsets[1]);
    let model = linear_model(spec, offsets[0], bias_offset)?;
    let data = dir.join("Data").join("com.apple.CoreML");
    let io = |what: &str, e: std::io::Error| format!("{what}: {e}");
    std::fs::create_dir_all(data.join("weights"))
        .map_err(|e| io("create package directories", e))?;
    std::fs::write(dir.join("Manifest.json"), manifest_json())
        .map_err(|e| io("write Manifest.json", e))?;
    std::fs::write(data.join("model.mlmodel"), model).map_err(|e| io("write model.mlmodel", e))?;
    std::fs::write(data.join("weights").join("weight.bin"), weights)
        .map_err(|e| io("write weight.bin", e))?;
    Ok(super::mil::linear_program_text(
        spec.m,
        spec.k,
        spec.n,
        offsets[0],
        bias_offset,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_blob_layout_matches_the_storage_format() {
        let a: Vec<u16> = (0..40).collect(); // 80 bytes
        let b: Vec<u16> = vec![7; 3]; // 6 bytes
        let (file, offsets) = build_weight_blob(&[&a, &b]);
        assert_eq!(offsets, [64, 64 + 64 + 128]); // 80 bytes of data are padded to 128
        assert_eq!(&file[0..4], &2u32.to_le_bytes()); // count
        assert_eq!(&file[4..8], &2u32.to_le_bytes()); // version
                                                      // first metadata record at 64
        assert_eq!(&file[64..68], &0xDEAD_BEEFu32.to_le_bytes());
        assert_eq!(&file[68..72], &1u32.to_le_bytes()); // Float16
        assert_eq!(&file[72..80], &80u64.to_le_bytes()); // size in bytes
        assert_eq!(&file[80..88], &128u64.to_le_bytes()); // data offset
        assert_eq!(&file[128..130], &0u16.to_le_bytes());
        assert_eq!(&file[130..132], &1u16.to_le_bytes());
        // second metadata record starts after the padded data
        let second = offsets[1] as usize;
        assert_eq!(&file[second..second + 4], &0xDEAD_BEEFu32.to_le_bytes());
        assert_eq!(&file[second + 8..second + 16], &6u64.to_le_bytes());
        assert_eq!(
            &file[second + 16..second + 24],
            &((second + 64) as u64).to_le_bytes()
        );
        assert_eq!(file.len() % 64, 0);
    }

    #[test]
    fn model_contains_the_expected_names_and_numbers() {
        let weight = vec![0x3c00u16; 8 * 4]; // 1.0 in fp16
        let spec = LinearSpec {
            m: 2,
            k: 4,
            n: 8,
            weight: &weight,
            bias: None,
        };
        let (_, offsets) = build_weight_blob(&[&weight]);
        let model = linear_model(&spec, offsets[0], None).unwrap();
        // specificationVersion = 7 is the first field
        assert_eq!(&model[..2], &[0x08, 0x07]);
        let text = String::from_utf8_lossy(&model);
        for needle in [
            "CoreML6",
            "main",
            "const",
            "weight",
            "linear",
            "linear_0",
            "y",
            "x",
            WEIGHT_FILE_NAME,
        ] {
            assert!(text.contains(needle), "missing {needle}");
        }
        // mlProgram is field 502 (wire type 2): tag varint (502 << 3) | 2 = 4018 -> b2 1f
        assert!(model.windows(2).any(|w| w == [0xb2, 0x1f]));
    }

    #[test]
    fn feature_description_encodes_shape_and_fp16() {
        let d = feature_description("x", &[2, 3]).into_bytes();
        // name "x"; type { multiArrayType (5) { shape packed [2,3], dataType 65552 } }
        assert_eq!(d[..3], [0x0a, 0x01, b'x']);
        // dataType varint 65552 = 0x10010 -> 90 80 04
        assert!(d.windows(4).any(|w| w == [0x10, 0x90, 0x80, 0x04]));
        assert!(d.windows(4).any(|w| w == [0x0a, 0x02, 0x02, 0x03]));
    }

    #[test]
    fn validation_catches_wrong_sizes() {
        let weight = vec![0u16; 5];
        assert!(linear_model(
            &LinearSpec {
                m: 1,
                k: 2,
                n: 3,
                weight: &weight,
                bias: None
            },
            64,
            None
        )
        .is_err());
        let weight = vec![0u16; 6];
        assert!(linear_model(
            &LinearSpec {
                m: 1,
                k: 2,
                n: 3,
                weight: &weight,
                bias: Some(&[0, 0])
            },
            64,
            Some(256)
        )
        .is_err());
        assert!(linear_model(
            &LinearSpec {
                m: 0,
                k: 2,
                n: 3,
                weight: &weight,
                bias: None
            },
            64,
            None
        )
        .is_err());
    }

    #[test]
    fn package_is_written_with_all_three_files() {
        let dir = std::env::temp_dir().join(format!(
            "fllm-mlpackage-test-{}-{}",
            std::process::id(),
            line!()
        ));
        let weight = vec![0x3c00u16; 6];
        let spec = LinearSpec {
            m: 1,
            k: 2,
            n: 3,
            weight: &weight,
            bias: Some(&[0, 0, 0]),
        };
        let text = write_linear_package(&dir, &spec).unwrap();
        assert!(text.contains("linear(x = x, weight = weight, bias = bias)"));
        assert!(dir.join("Manifest.json").is_file());
        assert!(dir.join("Data/com.apple.CoreML/model.mlmodel").is_file());
        assert_eq!(
            std::fs::metadata(dir.join("Data/com.apple.CoreML/weights/weight.bin"))
                .unwrap()
                .len()
                % 64,
            0
        );
        let manifest = std::fs::read_to_string(dir.join("Manifest.json")).unwrap();
        assert!(
            manifest.contains("rootModelIdentifier")
                && manifest.contains("com.apple.CoreML/model.mlmodel")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
