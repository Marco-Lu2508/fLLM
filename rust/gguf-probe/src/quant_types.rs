//! GGML tensor type ids and block layouts from llama.cpp/ggml.
//!
//! Pinned reference: llama.cpp commit b1ff4ca23630ac7c5a275405353ddb5c486332c9,
//! `ggml/include/ggml.h`, `ggml/src/ggml-common.h`, and `ggml/src/ggml.c`.
//! The latest upstream master checked 2026-10-06 is 58cb9138e46180d45854cf45d4c1e25ad5f7e1c4;
//! its Q2_0 type id and block layout agree with the pinned decoder source.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecoderClass {
    /// Plain fixed-width numeric element type (one element per stored value).
    Scalar,
    /// Quantized type handled by the existing `compute::dequantize`.
    ExistingRust,
    /// Added in `engine::quant_extra::dequantize_extra`.
    ExtraRust,
    /// IQ block layout is known; values are handled by the independent IQ decoder.
    SeparateIq,
    /// Valid GGML layout, but computation/dequantization is not implemented here.
    RuntimeLayoutOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuantType {
    pub id: u32,
    pub name: &'static str,
    pub block_elements: usize,
    pub bytes_per_block: usize,
    pub decoder: DecoderClass,
}

macro_rules! ty {
    ($id:literal, $name:literal, $elems:literal, $bytes:literal, $class:ident) => {
        QuantType {
            id: $id,
            name: $name,
            block_elements: $elems,
            bytes_per_block: $bytes,
            decoder: DecoderClass::$class,
        }
    };
}

/// All currently valid enum ids in the pinned upstream revision.
pub const GGML_TYPES: &[QuantType] = &[
    ty!(0, "f32", 1, 4, Scalar),
    ty!(1, "f16", 1, 2, Scalar),
    ty!(2, "q4_0", 32, 18, ExistingRust),
    ty!(3, "q4_1", 32, 20, ExistingRust),
    ty!(6, "q5_0", 32, 22, ExistingRust),
    ty!(7, "q5_1", 32, 24, ExistingRust),
    ty!(8, "q8_0", 32, 34, ExistingRust),
    ty!(9, "q8_1", 32, 36, ExtraRust),
    ty!(10, "q2_K", 256, 84, ExtraRust),
    ty!(11, "q3_K", 256, 110, ExtraRust),
    ty!(12, "q4_K", 256, 144, ExistingRust),
    ty!(13, "q5_K", 256, 176, ExistingRust),
    ty!(14, "q6_K", 256, 210, ExistingRust),
    ty!(15, "q8_K", 256, 292, ExtraRust),
    ty!(16, "iq2_xxs", 256, 66, SeparateIq),
    ty!(17, "iq2_xs", 256, 74, SeparateIq),
    ty!(18, "iq3_xxs", 256, 98, SeparateIq),
    ty!(19, "iq1_s", 256, 50, SeparateIq),
    ty!(20, "iq4_nl", 32, 18, SeparateIq),
    ty!(21, "iq3_s", 256, 110, SeparateIq),
    ty!(22, "iq2_s", 256, 82, SeparateIq),
    ty!(23, "iq4_xs", 256, 136, SeparateIq),
    ty!(24, "i8", 1, 1, Scalar),
    ty!(25, "i16", 1, 2, Scalar),
    ty!(26, "i32", 1, 4, Scalar),
    ty!(27, "i64", 1, 8, Scalar),
    ty!(28, "f64", 1, 8, Scalar),
    ty!(29, "iq1_m", 256, 56, SeparateIq),
    ty!(30, "bf16", 1, 2, Scalar),
    ty!(34, "tq1_0", 256, 54, ExtraRust),
    ty!(35, "tq2_0", 256, 66, ExtraRust),
    ty!(39, "mxfp4", 32, 17, ExtraRust),
    ty!(40, "nvfp4", 64, 36, ExtraRust),
    ty!(41, "q1_0", 128, 18, ExtraRust),
    ty!(42, "q2_0", 64, 18, ExtraRust),
];

/// Removed/reserved slots retained by the stable enum so they are rejected
/// explicitly instead of being mistaken for unknown future types.
pub const REMOVED_GGML_TYPE_IDS: &[u32] = &[4, 5, 31, 32, 33, 36, 37, 38];

pub fn by_id(id: u32) -> Option<&'static QuantType> {
    GGML_TYPES.iter().find(|ty| ty.id == id)
}

pub fn is_removed_id(id: u32) -> bool {
    REMOVED_GGML_TYPE_IDS.contains(&id)
}

#[cfg(test)]
mod tests {
    use super::{DecoderClass, GGML_TYPES, by_id, is_removed_id};

    #[test]
    fn current_type_registry_is_contiguous_and_marks_removed_ids() {
        assert_eq!(GGML_TYPES.len(), 35);
        for pair in GGML_TYPES.windows(2) {
            assert!(pair[0].id < pair[1].id);
        }
        for id in [4, 5, 31, 32, 33, 36, 37, 38] {
            assert!(is_removed_id(id));
            assert!(by_id(id).is_none());
        }
        assert_eq!(by_id(16).unwrap().decoder, DecoderClass::SeparateIq);
        assert_eq!(by_id(40).unwrap().block_elements, 64);
        assert!(by_id(43).is_none());
    }
}
