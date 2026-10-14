//! Pure host-side helpers of the Metal backend: float conversions for weights without a kernel and the naming rule that finds routed experts.
//!
//! No Metal types in here, so the unit tests run on any host (see `p07-harness`).

use crate::backend::QuantType;
use half::f16;

/// Largest magnitude the f16 expansion accepts (binary16 overflows at 65504; keep a margin so that rounding cannot produce an infinity).
const F16_LIMIT: f32 = 60000.0;

/// `values` as little-endian binary16, or `None` when a value is not finite or too large for binary16 (the caller keeps F32 then).
pub fn to_f16_bytes(values: &[f32]) -> Option<Vec<u8>> {
    if values.iter().any(|v| !v.is_finite() || v.abs() > F16_LIMIT) {
        return None;
    }
    Some(
        values
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect(),
    )
}

/// `values` as little-endian binary32.
pub fn to_f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// `values` as little-endian 32-bit unsigned integers.
pub fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Is `name` the tensor name of a routed expert projection (`blk.3.exp17.gate`)? The decoder names the weights it later hands to
/// `upload_pointer_table` like this; MXFP4 experts are kept in the raw GGUF block layout for the `moe_linear_mxfp4` kernel, every other MXFP4
/// weight is re-sorted into planes for the matrix kernels (`weights::pack_weight`).
pub fn is_expert_name(name: &str) -> bool {
    name.match_indices(".exp").any(|(at, _)| {
        let rest = &name[at + 4..];
        let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
        digits > 0 && rest[digits..].starts_with('.')
    })
}

/// Bytes of a raw (GGUF layout) MXFP4 tensor of `rows` x `cols` (`None`: `cols` is not a multiple of 32).
pub fn mxfp4_raw_len(rows: usize, cols: usize) -> Option<usize> {
    cols.is_multiple_of(32)
        .then(|| rows * (cols / 32) * 17)
}

/// Bytes of one row of a tensor in the layout of the GGUF file, for the formats whose rows are a whole number of blocks. `None` for the
/// formats that are not stored row by row (the Q4Z family with its separate scale planes, FP8M with its row scales behind the bytes) and for
/// unknown ones.
pub fn raw_row_bytes(quant: QuantType, cols: usize) -> Option<usize> {
    use QuantType as Q;
    match quant {
        Q::Q4Z32
        | Q::Q4Z64
        | Q::Q4Z128
        | Q::Q4ZP32
        | Q::Q4ZP64
        | Q::Q4ZP128
        | Q::Fp8Marlin
        | Q::Nvfp4Marlin
        | Q::Q4_K_Marlin
        | Q::Other(_) => None,
        q => {
            let block = q.block_size();
            cols.is_multiple_of(block)
                .then(|| cols / block * q.block_bytes())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_names_are_recognised() {
        assert!(is_expert_name("blk.3.exp17.gate"));
        assert!(is_expert_name("blk.0.exp0.down"));
        assert!(is_expert_name("model.layers.2.exp5.up"));
        assert!(!is_expert_name("blk.3.shared.gate"));
        assert!(!is_expert_name("blk.3.attn_q"));
        assert!(!is_expert_name("blk.3.exp.gate"), "no digits");
        assert!(!is_expert_name("blk.3.expert17.gate"), "not the decoder's naming");
        assert!(!is_expert_name("blk.3.exp17"), "no projection after the number");
        assert!(!is_expert_name("output_head"));
        // a later match is found even if an earlier ".exp" does not fit
        assert!(is_expert_name("blk.1.expx.exp4.gate"));
    }

    #[test]
    fn f16_expansion_checks_the_range() {
        assert!(to_f16_bytes(&[1.0, -2.5, 0.0]).is_some());
        assert!(to_f16_bytes(&[1.0e5]).is_none());
        assert!(to_f16_bytes(&[f32::NAN]).is_none());
        assert!(to_f16_bytes(&[f32::INFINITY]).is_none());
        let bytes = to_f16_bytes(&[1.5, -0.25]).unwrap();
        assert_eq!(bytes.len(), 4);
        assert_eq!(f16::from_le_bytes([bytes[0], bytes[1]]).to_f32(), 1.5);
        assert_eq!(f16::from_le_bytes([bytes[2], bytes[3]]).to_f32(), -0.25);
        // the margin below 65504 keeps rounding finite
        let edge = to_f16_bytes(&[F16_LIMIT]).unwrap();
        assert!(f16::from_le_bytes([edge[0], edge[1]]).is_finite());
    }

    #[test]
    fn word_packing_is_little_endian() {
        assert_eq!(u32_bytes(&[1, 0x0102_0304]), vec![1, 0, 0, 0, 4, 3, 2, 1]);
        assert_eq!(to_f32_bytes(&[1.0]), 1.0f32.to_le_bytes().to_vec());
        assert!(u32_bytes(&[]).is_empty());
    }

    #[test]
    fn raw_geometry() {
        assert_eq!(mxfp4_raw_len(3, 64), Some(3 * 2 * 17));
        assert_eq!(mxfp4_raw_len(3, 48), None);
        assert_eq!(raw_row_bytes(QuantType::Q4_0, 64), Some(2 * 18));
        assert_eq!(raw_row_bytes(QuantType::Q4_K, 512), Some(2 * 144));
        assert_eq!(raw_row_bytes(QuantType::F32, 10), Some(40));
        assert_eq!(raw_row_bytes(QuantType::F16, 10), Some(20));
        assert_eq!(raw_row_bytes(QuantType::Fp8E4m3, 10), Some(10));
        assert_eq!(raw_row_bytes(QuantType::NvFp4, 128), Some(2 * 36));
        assert_eq!(raw_row_bytes(QuantType::Mxfp4, 64), Some(2 * 17));
        assert_eq!(raw_row_bytes(QuantType::Q4_0, 40), None, "not whole blocks");
        assert_eq!(raw_row_bytes(QuantType::Q4Z32, 64), None);
        assert_eq!(raw_row_bytes(QuantType::Fp8Marlin, 64), None);
        assert_eq!(raw_row_bytes(QuantType::Other(99), 64), None);
    }
}
