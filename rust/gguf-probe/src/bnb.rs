//! The fLLM-specific tensor types that hold bitsandbytes weights as they are: NF4 and FP4 (4-bit codes into a 16-entry code book, one
//! absolute maximum per block) and LLM.int8() (int8 codes, one scale per output row).
//!
//! The ids sit in the private range (`0x8000_0040..0x8000_0044`; the block of ids `0x8000_0040` to `0x8000_004F` is bitsandbytes' own) next to the other fLLM types (Q4Z, FP8M, FP8B, EXL). Layouts, all little endian, every
//! row a whole number of blocks:
//!
//! | type | id | values per block | bytes per block | block |
//! |---|---|---|---|---|
//! | `NF4B64`  | `0x8000_0040` | 64  | 36 | `f32 absmax`, 32 code bytes |
//! | `NF4B128` | `0x8000_0041` | 128 | 68 | `f32 absmax`, 64 code bytes |
//! | `FP4B64`  | `0x8000_0042` | 64  | 36 | `f32 absmax`, 32 code bytes |
//! | `FP4B128` | `0x8000_0043` | 128 | 68 | `f32 absmax`, 64 code bytes |
//! | `INT8R`   | `0x8000_0044` | 64  | 68 | `f32 scale`, 64 `i8` codes |
//!
//! Code byte `j` of a 4-bit block holds the codes of elements `2j` (high nibble) and `2j + 1` (low nibble), the order bitsandbytes packs
//! them in; the value of an element is `code_book[nibble] * absmax`. The absmax of a double-quantized checkpoint is dequantized once
//! when the block is written (`nested_absmax`), so a block never needs the second level. An int8 element is `q * scale` with
//! `scale = SCB / 127`, `SCB` being the absolute maximum of its output row that LLM.int8() stores; the same scale sits in every block
//! of the row. Both layouts need rows of a multiple of the block size.

/// NF4, 64 values per block.
pub const NF4_B64: u32 = 0x8000_0040;
/// NF4, 128 values per block.
pub const NF4_B128: u32 = 0x8000_0041;
/// FP4, 64 values per block.
pub const FP4_B64: u32 = 0x8000_0042;
/// FP4, 128 values per block.
pub const FP4_B128: u32 = 0x8000_0043;
/// LLM.int8() weights, 64 values per block.
pub const INT8_ROW: u32 = 0x8000_0044;

/// What the codes of a block mean.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Family {
    /// 4-bit NormalFloat.
    Nf4,
    /// 4-bit floating point (sign, 2 exponent bits, 1 mantissa bit, as bitsandbytes defines the grid).
    Fp4,
    /// Signed 8-bit integers scaled per output row.
    Int8,
}

/// Block geometry of one of the types.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Layout {
    pub family: Family,
    /// Values per block.
    pub block: usize,
    /// Bytes per block, the `f32` scale included.
    pub bytes: usize,
}

const NF4_F64: [f64; 16] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
];

/// The NF4 code book of bitsandbytes (`create_normal_map`), identical in every checkpoint.
pub const NF4_CODE: [f32; 16] = {
    let mut code = [0f32; 16];
    let mut i = 0;
    while i < 16 {
        code[i] = NF4_F64[i] as f32;
        i += 1;
    }
    code
};

/// The FP4 code book of bitsandbytes: the values `0, 1/16, 8, 12, 4, 6, 2, 3` and their negatives, divided by 12.
pub const FP4_CODE: [f32; 16] = [
    0.0,
    0.0625 / 12.0,
    8.0 / 12.0,
    1.0,
    4.0 / 12.0,
    6.0 / 12.0,
    2.0 / 12.0,
    3.0 / 12.0,
    -0.0,
    -0.0625 / 12.0,
    -8.0 / 12.0,
    -1.0,
    -4.0 / 12.0,
    -6.0 / 12.0,
    -2.0 / 12.0,
    -3.0 / 12.0,
];

/// Every bitsandbytes type: (id, name, values per block, bytes per block).
pub const TYPES: [(u32, &str, usize, usize); 5] = [
    (NF4_B64, "nf4b64", 64, 36),
    (NF4_B128, "nf4b128", 128, 68),
    (FP4_B64, "fp4b64", 64, 36),
    (FP4_B128, "fp4b128", 128, 68),
    (INT8_ROW, "int8r", 64, 68),
];

/// Geometry of a bitsandbytes type (`None` for any other type id).
pub fn layout(type_id: u32) -> Option<Layout> {
    let family = match type_id {
        NF4_B64 | NF4_B128 => Family::Nf4,
        FP4_B64 | FP4_B128 => Family::Fp4,
        INT8_ROW => Family::Int8,
        _ => return None,
    };
    let (_, _, block, bytes) = TYPES.iter().copied().find(|t| t.0 == type_id)?;
    Some(Layout { family, block, bytes })
}

/// The type id of `family` with `block` values per block (`None` when there is no such type).
pub fn type_id(family: Family, block: usize) -> Option<u32> {
    match (family, block) {
        (Family::Nf4, 64) => Some(NF4_B64),
        (Family::Nf4, 128) => Some(NF4_B128),
        (Family::Fp4, 64) => Some(FP4_B64),
        (Family::Fp4, 128) => Some(FP4_B128),
        (Family::Int8, 64) => Some(INT8_ROW),
        _ => None,
    }
}

/// The type called `name` (`NF4B64`, `int8r`, ...; case does not matter).
pub fn type_id_by_name(name: &str) -> Option<u32> {
    TYPES.iter().find(|t| t.1.eq_ignore_ascii_case(name)).map(|t| t.0)
}

/// Whether `type_id` is one of the bitsandbytes types.
pub fn is_bnb(type_id: u32) -> bool {
    layout(type_id).is_some()
}

/// The code book of a 4-bit family.
pub fn code_book(family: Family) -> Option<&'static [f32; 16]> {
    match family {
        Family::Nf4 => Some(&NF4_CODE),
        Family::Fp4 => Some(&FP4_CODE),
        Family::Int8 => None,
    }
}

/// The family whose code book is `code` exactly (a checkpoint's own `quant_map`), if any.
pub fn family_of_code(code: &[f32; 16]) -> Option<Family> {
    [Family::Nf4, Family::Fp4].into_iter().find(|&f| code_book(f).is_some_and(|c| c.iter().zip(code).all(|(a, b)| a.to_bits() == b.to_bits() || a == b)))
}

/// Bytes of a row of `cols` values (`None` when `cols` is not a multiple of the block size or the type is not a bitsandbytes type).
pub fn row_bytes(type_id: u32, cols: usize) -> Option<usize> {
    let l = layout(type_id)?;
    (cols > 0 && cols.is_multiple_of(l.block)).then(|| cols / l.block * l.bytes)
}

/// The scale of an int8 row from its stored absolute maximum.
pub fn int8_scale(scb: f32) -> f32 {
    scb / 127.0
}

/// Dequantizes a double-quantized absmax: `code[q] * nested_absmax[i / nested_block] + offset`, the arithmetic of
/// `bitsandbytes.functional.dequantize_blockwise` followed by the offset. `index` is the position in the absmax array.
pub fn nested_absmax(codes: &[u8], nested_code: &[f32], nested_absmax: &[f32], nested_block: usize, offset: f32, index: usize) -> f32 {
    nested_code[codes[index] as usize] * nested_absmax[index / nested_block] + offset
}

/// Appends one 4-bit block: the `f32` absmax and the code bytes (`codes.len()` is half the block size).
pub fn push_block4(dst: &mut Vec<u8>, absmax: f32, codes: &[u8]) {
    dst.extend_from_slice(&absmax.to_le_bytes());
    dst.extend_from_slice(codes);
}

/// Appends one int8 block: the `f32` row scale and the codes.
pub fn push_block8(dst: &mut Vec<u8>, scale: f32, codes: &[u8]) {
    dst.extend_from_slice(&scale.to_le_bytes());
    dst.extend_from_slice(codes);
}

/// Dequantizes whole blocks of `type_id` into `out` (`out.len()` values, a multiple of the block size).
pub fn dequantize_into(bytes: &[u8], type_id: u32, out: &mut [f32]) -> Result<(), String> {
    let l = layout(type_id).ok_or_else(|| format!("{type_id:#x} is not a bitsandbytes type"))?;
    if !out.len().is_multiple_of(l.block) || out.len() / l.block * l.bytes != bytes.len() {
        return Err(format!("invalid bitsandbytes tensor size: {} bytes for {} values", bytes.len(), out.len()));
    }
    for (block, dst) in bytes.chunks_exact(l.bytes).zip(out.chunks_exact_mut(l.block)) {
        let scale = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        let codes = &block[4..];
        match l.family {
            Family::Int8 => {
                for (slot, &q) in dst.iter_mut().zip(codes) {
                    *slot = f32::from(q as i8) * scale;
                }
            }
            family => {
                let book = code_book(family).ok_or("no code book")?;
                for (pair, &byte) in dst.as_chunks_mut::<2>().0.iter_mut().zip(codes) {
                    pair[0] = book[(byte >> 4) as usize] * scale;
                    pair[1] = book[(byte & 15) as usize] * scale;
                }
            }
        }
    }
    Ok(())
}

/// Dequantizes `count` values of `type_id`.
pub fn dequantize(bytes: &[u8], type_id: u32, count: usize) -> Result<Vec<f32>, String> {
    let mut out = vec![0f32; count];
    dequantize_into(bytes, type_id, &mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_agree_with_the_table() {
        for (id, _, block, bytes) in TYPES {
            let l = layout(id).unwrap();
            assert_eq!((l.block, l.bytes), (block, bytes));
            assert_eq!(l.bytes, 4 + if l.family == Family::Int8 { l.block } else { l.block / 2 });
            assert_eq!(type_id(l.family, l.block), Some(id));
            assert!(is_bnb(id));
        }
        assert!(!is_bnb(0x8000_0010));
        assert!(!is_bnb(12));
        assert_eq!(row_bytes(NF4_B64, 192), Some(3 * 36));
        assert_eq!(row_bytes(NF4_B128, 192), None);
        assert_eq!(row_bytes(INT8_ROW, 128), Some(2 * 68));
    }

    #[test]
    fn code_books_have_the_documented_shape() {
        // NF4: sorted, symmetric ends, an exact zero
        assert!(NF4_CODE.windows(2).all(|w| w[0] < w[1]));
        assert_eq!((NF4_CODE[0], NF4_CODE[7], NF4_CODE[15]), (-1.0, 0.0, 1.0));
        // FP4: sign bit 8, magnitudes 0, 1/192, 2/3, 1, 1/3, 1/2, 1/6, 1/4
        let magnitudes = [0.0, 0.0625 / 12.0, 8.0 / 12.0, 1.0, 4.0 / 12.0, 0.5, 2.0 / 12.0, 0.25];
        for (i, m) in magnitudes.iter().enumerate() {
            assert_eq!(FP4_CODE[i], *m);
            assert_eq!(FP4_CODE[i + 8], -*m);
        }
        assert_eq!(family_of_code(&NF4_CODE), Some(Family::Nf4));
        assert_eq!(family_of_code(&FP4_CODE), Some(Family::Fp4));
        let mut other = NF4_CODE;
        other[3] += 1e-3;
        assert_eq!(family_of_code(&other), None);
    }

    #[test]
    fn four_bit_blocks_decode_high_nibble_first() {
        let mut bytes = Vec::new();
        let codes: Vec<u8> = (0..32u8).map(|j| ((j % 16) << 4) | (15 - j % 16)).collect();
        push_block4(&mut bytes, 2.0, &codes);
        let values = dequantize(&bytes, NF4_B64, 64).unwrap();
        for j in 0..32usize {
            assert_eq!(values[2 * j], NF4_CODE[j % 16] * 2.0);
            assert_eq!(values[2 * j + 1], NF4_CODE[15 - j % 16] * 2.0);
        }
        let fp4 = dequantize(&bytes, FP4_B64, 64).unwrap();
        assert_eq!(fp4[2], FP4_CODE[1] * 2.0);
        // 128-value blocks
        let mut big = Vec::new();
        push_block4(&mut big, -0.5, &[0x7f; 64]);
        let values = dequantize(&big, NF4_B128, 128).unwrap();
        assert!(values.iter().step_by(2).all(|&v| v == 0.0));
        assert!(values.iter().skip(1).step_by(2).all(|&v| v == -0.5));
        assert!(dequantize(&big, NF4_B128, 64).is_err());
    }

    #[test]
    fn int8_blocks_use_the_row_scale() {
        let scale = int8_scale(3.81);
        assert_eq!(scale, 3.81f32 / 127.0);
        let mut bytes = Vec::new();
        let codes: Vec<u8> = (0..64i32).map(|i| (i * 4 - 128) as i8 as u8).collect();
        push_block8(&mut bytes, scale, &codes);
        let values = dequantize(&bytes, INT8_ROW, 64).unwrap();
        for (i, v) in values.iter().enumerate() {
            assert_eq!(*v, (i as i32 * 4 - 128) as f32 * scale);
        }
        assert_eq!(values[0], -128.0 * scale);
    }

    #[test]
    fn double_quantization_arithmetic() {
        // absmax[i] = nested_code[q[i]] * nested_absmax[i / block] + offset
        let nested_code: Vec<f32> = (0..256).map(|i| (i as f32 - 127.5) / 127.5).collect();
        let codes = [0u8, 255, 128, 127, 3];
        let nested = [0.5f32, 4.0];
        let offset = 0.25f32;
        let got: Vec<f32> = (0..codes.len()).map(|i| nested_absmax(&codes, &nested_code, &nested, 3, offset, i)).collect();
        let want = [
            nested_code[0] * 0.5 + 0.25,
            nested_code[255] * 0.5 + 0.25,
            nested_code[128] * 0.5 + 0.25,
            nested_code[127] * 4.0 + 0.25,
            nested_code[3] * 4.0 + 0.25,
        ];
        assert_eq!(got, want);
    }
}
