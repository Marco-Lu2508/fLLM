//! Scalar models of the 512-bit integer operations the wide kernels use, for tests that run on any
//! CPU. A kernel test built on them re-states the kernel's data flow (which bytes meet which
//! activations in which dword lane, which scale reaches which lane) with the same operation names;
//! it checks the lane arrangement and scale mapping against the scalar reference. It cannot check
//! that the real intrinsic behaves like its model: the models are written from the instruction
//! descriptions.

/// A 512-bit vector as bytes (little-endian lanes).
pub(crate) type B64 = [u8; 64];

pub(crate) fn zero() -> B64 {
    [0; 64]
}

/// `_mm512_set1_epi8`.
pub(crate) fn set1_epi8(v: u8) -> B64 {
    [v; 64]
}

/// 128-bit lane `l` (0..4).
pub(crate) fn lane(v: &B64, l: usize) -> [u8; 16] {
    v[16 * l..16 * l + 16].try_into().unwrap()
}

/// `join256`: two 256-bit halves as one vector.
pub(crate) fn join256(low: &[u8; 32], high: &[u8; 32]) -> B64 {
    let mut v = [0; 64];
    v[..32].copy_from_slice(low);
    v[32..].copy_from_slice(high);
    v
}

/// `_mm512_broadcast_i64x4`: the 256-bit value in both halves.
pub(crate) fn broadcast256(a: &[u8; 32]) -> B64 {
    join256(a, a)
}

/// `_mm512_broadcast_i32x4`: the 128-bit value in all four lanes.
pub(crate) fn broadcast128(a: &[u8; 16]) -> B64 {
    let mut v = [0; 64];
    for l in 0..4 {
        v[16 * l..16 * l + 16].copy_from_slice(a);
    }
    v
}

/// `_mm512_shuffle_i64x2::<IMM>(a, b)`: result lanes 0 and 1 from `a`, lanes 2 and 3 from `b`,
/// each chosen by two bits of `imm`.
pub(crate) fn shuffle_i64x2(a: &B64, b: &B64, imm: u8) -> B64 {
    let pick = |v: &B64, bits: u8| lane(v, usize::from(bits & 3));
    let mut out = [0; 64];
    out[..16].copy_from_slice(&pick(a, imm));
    out[16..32].copy_from_slice(&pick(a, imm >> 2));
    out[32..48].copy_from_slice(&pick(b, imm >> 4));
    out[48..].copy_from_slice(&pick(b, imm >> 6));
    out
}

/// `_mm512_shuffle_epi8(table, index)`: per 128-bit lane, byte `i` is `table[index[i] & 15]` of the
/// same lane, or zero if the index byte has its top bit set.
pub(crate) fn shuffle_epi8(table: &B64, index: &B64) -> B64 {
    let mut out = [0; 64];
    for i in 0..64 {
        let l = i / 16;
        out[i] = if index[i] & 0x80 != 0 {
            0
        } else {
            table[16 * l + usize::from(index[i] & 15)]
        };
    }
    out
}

/// `_mm512_srlv_epi16`: each 16-bit element shifted right by its own count (0 for counts > 15).
pub(crate) fn srlv_epi16(a: &B64, counts: &[u16; 32]) -> B64 {
    let mut out = [0; 64];
    for i in 0..32 {
        let v = u16::from_le_bytes([a[2 * i], a[2 * i + 1]]);
        let r = if counts[i] > 15 { 0 } else { v >> counts[i] };
        out[2 * i..2 * i + 2].copy_from_slice(&r.to_le_bytes());
    }
    out
}

/// 16-bit elements 0..16 get `low`, 16..32 get `high` (the `halves_epi16` helper).
pub(crate) fn halves_epi16(low: u16, high: u16) -> [u16; 32] {
    std::array::from_fn(|i| if i < 16 { low } else { high })
}

/// `_mm512_and_si512` with a byte constant.
pub(crate) fn and8(a: &B64, mask: u8) -> B64 {
    a.map(|b| b & mask)
}

/// `_mm512_or_si512`.
pub(crate) fn or(a: &B64, b: &B64) -> B64 {
    std::array::from_fn(|i| a[i] | b[i])
}

/// `_mm512_slli_epi16::<N>` on 0/1 bytes (no cross-byte carries for the values used).
pub(crate) fn slli_epi16(a: &B64, n: u32) -> B64 {
    let mut out = [0; 64];
    for i in 0..32 {
        let v = u16::from_le_bytes([a[2 * i], a[2 * i + 1]]) << n;
        out[2 * i..2 * i + 2].copy_from_slice(&v.to_le_bytes());
    }
    out
}

/// `_mm512_mask_blend_epi16(k, a, b)`: element `i` is `b[i]` when bit `i` of `k` is set.
pub(crate) fn blend_epi16(k: u32, a: &[i16; 32], b: &[i16; 32]) -> [i16; 32] {
    std::array::from_fn(|i| if k >> i & 1 == 1 { b[i] } else { a[i] })
}

/// `_mm512_mask_blend_ps(k, a, b)` on 16 floats.
pub(crate) fn blend_ps(k: u16, a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    std::array::from_fn(|i| if k >> i & 1 == 1 { b[i] } else { a[i] })
}

/// The bytes as 32 little-endian 16-bit words.
pub(crate) fn words(v: &B64) -> [i16; 32] {
    std::array::from_fn(|i| i16::from_le_bytes([v[2 * i], v[2 * i + 1]]))
}

/// `_mm512_maddubs_epi16(u, s)`: unsigned bytes of `u` times signed bytes of `s`, adjacent pairs
/// summed with signed saturation.
pub(crate) fn maddubs(u: &B64, s: &B64) -> [i16; 32] {
    std::array::from_fn(|i| {
        let sum = i32::from(u[2 * i]) * i32::from(s[2 * i] as i8)
            + i32::from(u[2 * i + 1]) * i32::from(s[2 * i + 1] as i8);
        sum.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
    })
}

/// `_mm512_madd_epi16(a, b)`: products of 16-bit elements, adjacent pairs summed.
pub(crate) fn madd16(a: &[i16; 32], b: &[i16; 32]) -> [i32; 16] {
    std::array::from_fn(|i| {
        i32::from(a[2 * i]) * i32::from(b[2 * i])
            + i32::from(a[2 * i + 1]) * i32::from(b[2 * i + 1])
    })
}

/// `_mm512_packs_epi32(a, b)`: per 128-bit lane the four dwords of `a` then those of `b`, as
/// signed-saturated 16-bit words.
pub(crate) fn packs32(a: &[i32; 16], b: &[i32; 16]) -> [i16; 32] {
    let sat = |v: i32| v.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    let mut out = [0i16; 32];
    for l in 0..4 {
        for j in 0..4 {
            out[8 * l + j] = sat(a[4 * l + j]);
            out[8 * l + 4 + j] = sat(b[4 * l + j]);
        }
    }
    out
}

/// `_mm512_dpbusd_epi32(src, a, b)`: four unsigned-by-signed byte products per dword added to
/// `src` (no saturation).
pub(crate) fn dpbusd(src: &[i32; 16], a: &B64, b: &B64) -> [i32; 16] {
    std::array::from_fn(|i| {
        let mut sum = src[i];
        for j in 0..4 {
            sum = sum.wrapping_add(i32::from(a[4 * i + j]) * i32::from(b[4 * i + j] as i8));
        }
        sum
    })
}

/// The BW dot with `init`: `madd(maddubs(u, s), ones) + init`.
pub(crate) fn dot_bw(init: &[i32; 16], u: &B64, s: &B64) -> [i32; 16] {
    let sums = madd16(&maddubs(u, s), &[1; 32]);
    std::array::from_fn(|i| sums[i].wrapping_add(init[i]))
}

/// `dot512_signed(w, s)`: the signed dot via sign transfer onto the activations.
pub(crate) fn dot_signed(w: &B64, s: &B64) -> [i32; 16] {
    let x: B64 = std::array::from_fn(|i| {
        if (w[i] as i8) < 0 {
            (s[i] as i8).wrapping_neg() as u8
        } else {
            s[i]
        }
    });
    let abs: B64 = std::array::from_fn(|i| (w[i] as i8).unsigned_abs());
    madd16(&maddubs(&abs, &x), &[1; 32])
}

/// `dot_signed_scaled512(w, x, scale)`: the sign-transfer `maddubs` with the scale words applied
/// in the `madd`.
pub(crate) fn dot_signed_scaled(w: &B64, s: &B64, scale: &[i16; 32]) -> [i32; 16] {
    let x: B64 = std::array::from_fn(|i| {
        if (w[i] as i8) < 0 {
            (s[i] as i8).wrapping_neg() as u8
        } else {
            s[i]
        }
    });
    let abs: B64 = std::array::from_fn(|i| (w[i] as i8).unsigned_abs());
    madd16(&maddubs(&abs, &x), scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffle_i64x2_with_0xd8_swaps_the_middle_lanes() {
        let a: B64 = std::array::from_fn(|i| (i / 16) as u8); // lane numbers 0 1 2 3
        let out = shuffle_i64x2(&a, &a, 0xD8);
        assert_eq!([out[0], out[16], out[32], out[48]], [0, 2, 1, 3]);
    }

    #[test]
    fn packs_and_madd_follow_the_lane_structure() {
        let a: [i32; 16] = std::array::from_fn(|i| i as i32);
        let b: [i32; 16] = std::array::from_fn(|i| 100 + i as i32);
        let p = packs32(&a, &b);
        // lane 1: words 8..16 hold a[4..8] then b[4..8]
        assert_eq!(&p[8..16], &[4, 5, 6, 7, 104, 105, 106, 107]);
        let scale: [i16; 32] = std::array::from_fn(|i| if i % 8 < 4 { 2 } else { 3 });
        let m = madd16(&p, &scale);
        // lane 1: dwords 4..8 = (4+5)*2, (6+7)*2, (104+105)*3, (106+107)*3
        assert_eq!(&m[4..8], &[18, 26, 627, 639]);
    }

    #[test]
    fn dot_models_agree_on_signed_inputs() {
        let w: B64 = std::array::from_fn(|i| ((i as i32 * 37) % 255 - 127) as i8 as u8);
        let s: B64 = std::array::from_fn(|i| ((i as i32 * 91) % 255 - 127) as i8 as u8);
        let signed = dot_signed(&w, &s);
        for i in 0..16 {
            let want: i32 = (0..4)
                .map(|j| i32::from(w[4 * i + j] as i8) * i32::from(s[4 * i + j] as i8))
                .sum();
            assert_eq!(signed[i], want);
        }
        // biased: (w + 128) unsigned against s, minus 128 * sum(s)
        let biased: B64 = std::array::from_fn(|i| ((w[i] as i8 as i32) + 128) as u8);
        let init: [i32; 16] = std::array::from_fn(|i| {
            -128 * (0..4).map(|j| i32::from(s[4 * i + j] as i8)).sum::<i32>()
        });
        assert_eq!(dpbusd(&init, &biased, &s), signed);
    }
}
