//! Scalar models of the AArch64 instructions the integer kernels use. Test-only: they let the
//! dataflow of the NEON kernels (`model.rs`) run, and be checked, on any host.
//!
//! Each function states the architectural definition it models.

pub(crate) type I8x16 = [i8; 16];
pub(crate) type I32x4 = [i32; 4];
pub(crate) type F32x4 = [f32; 4];

/// `sdot Vd.4S, Vn.16B, Vm.16B`: lane `j` of the result is `acc[j]` plus the dot product of
/// bytes `4j..4j + 4` of both sources.
pub(crate) fn sdot(acc: I32x4, a: I8x16, b: I8x16) -> I32x4 {
    let mut out = acc;
    for (j, lane) in out.iter_mut().enumerate() {
        for k in 0..4 {
            *lane += i32::from(a[4 * j + k]) * i32::from(b[4 * j + k]);
        }
    }
    out
}

/// `smmla Vd.4S, Vn.16B, Vm.16B`: `Vn` holds a 2x8 matrix `A` and `Vm` an 8x2 matrix stored as
/// two 8-byte rows `B0`, `B1` (that is `B` transposed); the result is `acc + A * B` as the 2x2
/// matrix `[A0.B0, A0.B1, A1.B0, A1.B1]`.
pub(crate) fn smmla(acc: I32x4, a: I8x16, b: I8x16) -> I32x4 {
    let mut out = acc;
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..8 {
                out[2 * i + j] += i32::from(a[8 * i + k]) * i32::from(b[8 * j + k]);
            }
        }
    }
    out
}

/// `zip1 Vd.2D, Vn.2D, Vm.2D`: the low 64 bits of `a`, then the low 64 bits of `b`.
pub(crate) fn zip_lo(a: I8x16, b: I8x16) -> I8x16 {
    let mut out = [0; 16];
    out[..8].copy_from_slice(&a[..8]);
    out[8..].copy_from_slice(&b[..8]);
    out
}

/// `zip2 Vd.2D, Vn.2D, Vm.2D`: the high 64 bits of `a`, then the high 64 bits of `b`.
pub(crate) fn zip_hi(a: I8x16, b: I8x16) -> I8x16 {
    let mut out = [0; 16];
    out[..8].copy_from_slice(&a[8..]);
    out[8..].copy_from_slice(&b[8..]);
    out
}

/// `tbl Vd.16B, {Vn.16B}, Vm.16B` (`vqtbl1q_s8`): byte `i` is `table[index[i]]`, or zero when
/// the index is 16 or more.
pub(crate) fn tbl(table: I8x16, index: [u8; 16]) -> I8x16 {
    index.map(|i| table.get(usize::from(i)).copied().unwrap_or(0))
}

/// `faddp` twice (`vaddvq_f32`): `(l0 + l1) + (l2 + l3)`.
pub(crate) fn addv(v: F32x4) -> f32 {
    (v[0] + v[1]) + (v[2] + v[3])
}

/// `fmla Vd.4S, Vn.4S, Vm.4S` (`vfmaq_f32(acc, a, b)`): fused `acc + a * b` per lane.
pub(crate) fn fma(acc: F32x4, a: F32x4, b: F32x4) -> F32x4 {
    std::array::from_fn(|i| a[i].mul_add(b[i], acc[i]))
}

/// `scvtf Vd.4S, Vn.4S` (`vcvtq_f32_s32`).
pub(crate) fn cvt(v: I32x4) -> F32x4 {
    v.map(|x| x as f32)
}

pub(crate) fn dup(x: f32) -> F32x4 {
    [x; 4]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(offset: i32) -> I8x16 {
        std::array::from_fn(|i| ((i as i32 * 7 + offset).rem_euclid(201) - 100) as i8)
    }

    #[test]
    fn sdot_sums_groups_of_four_bytes() {
        let (a, b) = (ramp(3), ramp(40));
        let out = sdot([1, 2, 3, 4], a, b);
        for j in 0..4 {
            let want: i32 = (0..4)
                .map(|k| i32::from(a[4 * j + k]) * i32::from(b[4 * j + k]))
                .sum();
            assert_eq!(out[j], j as i32 + 1 + want);
        }
    }

    #[test]
    fn smmla_is_a_2x8_by_8x2_product() {
        let (a, b) = (ramp(5), ramp(77));
        let out = smmla([0; 4], a, b);
        let row = |m: &I8x16, r: usize| {
            m[8 * r..8 * r + 8]
                .iter()
                .map(|&v| i32::from(v))
                .collect::<Vec<_>>()
        };
        let dot = |x: Vec<i32>, y: Vec<i32>| x.iter().zip(&y).map(|(x, y)| x * y).sum::<i32>();
        assert_eq!(out[0], dot(row(&a, 0), row(&b, 0)));
        assert_eq!(out[1], dot(row(&a, 0), row(&b, 1)));
        assert_eq!(out[2], dot(row(&a, 1), row(&b, 0)));
        assert_eq!(out[3], dot(row(&a, 1), row(&b, 1)));
        // Accumulation adds.
        assert_eq!(
            smmla([10, 20, 30, 40], a, b),
            [out[0] + 10, out[1] + 20, out[2] + 30, out[3] + 40]
        );
    }

    #[test]
    fn zips_interleave_the_64_bit_halves() {
        let (a, b) = (ramp(0), ramp(60));
        let lo = zip_lo(a, b);
        let hi = zip_hi(a, b);
        assert_eq!(&lo[..8], &a[..8]);
        assert_eq!(&lo[8..], &b[..8]);
        assert_eq!(&hi[..8], &a[8..]);
        assert_eq!(&hi[8..], &b[8..]);
    }

    #[test]
    fn table_lookup_zeroes_out_of_range_indices() {
        let table: I8x16 = std::array::from_fn(|i| i as i8 * 3 - 20);
        let index = [0, 15, 16, 255, 7, 8, 9, 31, 1, 2, 3, 4, 5, 6, 14, 13];
        let out = tbl(table, index);
        assert_eq!(out[0], table[0]);
        assert_eq!(out[1], table[15]);
        assert_eq!(out[2], 0);
        assert_eq!(out[3], 0);
        assert_eq!(out[7], 0);
        assert_eq!(out[14], table[14]);
    }
}
