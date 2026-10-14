//! SVE `sdot` matvec for Q8_0 weights (decode), the `neon-sve` tier of `cpu_kernels::CpuTier`.
//!
//! Scope, honestly: this is one kernel, written as inline assembly because the SVE intrinsics
//! of `std::arch` are not stable. It targets CPUs whose SVE vectors are at least 256 bits wide
//! (AWS Graviton3 and 3E, Neoverse V1, Fujitsu A64FX), where one 32-byte Q8_0 block fits in one
//! vector: a block costs one predicated load, one `sdot`, one conversion and one fused
//! multiply-add per row. On 128-bit SVE cores (Graviton4, Neoverse N2/V2, NVIDIA Grace) the
//! NEON kernels are used; there is nothing an SVE vector of the same width could add.
//!
//! No speed-up is claimed. A decode matvec is limited by memory bandwidth, and the NEON kernel
//! (two `sdot`s per block) already reaches it on most cores; this kernel exists so the tier
//! can be measured on SVE hardware. Because it has never run, it is **opt-in**: the tier is only
//! selected with `FLLM_ARM_SVE=1` (or `FLLM_CPU_ISA=neon-sve`), and `selftest-cpu` enables it
//! to check it. Flip [`DEFAULT_ON`] once `selftest-cpu` passes and a benchmark shows a gain.
//!
//! Registers: `z4` holds the activation block (`ld1b` with a 32-lane predicate, the rest of the
//! vector is zero), `z0` a weight block, `z1` the integer dot products of the lanes that carry
//! data (the others stay zero), `z16..z19` the four row accumulators, `p0` the 32-byte
//! predicate and `p1` the all-true `.s` predicate. The f16 block scale is converted with the
//! base-architecture `fcvt` and broadcast. The assembler is told `.arch_extension sve`, so the
//! crate does not need the (unstable) `sve` target feature.
use std::sync::atomic::{AtomicBool, Ordering};

/// Whether the SVE tier needs no opt-in. Left off until the kernel has run on hardware.
pub(crate) const DEFAULT_ON: bool = false;

/// Bytes of a decode activation record (`cpu_kernels::XSingle`): 32 quants, the f32 scale at
/// byte 32, an i32 at 36. (`cpu_kernels` asserts the size; the unit test uses the constant.)
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const RECORD_BYTES: usize = 40;

static ALLOWED: AtomicBool = AtomicBool::new(false);

/// Opt in for this process (`selftest-cpu` does it to be able to check the tier).
pub(crate) fn allow() {
    ALLOWED.store(true, Ordering::Relaxed);
}

fn opted_in() -> bool {
    // The environment is read once: `usable()` runs on every tier query.
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let env = *ENV.get_or_init(|| {
        std::env::var_os("FLLM_ARM_SVE").is_some_and(|v| !v.is_empty() && v != "0")
            || std::env::var("FLLM_CPU_ISA").is_ok_and(|v| v.trim() == "neon-sve")
    });
    DEFAULT_ON || env || ALLOWED.load(Ordering::Relaxed)
}

/// The CPU can run the kernel (SVE with vectors of 256 bits or more) and the tier is opted in.
pub(crate) fn usable() -> bool {
    super::features::detect().sve_wide() && opted_in()
}

/// Four rows of Q8_0 against one activation row.
///
/// `rows[r]` points at the first int8 quant (block start + 2) of row `r`; blocks are 34 bytes
/// apart. `x` points at `nb` activation records of [`RECORD_BYTES`] bytes. The four dot
/// products are stored to `out[0..4]`.
///
/// # Safety
/// The CPU must support SVE with a vector of at least 32 bytes; every row holds `nb` blocks
/// (the 2-byte scale before the first quant included); `x` holds `nb` records; `out` has room
/// for four floats.
#[inline]
unsafe fn kernel4(rows: [*const u8; 4], x: *const u8, nb: usize, out: *mut f32) {
    let [r0, r1, r2, r3] = rows;
    std::arch::asm!(
        ".arch_extension sve",
        "ptrue p0.b, vl32",
        "ptrue p1.s",
        "mov z16.s, #0",
        "mov z17.s, #0",
        "mov z18.s, #0",
        "mov z19.s, #0",
        "cbz {nb}, 3f",
        "2:",
        "ld1b {{z4.b}}, p0/z, [{x}]",
        "ldr s5, [{x}, #32]",
        // row 0
        "ld1b {{z0.b}}, p0/z, [{r0}]",
        "ldur h6, [{r0}, #-2]",
        "mov z1.s, #0",
        "sdot z1.s, z0.b, z4.b",
        "fcvt s6, h6",
        "fmul s6, s6, s5",
        "scvtf z1.s, p1/m, z1.s",
        "dup z2.s, z6.s[0]",
        "fmla z16.s, p1/m, z1.s, z2.s",
        // row 1
        "ld1b {{z0.b}}, p0/z, [{r1}]",
        "ldur h6, [{r1}, #-2]",
        "mov z1.s, #0",
        "sdot z1.s, z0.b, z4.b",
        "fcvt s6, h6",
        "fmul s6, s6, s5",
        "scvtf z1.s, p1/m, z1.s",
        "dup z2.s, z6.s[0]",
        "fmla z17.s, p1/m, z1.s, z2.s",
        // row 2
        "ld1b {{z0.b}}, p0/z, [{r2}]",
        "ldur h6, [{r2}, #-2]",
        "mov z1.s, #0",
        "sdot z1.s, z0.b, z4.b",
        "fcvt s6, h6",
        "fmul s6, s6, s5",
        "scvtf z1.s, p1/m, z1.s",
        "dup z2.s, z6.s[0]",
        "fmla z18.s, p1/m, z1.s, z2.s",
        // row 3
        "ld1b {{z0.b}}, p0/z, [{r3}]",
        "ldur h6, [{r3}, #-2]",
        "mov z1.s, #0",
        "sdot z1.s, z0.b, z4.b",
        "fcvt s6, h6",
        "fmul s6, s6, s5",
        "scvtf z1.s, p1/m, z1.s",
        "dup z2.s, z6.s[0]",
        "fmla z19.s, p1/m, z1.s, z2.s",
        "add {r0}, {r0}, #34",
        "add {r1}, {r1}, #34",
        "add {r2}, {r2}, #34",
        "add {r3}, {r3}, #34",
        "add {x}, {x}, #40",
        "subs {nb}, {nb}, #1",
        "b.ne 2b",
        "3:",
        "faddv s24, p1, z16.s",
        "faddv s25, p1, z17.s",
        "faddv s26, p1, z18.s",
        "faddv s27, p1, z19.s",
        "str s24, [{out}]",
        "str s25, [{out}, #4]",
        "str s26, [{out}, #8]",
        "str s27, [{out}, #12]",
        r0 = inout(reg) r0 => _,
        r1 = inout(reg) r1 => _,
        r2 = inout(reg) r2 => _,
        r3 = inout(reg) r3 => _,
        x = inout(reg) x => _,
        nb = inout(reg) nb => _,
        out = in(reg) out,
        out("v0") _,
        out("v1") _,
        out("v2") _,
        out("v4") _,
        out("v5") _,
        out("v6") _,
        out("v16") _,
        out("v17") _,
        out("v18") _,
        out("v19") _,
        out("v24") _,
        out("v25") _,
        out("v26") _,
        out("v27") _,
        options(nostack),
    );
}

/// Rows `row0..row0 + out.len()` of `w` (Q8_0, `nb` blocks per row) against one activation row of
/// `nb` records at `x`.
///
/// # Safety
/// As [`kernel4`] for every row of the range; `w` is checked against the row range.
pub(crate) unsafe fn q8_gemv(w: &[u8], x: *const u8, nb: usize, row0: usize, out: &mut [f32]) {
    let stride = nb * 34;
    assert!(w.len() >= (row0 + out.len()) * stride);
    let mut done = 0;
    while done < out.len() {
        let n = (out.len() - done).min(4);
        // A short last panel repeats its final row; the extra results are dropped.
        let rows: [*const u8; 4] =
            std::array::from_fn(|r| w.as_ptr().add((row0 + done + r.min(n - 1)) * stride + 2));
        let mut y = [0.0f32; 4];
        kernel4(rows, x, nb, y.as_mut_ptr());
        out[done..done + n].copy_from_slice(&y[..n]);
        done += n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn sve_matvec_matches_the_scalar_dot_products() {
        if !super::super::features::detect().sve_wide() {
            return;
        }
        let nb = 7;
        let rows = 11;
        let mut rng = crate::cpu_selftest::Lcg(0x5ee);
        // Weights: Q8_0 blocks of random int8 values with moderate f16 scales.
        let mut w = Vec::new();
        for _ in 0..rows * nb {
            w.extend_from_slice(
                &f16::from_f32(0.01 + (rng.next() % 50) as f32 / 1000.0).to_le_bytes(),
            );
            w.extend((0..32).map(|_| (rng.next() % 255) as u8));
        }
        // Activations: 40-byte records (32 quants, f32 scale, pad).
        let mut x = Vec::new();
        let mut quants = Vec::new();
        let mut scales = Vec::new();
        for _ in 0..nb {
            let q: Vec<i8> = (0..32)
                .map(|_| ((rng.next() % 255) as i32 - 127) as i8)
                .collect();
            let d = 0.02 + (rng.next() % 40) as f32 / 1000.0;
            x.extend(q.iter().map(|&v| v as u8));
            x.extend_from_slice(&d.to_le_bytes());
            x.extend_from_slice(&0i32.to_le_bytes());
            quants.push(q);
            scales.push(d);
        }
        assert_eq!(x.len(), nb * RECORD_BYTES);
        for (row0, count) in [(0usize, 4usize), (0, 8), (3, 5), (9, 2), (10, 1)] {
            let mut have = vec![0.0f32; count];
            // SAFETY: SVE with >= 256-bit vectors was detected; `w` and `x` hold `rows` rows and
            // `nb` records.
            unsafe { q8_gemv(&w, x.as_ptr(), nb, row0, &mut have) };
            for (i, &y) in have.iter().enumerate() {
                let row = row0 + i;
                let (mut want, mut magnitude) = (0.0f64, 0.0f64);
                for kb in 0..nb {
                    let block = &w[(row * nb + kb) * 34..][..34];
                    let dw = f64::from(f16::from_le_bytes([block[0], block[1]]).to_f32());
                    let dot: i32 = block[2..]
                        .iter()
                        .zip(&quants[kb])
                        .map(|(&a, &b)| i32::from(a as i8) * i32::from(b))
                        .sum();
                    let term = dw * f64::from(scales[kb]) * f64::from(dot);
                    want += term;
                    magnitude += term.abs();
                }
                assert!(
                    (f64::from(y) - want).abs() <= 1e-4 * magnitude + 1e-6,
                    "row {row}: {y} vs {want}"
                );
            }
        }
    }
}
