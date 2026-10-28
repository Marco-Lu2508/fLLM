//! Host mirrors of the CUDA kernels in `cuda/fp8_block.cu` and of the scale addressing of the grouped MoE GEMM (`cuda/moe.cu`) and the FP8
//! tensor-core GEMM (`cuda/fp8_gemm.cu`): the same index arithmetic and the same order of floating-point operations, run on the CPU against
//! the reference. They are how these kernels are checked without a GPU (a previous round found a self-recursive device function and a wrong
//! warp count this way); the tile constants are read from the kernel source, so a change of the kernel that breaks the mirror fails here.
use super::tests::{random_tensor, reference, Lcg};
use crate::compute::e4m3fn_to_f32;
use fllm_gguf_probe::fp8b_type_id;

/// `fb_e4m3` of fp8_block.cu.
fn cuda_e4m3(b: u32) -> f32 {
    let (e, m) = ((b >> 3) & 15, b & 7);
    let v = if e == 0 { m as f32 * (1.0 / 512.0) } else { f32::from_bits(((e + 120) << 23) | (m << 20)) };
    if b & 0x80 != 0 {
        -v
    } else {
        v
    }
}

#[test]
fn the_cuda_decoder_formula_matches_the_reference_for_every_finite_code() {
    for code in 0..=255u32 {
        if code & 0x7f == 0x7f {
            continue; // the NaN codes: the kernels decode them as +-480, they do not occur in checkpoints
        }
        assert_eq!(cuda_e4m3(code).to_bits(), e4m3fn_to_f32(code as u8).to_bits(), "code {code:#x}");
    }
}

/// One output row of `fp8blk_gemv_kernel` for one token: lane `l` takes runs `l, l + 32, ...` of 16 codes, expands them with the tile scale
/// (`lbc >= 4`: one scale per run; narrower tiles: per element), accumulates with fused multiply-adds in run order, then the xor butterfly.
fn mirror_gemv_row(codes: &[u8], scales: &[f32], x: &[f32], lbc: u32) -> f32 {
    let cols = codes.len();
    let runs = cols.div_ceil(16);
    let mut lanes = [0f32; 32];
    for (lane, acc_out) in lanes.iter_mut().enumerate() {
        let mut acc = 0f32;
        let mut run = lane;
        while run < runs {
            let k = run * 16;
            let mut w = [0f32; 16];
            for (i, wi) in w.iter_mut().enumerate() {
                *wi = if k + i < cols { cuda_e4m3(u32::from(codes[k + i])) } else { 0.0 };
            }
            if lbc >= 4 {
                let s = scales[k >> lbc];
                w.iter_mut().for_each(|v| *v *= s);
            } else {
                for (i, wi) in w.iter_mut().enumerate() {
                    if k + i < cols {
                        *wi *= scales[(k + i) >> lbc];
                    }
                }
            }
            for (i, &wi) in w.iter().enumerate() {
                let xv = if k + i < cols { x[k + i] } else { 0.0 };
                acc = wi.mul_add(xv, acc);
            }
            run += 32;
        }
        *acc_out = acc;
    }
    let mut offset = 16;
    while offset > 0 {
        let before = lanes;
        for (l, v) in lanes.iter_mut().enumerate() {
            *v = before[l] + before[l ^ offset];
        }
        offset >>= 1;
    }
    lanes[0]
}

#[test]
fn the_gemv_kernel_arithmetic_matches_the_reference_for_odd_shapes_and_tiles() {
    let mut rng = Lcg(21);
    for (br, bc, rows, cols) in [(128, 128, 5, 130), (128, 128, 129, 1000), (32, 32, 33, 65), (64, 8, 9, 100), (1, 128, 6, 17), (16, 1, 7, 52), (128, 128, 4, 16), (128, 128, 3, 4)] {
        let kind = fp8b_type_id(br, bc).unwrap();
        let (layout, bytes) = random_tensor(kind, cols, rows, 1, &mut rng);
        let input: Vec<f32> = (0..cols).map(|_| rng.unit()).collect();
        let (want, mag) = reference(kind, &bytes, &layout, &input, None);
        let lbc = bc.trailing_zeros();
        let tiles_c = layout.tile_cols();
        for r in 0..rows {
            // the kernel's addressing: codes at r * cols, the row of tile scales at (r >> lbr) * tiles_c of the table behind the codes
            let codes = &bytes[r * cols..(r + 1) * cols];
            let table: Vec<f32> = (0..tiles_c).map(|t| layout.scale(&bytes, (r >> br.trailing_zeros()) * tiles_c + t)).collect();
            let got = mirror_gemv_row(codes, &table, &input, lbc);
            let gamma = (cols + 40) as f64 * f64::from(f32::EPSILON) * 0.5;
            assert!((f64::from(got) - want[r]).abs() <= gamma * mag[r] + 1e-30, "tile {br}x{bc} {rows}x{cols} row {r}: {got} vs {}", want[r]);
        }
    }
}

/// The numbers behind `NAME = value` of a kernel source.
fn cuda_constants(source: &str, names: &[&str]) -> Vec<usize> {
    names
        .iter()
        .map(|name| {
            let at = source.find(&format!("{name} = ")).unwrap_or_else(|| panic!("{name} not found"));
            let text: String = source[at + name.len() + 3..].chars().take_while(char::is_ascii_digit).collect();
            text.parse().unwrap_or_else(|_| panic!("{name} is not a number"))
        })
        .collect()
}

#[test]
fn the_tensor_core_kernel_covers_its_tiles_exactly_once() {
    let source = include_str!("../../cuda/fp8_block.cu");
    let c = cuda_constants(source, &["FB_BM", "FB_BK", "FB_WM", "FB_BN_BIG", "FB_WN_BIG", "FB_BN_SMALL", "FB_WN_SMALL"]);
    let (bm, bk, wm_n) = (c[0], c[1], c[2]);
    // the two tile configurations of the launcher: (tokens per block, warps along the tokens, threads)
    for (bn, wn_n, threads) in [(c[3], c[4], 256), (c[5], c[6], 128)] {
        let nt = wm_n * wn_n * 32;
        assert_eq!(nt, threads, "threads of the {bn}-token tile");
        let (mi_n, ni_n) = (bm / wm_n / 16, bn / wn_n / 8);
        let (apt, bpt, rpk) = (bm * bk / 16 / nt, bn * bk / 16 / nt, bk / 16);
        // weight and activation runs: every (tile row, 16-wide chunk) of the shared-memory tile is written by exactly one thread
        for (count, per_thread) in [(bm, apt), (bn, bpt)] {
            assert_eq!(count * rpk, nt * per_thread, "threads times runs per thread is the tile");
            let mut seen = vec![0u8; count * rpk];
            for tid in 0..nt {
                for i in 0..per_thread {
                    let run = tid + i * nt;
                    seen[(run / rpk) * rpk + run % rpk] += 1;
                }
            }
            assert!(seen.iter().all(|&n| n == 1), "tile of {count} rows, {bn}-token configuration");
        }
        // the epilogue: every output element of the block tile is stored by exactly one (warp, lane, fragment, register)
        let mut seen = vec![0u8; bm * bn];
        for warp in 0..nt / 32 {
            let (wm, wn) = (warp % wm_n, warp / wm_n);
            for lane in 0..32 {
                let (g, t) = (lane >> 2, lane & 3);
                for mi in 0..mi_n {
                    for ni in 0..ni_n {
                        for e in 0..4 {
                            let m = wm * (bm / wm_n) + mi * 16 + g + (e >> 1) * 8;
                            let n = wn * (bn / wn_n) + ni * 8 + 2 * t + (e & 1);
                            seen[m * bn + n] += 1;
                        }
                    }
                }
            }
        }
        assert!(seen.iter().all(|&n| n == 1), "outputs of the {bn}-token configuration");
        // the shared memory of the configuration (two stages of weight and activation tiles, 72 halves per row) as the launcher computes it
        let smem = 2 * (bm + bn) * (bk + 8) * 2;
        assert!(smem <= if bn == c[3] { 73_728 } else { 65_536 }, "{bn}-token tile uses {smem} bytes");
    }
}

#[test]
fn the_moe_and_w8a8_scale_addressing_equals_the_layout() {
    let mut rng = Lcg(33);
    // grouped MoE (cuda/moe.cu, 128 x 128 tiles): the scale table starts rows * cols bytes into the expert buffer, row r reads the row of
    // tiles (r >> 7) * ((cols + 127) >> 7) and a run starting at k reads entry k >> 7
    for (rows, cols) in [(130usize, 528usize), (260, 144), (256, 256), (7, 64)] {
        let (layout, bytes) = random_tensor(0x8000_0021, cols, rows, 1, &mut rng);
        assert_eq!(layout.scale_offset(), rows * cols);
        for r in 0..rows {
            let base = (r >> 7) * ((cols + 127) >> 7);
            for k in (0..cols).step_by(16) {
                assert_eq!(layout.scale(&bytes, base + (k >> 7)).to_bits(), layout.scale(&bytes, layout.tile_index(0, r, k)).to_bits());
            }
        }
    }
    // fp8_gemm.cu with block weights: the scale of (row block nb, k step kt) is ws[nb * (cols / 128) + kt]
    for (rows, cols) in [(256usize, 128usize), (200, 640), (130, 512)] {
        let (layout, bytes) = random_tensor(0x8000_0021, cols, rows, 1, &mut rng);
        for r in 0..rows {
            for kt in 0..cols / 128 {
                let nb = r >> 7;
                assert_eq!(layout.scale(&bytes, nb * (cols >> 7) + kt).to_bits(), layout.scale(&bytes, layout.tile_index(0, r, kt * 128)).to_bits());
            }
        }
    }
}
