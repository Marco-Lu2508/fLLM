//! Self-test cases for MLX affine group-quantized weights (`MLXA` types, `backend/mlx.rs`, `cuda/mlx_affine.cu`).
//!
//! `quantized_linear` of every bit width (2, 3, 4, 5, 6, 8), group sizes 32, 64 and 128 and both scale formats (f16, bf16) against the f64
//! reference of the dequantized weights (the CPU reference `cpu_mlx` dequantizes the same bytes). Shapes have an odd number of groups per
//! row (partial warps), rows that fill no tile, batches on both sides of every kernel switch, wide rows (more chunks than a warp has
//! lanes) and a bias. On CUDA each device kernel also runs on its own (GEMV, FP32 tile, tensor cores) through `fllm_cuda_mlx_force`, so that the
//! batch-size switch cannot hide a kernel: the exact pass holds the FP32 kernels to the f32 rounding bound, the fast pass the tensor-core
//! kernel to the fp16 bound of the other fast kernels. Backends without a kernel for the format skip the checks (or run the decoder's f16
//! expansion, which the tolerance of the fast pass covers).
use super::*;

#[cfg(feature = "cuda")]
extern "C" {
    /// Force one MLX affine kernel (0 GEMV, 1 FP32 tile, 2 tensor cores) for the following linears; -1 chooses by batch size.
    fn fllm_cuda_mlx_force(path: i32) -> i32;
}

fn force(path: i32) {
    #[cfg(feature = "cuda")]
    // SAFETY: plain setter of a process-wide switch in the CUDA library.
    unsafe {
        fllm_cuda_mlx_force(path);
    }
    #[cfg(not(feature = "cuda"))]
    let _ = path;
}

fn mlx_name(kind: u32) -> String {
    match fllm_gguf_probe::mlx_affine(kind) {
        Some(m) => format!("mlx{}g{}{}", m.bits, m.group, if m.bf16 { "b" } else { "" }),
        None => format!("{kind:#x}"),
    }
}

/// (bits, group, bf16): every width at group 64, every group size, both scale formats.
const TYPES: [(u32, u32, bool); 12] = [
    (2, 64, false),
    (3, 64, false),
    (4, 64, false),
    (5, 64, false),
    (6, 64, false),
    (8, 64, false),
    (4, 32, true),
    (4, 128, false),
    (3, 32, true),
    (5, 128, true),
    (6, 32, false),
    (8, 128, true),
];

impl Tester<'_> {
    #[allow(clippy::too_many_arguments)]
    fn mlx_case(&mut self, kind: u32, rows: usize, cols: usize, batch: usize, bias: bool, loose: bool, note: &str) {
        let r = self.linear(kind, rows, cols, batch, bias).map(|mut o| {
            if loose {
                o.tol = o.tol.max(Self::FAST_TOL);
            }
            o
        });
        self.record(
            &format!("mlx linear {}", mlx_name(kind)),
            &format!("{rows}x{cols} batch {batch}{}{note}", if bias { " +bias" } else { "" }),
            r,
        );
    }

    /// `fast`: the second pass (tensor cores, loose bound); otherwise the exact pass.
    pub(super) fn mlx_cases(&mut self, fast: bool) {
        for (bits, group, bf16) in TYPES {
            let kind = fllm_gguf_probe::MlxAffine { bits, group, bf16 }.type_id();
            let narrow = group as usize * 3;
            // wide rows: more than 32 chunks, so that every lane of a warp takes more than one
            let wide = (2048 / group as usize + 1) * group as usize;
            if !fast {
                for batch in [1usize, 2, 5, 9] {
                    self.mlx_case(kind, 13, narrow, batch, batch == 5, false, "");
                }
                self.mlx_case(kind, 259, wide, 3, true, false, "");
                // the FP32 tile and the GEMV on their own, at batches the automatic choice sends elsewhere
                force(1);
                for batch in [1usize, 5, 70] {
                    self.mlx_case(kind, 77, narrow, batch, batch == 70, false, " (FP32 tile)");
                }
                force(0);
                self.mlx_case(kind, 13, wide, 9, false, false, " (GEMV)");
                force(-1);
            } else {
                for batch in [1usize, 4, 8, 9, 16, 40, 129] {
                    self.mlx_case(kind, 259, wide, batch, batch == 40 || batch == 129, true, "");
                }
                // tensor cores on their own: a tile with one token, ragged tiles in both directions
                force(2);
                for batch in [1usize, 5, 65, 130] {
                    self.mlx_case(kind, 131, narrow, batch, batch == 65, true, " (tensor cores)");
                }
                self.mlx_case(kind, 259, wide, 129, false, true, " (tensor cores)");
                force(0);
                self.mlx_case(kind, 130, wide, 12, false, true, " (GEMV)");
                force(-1);
            }
        }
    }
}
