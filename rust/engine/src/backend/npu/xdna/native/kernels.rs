//! The NPU programs of the native XDNA backend: one `.xclbin` and a few instruction streams, embedded in the binary.
//!
//! Source and provenance: the design is the whole-array GEMM of the `mlir-aie` project (IRON, `whole_array.py`, Apache-2.0 with LLVM
//! exceptions) compiled with `aiecc` and the Peano (llvm-aie) compiler for `npu2` (AIE2P, 8 columns x 4 rows of compute tiles): bf16
//! inputs, f32 accumulation and output, `B` stored as `[n][k]` (so a weight matrix `[out][in]` is used as it is), tile `m=32 k=64 n=64`.
//! `scripts/xdna-build-kernels.sh` regenerates these files; `docs/npu-xdna.md` has the exact versions and the measured speed.
//! They are not FastFlowLM's or AMD's proprietary binaries; they were built from open sources by this project.
//!
//! The `.xclbin` does not depend on the matrix shape; the shape lives in the instruction stream (`*.insts`, 32-bit words of NPU DMA
//! and control commands). One stream exists per `(m, k)` pair below, always for `n = 1024`: `m` is the number of rows of `A` (prompt
//! tokens) per call, `k` the depth, `n` the number of output columns (weight rows) per call.

/// The program image for the NPU (device-independent within `npu2`).
pub static XCLBIN: &[u8] = include_bytes!("../../../../../xdna/kernels/gemm_bf16_f32_8col.xclbin");

/// Output columns per call (and the granularity weight rows are padded to).
pub const TILE_N: usize = 1024;
/// Granularity the depth `k` of a weight is padded to.
pub const TILE_K: usize = 1024;

/// One instruction stream.
pub struct Variant {
    /// Rows of `A` per call.
    pub m: usize,
    /// Depth per call.
    pub k: usize,
    pub insts: &'static [u8],
}

pub static VARIANTS: [Variant; 8] = [
    Variant { m: 128, k: 1024, insts: include_bytes!("../../../../../xdna/kernels/m128_k1024_n1024.insts") },
    Variant { m: 128, k: 2048, insts: include_bytes!("../../../../../xdna/kernels/m128_k2048_n1024.insts") },
    Variant { m: 128, k: 3072, insts: include_bytes!("../../../../../xdna/kernels/m128_k3072_n1024.insts") },
    Variant { m: 128, k: 4096, insts: include_bytes!("../../../../../xdna/kernels/m128_k4096_n1024.insts") },
    Variant { m: 512, k: 1024, insts: include_bytes!("../../../../../xdna/kernels/m512_k1024_n1024.insts") },
    Variant { m: 512, k: 2048, insts: include_bytes!("../../../../../xdna/kernels/m512_k2048_n1024.insts") },
    Variant { m: 512, k: 3072, insts: include_bytes!("../../../../../xdna/kernels/m512_k3072_n1024.insts") },
    Variant { m: 512, k: 4096, insts: include_bytes!("../../../../../xdna/kernels/m512_k4096_n1024.insts") },
];

/// Index into [`VARIANTS`] for `(m, k)`.
pub fn variant_index(m: usize, k: usize) -> Option<usize> {
    VARIANTS.iter().position(|v| v.m == m && v.k == k)
}

/// The depth chunks a weight with `cols` columns is cut into: `(first column, chunk depth)`; the last chunk is zero padded.
/// Chunks are 4096 deep while more than 4096 columns remain; the last one is the smallest of 1024, 2048, 3072 or 4096 that covers the
/// rest. A depth of at most 4096 is therefore one NPU call per tile, with no partial sums to add on the CPU.
pub fn k_chunks(cols: usize) -> Vec<(usize, usize)> {
    let mut chunks = Vec::new();
    let mut at = 0;
    while at < cols {
        let rest = cols - at;
        let depth = if rest > 4 * TILE_K { 4 * TILE_K } else { rest.div_ceil(TILE_K) * TILE_K };
        chunks.push((at, depth));
        at += depth;
    }
    chunks
}

/// Row chunks of a prompt of `batch` rows: `(first row, rows to compute, variant rows m)`. Rows left over (fewer than `min_rows`) are
/// not covered: the caller computes them on the CPU.
pub fn row_chunks(batch: usize, min_rows: usize) -> Vec<(usize, usize, usize)> {
    let mut chunks = Vec::new();
    let mut at = 0;
    while batch - at >= 512 {
        chunks.push((at, 512, 512));
        at += 512;
    }
    let mut rest = batch - at;
    if rest >= min_rows.max(1) {
        // up to 4 calls of 128 rows are cheaper than one padded call of 512 (see docs/npu-xdna.md)
        while rest > 0 {
            let take = rest.min(128);
            chunks.push((at, take, 128));
            at += take;
            rest -= take;
        }
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_chunks_cover_the_depth_with_known_variants() {
        for cols in [1usize, 576, 1024, 1536, 2048, 3072, 4096, 5120, 8192, 11008, 14336] {
            let chunks = k_chunks(cols);
            let mut at = 0;
            for &(first, depth) in &chunks {
                assert_eq!(first, at);
                assert!(variant_index(128, depth).is_some() && variant_index(512, depth).is_some());
                at += depth;
            }
            assert!(at >= cols && at - cols < TILE_K, "{cols}");
        }
        assert_eq!(k_chunks(3072), vec![(0, 3072)]);
        assert_eq!(k_chunks(11008), vec![(0, 4096), (4096, 4096), (8192, 3072)]);
        assert_eq!(k_chunks(8192), vec![(0, 4096), (4096, 4096)]);
    }

    #[test]
    fn row_chunks_prefer_full_calls_and_leave_a_small_rest_to_the_cpu() {
        assert_eq!(row_chunks(1024, 96), vec![(0, 512, 512), (512, 512, 512)]);
        assert_eq!(row_chunks(600, 96), vec![(0, 512, 512)]);
        assert_eq!(row_chunks(600, 64), vec![(0, 512, 512), (512, 88, 128)]);
        assert_eq!(row_chunks(300, 96), vec![(0, 128, 128), (128, 128, 128), (256, 44, 128)]);
        assert!(row_chunks(50, 96).is_empty());
        let covered: usize = row_chunks(1337, 64).iter().map(|c| c.1).sum();
        assert_eq!(covered, 1337);
    }

    #[test]
    fn embedded_programs_are_present() {
        assert!(XCLBIN.len() > 10_000 && XCLBIN.starts_with(b"xclbin2"));
        for v in &VARIANTS {
            assert!(v.insts.len() > 1000 && v.insts.len() % 4 == 0);
        }
    }
}
