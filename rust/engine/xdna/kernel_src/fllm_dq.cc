//===- fllm_dq.cc ----------------------------------------------*- C++ -*-===//
//
// Dequantize one 4-bit weight tile (k x n) to the bf16 B operand of the bf16
// matrix-multiply kernel, inside the compute tile of an AIE2P NPU (XDNA 2).
//
// The weight stays 4 bits in memory and in the DMA streams (0.625 byte per
// weight with group scales); this kernel expands it right before the multiply.
//
// The nibble extraction, the repeat of the eight per-column scales and the
// min + scale * code multiply-add follow q4nx_dequant.cc of the mlir-aie
// project (Copyright (C) 2026 Advanced Micro Devices, Inc., Apache-2.0 with LLVM
// exceptions, https://github.com/Xilinx/mlir-aie); the output here is plain bf16
// in the (s x t) = (8 x 8) micro-tile order the bf16 matmul kernel reads B in
// (k-block outer, n-block inner, k fastest-but-one, n fastest), not bfp16.
//
// Payload of one tile (little endian), K_TILE rows of k, N_TILE columns of n:
//   bf16 scales[K_TILE/GROUP][N_TILE]
//   bf16 mins  [K_TILE/GROUP][N_TILE]
//   nibbles    [N_TILE/16][K_TILE][16]   (element 2i in the low nibble of byte i)
// value(k, n) = mins[k/GROUP][n] + scales[k/GROUP][n] * code(k, n)
//
// Geometry arrives as -D flags: FLLM_DQ_K, FLLM_DQ_N, FLLM_DQ_GROUP.
//===----------------------------------------------------------------------===//
#define FLLM_STR_(x) #x
#define FLLM_STR(x) FLLM_STR_(x)
// loop hints as in mlir-aie's aie_kernel_utils.h (the Peano compiler defines __AIECC__)
#define AIE_PREPARE_FOR_PIPELINING
#define AIE_LOOP_RANGE(a, b)   _Pragma(FLLM_STR(clang loop min_iteration_count(a))) _Pragma(FLLM_STR(clang loop max_iteration_count(b)))
#include <aie_api/aie.hpp>
#include <stdint.h>

#if !defined(FLLM_DQ_K) || !defined(FLLM_DQ_N) || !defined(FLLM_DQ_GROUP)
#error "fllm_dq.cc needs FLLM_DQ_K, FLLM_DQ_N and FLLM_DQ_GROUP"
#endif

namespace {

constexpr int K_TILE = FLLM_DQ_K;
constexpr int N_TILE = FLLM_DQ_N;
constexpr int GROUP = FLLM_DQ_GROUP;
constexpr int SCALES = (K_TILE / GROUP) * N_TILE;
constexpr int N8 = N_TILE / 8;

static_assert(N_TILE % 16 == 0, "n tile must hold whole 16-column groups");
static_assert(K_TILE % GROUP == 0 && GROUP % 8 == 0, "groups must hold whole 8-row blocks");

// 8 bf16 repeated to fill 32 lanes (two 128-bit shuffles).
inline aie::vector<bfloat16, 32> rep8(const bfloat16 *p) {
  aie::vector<uint32_t, 16> v =
      aie::load_v<8>(p).template grow<32>().template cast_to<uint32_t>();
  v = ::shuffle(v, v, T128_2x4_lo);
  v = ::shuffle(v, v, T128_2x4_lo);
  return v.template cast_to<bfloat16>();
}

} // namespace

extern "C" {

void fllm_dq_q4(const uint8_t *__restrict payload, bfloat16 *__restrict out) {
#ifdef FLLM_DQ_SKIP
  return;
#endif
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);

  const bfloat16 *scales = (const bfloat16 *)payload;
  const bfloat16 *mins = scales + SCALES;
  const uint8_t *qs = (const uint8_t *)(mins + SCALES);

  constexpr int JJ = GROUP / 8; // 8-row blocks per group
  for (int g = 0; g < K_TILE / GROUP; g++) {
    const bfloat16 *srow = scales + g * N_TILE;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(N8, N8)
    for (int n8 = 0; n8 < N8; n8++) {
      // the 8 scales and minima of this column block are shared by the JJ row blocks of the group
      aie::vector<bfloat16, 32> sc = rep8(srow + n8 * 8);
      aie::vector<bfloat16, 32> mn = rep8(srow + n8 * 8 + SCALES);
      aie::vector<bfloat16, 64> sc2 = aie::concat(sc, sc);
      aie::vector<bfloat16, 64> mn2 = aie::concat(mn, mn);
      const unsigned mode = (unsigned)(T32_16x2_lo + (n8 & 1));
#pragma unroll
      for (int jj = 0; jj < JJ; jj++) {
        const int j = g * JJ + jj;
        // 8 k rows x 16 n = 128 nibbles = one 512-bit load; the even words hold
        // n 0..7 of each row, the odd words n 8..15.
        const uint32_t *q = (const uint32_t *)(qs + ((n8 >> 1) * K_TILE + j * 8) * 8);
        aie::vector<uint32_t, 16> w = aie::load_v<16>(q);
        aie::vector<uint32_t, 16> half = ::shuffle(w, w, mode);
        aie::vector<uint4, 64> qv = half.template extract<8>(0).template cast_to<uint4>();
        aie::accum<accfloat, 64> qf;
        qf.from_vector(aie::to_float(qv, 0));
        aie::vector<bfloat16, 64> qb = qf.template to_vector<bfloat16>();
        aie::accum<accfloat, 64> acc(mn2);
        acc = aie::mac(acc, sc2, qb);
        aie::store_v(out + (j * N8 + n8) * 64, acc.template to_vector<bfloat16>());
      }
    }
  }

  aie::set_rounding(saved);
}

} // extern "C"
