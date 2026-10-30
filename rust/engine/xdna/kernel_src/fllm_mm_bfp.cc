//===- fllm_mm_bfp.cc -------------------------------------------*- C++ -*-===//
//
// Matrix multiply of a bf16 A tile with a bfp16ebs8 B tile (block floating point: 8 values share one exponent) on the native
// block-floating-point MAC of AIE2P (XDNA 2), accumulating into an f32 C tile.
//
// Derived from aie_kernels/linalg/mm_bfp_mixed.cc of the mlir-aie project (Copyright (C) 2025 Advanced Micro Devices, Inc.,
// Apache-2.0 with LLVM exceptions, https://github.com/Xilinx/mlir-aie). The only change: C is f32 instead of bf16, so that
// the partial sums of the K reduction (one call per k tile) are not rounded to bf16 between the calls. Layouts are those of the
// original: A in (8 x 8) micro-tiles, B^T as 72-byte vectors of eight 9-byte blocks ([n / 8][k / 8][n % 8] blocks, a block is
// an exponent byte and eight mantissa bytes along k), C in (8 x 8) micro-tiles.
//===----------------------------------------------------------------------===//
#include <aie_api/aie.hpp>

#ifndef DIM_M
#define DIM_M 64
#endif
#ifndef DIM_K
#define DIM_K 64
#endif
#ifndef DIM_N
#define DIM_N 64
#endif

#define FLLM_STR_(x) #x
#define FLLM_STR(x) FLLM_STR_(x)
#define FLLM_MIN_ITER(x) _Pragma(FLLM_STR(clang loop min_iteration_count(x)))

namespace {

template <unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s, unsigned t>
void matmul_2x2_bfp16_bf16_f32(const bfloat16 *__restrict pA, const bfp16ebs8 *__restrict pB, float *__restrict pC) {
  const unsigned sizeA = r * s;
  const unsigned sizeB = s * t;
  const unsigned sizeC = r * t;

  const bfloat16 *__restrict pArow = pA;
  float *__restrict pC1 = pC;
  unsigned j = 0;

  FLLM_MIN_ITER(4)
  for (unsigned tile = 0; tile < rowA * colB / 4; ++tile) {
    float *__restrict pC2 = pC1 + colB * sizeC;
    const bfloat16 *__restrict pA1 = pArow;
    const bfloat16 *__restrict pA2 = pArow + colA * sizeA;

    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pB1bfp16(pB);
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pB2bfp16(pB);
    pB1bfp16.seek(j * colA);
    pB2bfp16.seek((j + 1) * colA);

    aie::vector<bfloat16, sizeA> A0;
    aie::vector<bfloat16, sizeA> A1;
    aie::block_vector<bfp16ebs8, sizeB> B0;
    aie::block_vector<bfp16ebs8, sizeB> B1;

    aie::accum<accfloat, sizeC> accC00(aie::load_v<sizeC>(pC1));
    aie::accum<accfloat, sizeC> accC01(aie::load_v<sizeC>(pC1 + sizeC));
    aie::accum<accfloat, sizeC> accC10(aie::load_v<sizeC>(pC2));
    aie::accum<accfloat, sizeC> accC11(aie::load_v<sizeC>(pC2 + sizeC));

    aie::accum<accfloat, 64> accA0;
    aie::accum<accfloat, 64> accA1;

    for (unsigned i = 0; i < colA; ++i) {
      A0 = aie::load_v<sizeA>(pA1);
      pA1 += sizeA;
      A1 = aie::load_v<sizeA>(pA2);
      pA2 += sizeA;

      // convert A to bfp16 (the two paths are those of the original kernel)
      accA0 = A0;
      accA1 = mul_elem_64(A1, concat(broadcast_one_to_v32bfloat16(), broadcast_one_to_v32bfloat16()));

      B0 = pB1bfp16.pop();
      B1 = pB2bfp16.pop();

      accC00 = mac_8x8_8x8T(accA0.to_vector<bfp16ebs8>(), B0, accC00);
      accC01 = mac_8x8_8x8T(accA0.to_vector<bfp16ebs8>(), B1, accC01);
      accC10 = mac_8x8_8x8T(accA1.to_vector<bfp16ebs8>(), B0, accC10);
      accC11 = mac_8x8_8x8T(accA1.to_vector<bfp16ebs8>(), B1, accC11);
    }

    aie::store_v(pC1, accC00.template to_vector<float>());
    aie::store_v(pC1 + sizeC, accC01.template to_vector<float>());
    aie::store_v(pC2, accC10.template to_vector<float>());
    aie::store_v(pC2 + sizeC, accC11.template to_vector<float>());

    j += 2;
    const bool wrap = j == colB;
    j = wrap ? 0 : j;
    pC1 += wrap ? (colB + 2) * sizeC : 2 * sizeC;
    pArow += wrap ? 2 * colA * sizeA : 0;
  }
}

} // namespace

extern "C" {

void fllm_mm_bfp_f32(bfloat16 *__restrict pA, bfp16ebs8 *__restrict pB, float *__restrict pC) {
  constexpr int r = 8, s = 8, t = 8;
  static_assert(DIM_M % (2 * r) == 0 && DIM_K % s == 0 && DIM_N % (2 * t) == 0, "tile sizes");
  // the A conversion follows the core rounding mode: round to nearest even keeps the K reduction unbiased
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  matmul_2x2_bfp16_bf16_f32<DIM_M / r, DIM_K / s, DIM_N / t, r, s, t>(pA, pB, pC);
  aie::set_rounding(saved);
}

} // extern "C"
