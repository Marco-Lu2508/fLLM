/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar (IST-DASLab)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Adapted from https://github.com/IST-DASLab/marlin and vLLM
 * (see THIRD_PARTY.md, NOTICE, licenses/vLLM-Apache-2.0.txt, and licenses/MARLIN-Apache-2.0.txt).
 */

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin
#endif

#include "marlin.cuh"
#include "marlin_dtypes.cuh"
#include "core/scalar_type.hpp"

#define MARLIN_KERNEL_PARAMS                                               \
  const int4 *__restrict__ A, const int4 *__restrict__ B,                  \
      int4 *__restrict__ C, int4 *__restrict__ C_tmp,                      \
      const int4 *__restrict__ b_bias_ptr,                                 \
      const float *__restrict__ a_scales_ptr,                              \
      const int4 *__restrict__ scales_ptr,                                 \
      const float *__restrict__ global_scale_ptr,                          \
      const int4 *__restrict__ zp_ptr, int prob_m, int prob_n, int prob_k, \
      int lda, int *locks, bool has_bias, bool use_atomic_add,             \
      bool use_fp32_reduce, int max_shared_mem

namespace MARLIN_NAMESPACE_NAME {
template <const vllm::ScalarTypeId a_type_id,  // A ScalarType id
          const vllm::ScalarTypeId b_type_id,  // B ScalarType id
          const vllm::ScalarTypeId c_type_id,  // C ScalarType id
          const vllm::ScalarTypeId s_type_id,  // B_SCALE ScalarType id
          const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const bool m_block_size_8,  // whether m_block_size == 8
                                      // only works when thread_m_blocks == 1
          const int stages,  // number of stages for the async global->shared
                             // fetch pipeline
          const int group_blocks,  // number of consecutive 16x16 blocks
                                   // with a separate quantization scale
          const bool is_zp_float   // is zero point of float16 type?
          >
__global__ void Marlin(MARLIN_KERNEL_PARAMS);

}
