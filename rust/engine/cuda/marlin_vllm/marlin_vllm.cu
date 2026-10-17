/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
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
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */

// fLLM: Marlin GEMM host code, taken from vLLM (csrc/libtorch_stable/quantization/marlin/marlin.cu) with the PyTorch tensor glue
// replaced by the plain C entry point at the end of this file. Kernel choice and thread configuration are vLLM's.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "kernel.h"
#include <atomic>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>
#include <utility>

extern int fllm_opt(int id);
static const int kFllmOptStepCache = 0;  // FLLM_OPT_STEP_CACHE (fllm_cuda_internal.h): the host-work reductions

#undef STD_TORCH_CHECK
#define STD_TORCH_CHECK(cond, ...)     do { if (!(cond)) return -1; } while (0)

namespace marlin {

__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS){};

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128}};

thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128}};

typedef struct {
  int blocks_per_sm;
  thread_config_t tb_cfg;
} exec_config_t;

int get_scales_cache_size(thread_config_t const& th_config, int prob_m,
                          int prob_n, int prob_k, int num_bits, int group_size,
                          int stages) {
  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;

  // Get max scale groups per thread-block
  int tb_groups;
  if (group_size == -1) {
    tb_groups = 1;
  } else {
    tb_groups = div_ceil(tb_k, group_size);
  }

  int tb_scales = tb_groups * tb_n * 2;
  return tb_scales * stages;
}

int get_kernel_cache_size(thread_config_t const& th_config, int thread_m_blocks,
                          int prob_m, int prob_n, int prob_k, int num_bits,
                          int group_size, int has_zp, bool is_zp_float,
                          bool is_a_8bit, int stages) {
  int pack_factor = 32 / num_bits;

  // Get B size
  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;
  int tb_m = thread_m_blocks * 16;
  int sh_a_size = stages * (tb_m * tb_k) * (is_a_8bit ? 1 : 2);
  int sh_b_size = stages * (tb_k * tb_n / pack_factor) * 4;
  int sh_red_size = tb_m * (tb_n + 8) * 2;
  int sh_bias_size = tb_n * 2;
  int tmp_size =
      (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
  tmp_size = max(max(sh_b_size, sh_red_size), tmp_size);

  int sh_s_size = get_scales_cache_size(th_config, prob_m, prob_n, prob_k,
                                        num_bits, group_size, stages);
  int sh_zp_size = 0;
  if (has_zp) {
    if (is_zp_float)
      sh_zp_size = sh_s_size;
    else if (num_bits == 4)
      sh_zp_size = sh_s_size / 4;
    else if (num_bits == 8)
      sh_zp_size = sh_s_size / 2;
  }

  int total_size = tmp_size + sh_a_size + sh_s_size + sh_zp_size;

  return total_size;
}

bool is_valid_config(thread_config_t const& th_config, int thread_m_blocks,
                     int prob_m, int prob_n, int prob_k, int num_bits,
                     int group_size, int has_zp, bool is_zp_float,
                     bool is_a_8bit, int stages, int max_shared_mem) {
  // Sanity
  if (th_config.thread_k == -1 || th_config.thread_n == -1 ||
      th_config.num_threads == -1) {
    return false;
  }

  // Verify K/N are divisible by thread K/N
  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
    return false;
  }

  // Verify min for thread K/N
  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
    return false;
  }

  // num_threads must be at least 128 (= 4 warps)
  if (th_config.num_threads < 128) {
    return false;
  }

  // Check that pipeline fits into cache
  int cache_size = get_kernel_cache_size(
      th_config, thread_m_blocks, prob_m, prob_n, prob_k, num_bits, group_size,
      has_zp, is_zp_float, is_a_8bit, stages);
  return cache_size <= max_shared_mem;
}

MarlinFuncPtr get_marlin_kernel(const vllm::ScalarType a_type,
                                const vllm::ScalarType b_type,
                                const vllm::ScalarType c_type,
                                const vllm::ScalarType s_type,
                                int thread_m_blocks, int thread_n_blocks,
                                int thread_k_blocks, bool m_block_size_8,
                                bool has_zp, int group_blocks, int threads,
                                bool is_zp_float, int stages) {
  int num_bits = b_type.size_bits();
  auto kernel = MarlinDefault;

  #include "kernel_selector.h"

  return kernel;
}

exec_config_t determine_exec_config(
    const vllm::ScalarType& a_type, const vllm::ScalarType& b_type,
    const vllm::ScalarType& c_type, const vllm::ScalarType& s_type, int prob_m,
    int prob_n, int prob_k, int thread_m_blocks, bool m_block_size_8,
    int num_bits, int group_size, bool has_zp, bool is_zp_float, int is_a_8bit,
    int stages, int max_shared_mem, int sms) {
  exec_config_t exec_cfg = exec_config_t{1, thread_config_t{-1, -1, -1}};
  thread_config_t* thread_configs = thread_m_blocks > 1
                                        ? large_batch_thread_configs
                                        : small_batch_thread_configs;
  int thread_configs_size =
      thread_m_blocks > 1
          ? sizeof(large_batch_thread_configs) / sizeof(thread_config_t)
          : sizeof(small_batch_thread_configs) / sizeof(thread_config_t);

  for (int i = 0; i < thread_configs_size; i++) {
    thread_config_t th_config = thread_configs[i];

    if (!is_valid_config(th_config, thread_m_blocks, prob_m, prob_n, prob_k,
                         num_bits, group_size, has_zp, is_zp_float, is_a_8bit,
                         stages, max_shared_mem - 512)) {
      continue;
    }

    int cache_size = get_kernel_cache_size(
        th_config, thread_m_blocks, prob_m, prob_n, prob_k, num_bits,
        group_size, has_zp, is_zp_float, is_a_8bit, stages);

    int group_blocks = group_size == -1 ? -1 : group_size / 16;

    auto kernel = get_marlin_kernel(
        a_type, b_type, c_type, s_type, thread_m_blocks,
        th_config.thread_n / 16, th_config.thread_k / 16, m_block_size_8,
        has_zp, group_blocks, th_config.num_threads, is_zp_float, stages);

    if (kernel == MarlinDefault) continue;

    return {1, th_config};
  }

  return exec_cfg;
}

// fLLM: FLLM_MARLIN_CFG=K,N,T (thread_k, thread_n, number of threads, e.g. 128,64,128) makes every call try that thread-block
// configuration before vLLM's choice and use it where it is valid for the shape and a kernel for it was generated (a tuning aid for
// decode, where the GEMMs have a handful of rows). Unset, "vllm", "fllm" or "auto": vLLM's determine_exec_config, unchanged.
static thread_config_t fllm_forced_config() {
  static const thread_config_t cfg = [] {
    thread_config_t c{-1, -1, -1};
    const char* v = getenv("FLLM_MARLIN_CFG");
    int k = 0, n = 0, t = 0;
    if (v && sscanf(v, "%d,%d,%d", &k, &n, &t) == 3 && k > 0 && n > 0 && t > 0) {
      c = thread_config_t{k, n, t};
      fprintf(stderr, "[fllm-rs] Marlin: trying thread configuration %d,%d,%d first (FLLM_MARLIN_CFG)\n", k, n, t);
    }
    return c;
  }();
  return cfg;
}

// fLLM: host-side cost of a launch. vLLM sets the dynamic shared memory limit of the kernel and reads three device attributes before every
// launch (a prompt batch has about 700 launches per step); the values are the same every time, so they are cached (FLLM_PREFILL_STEP_CACHE).
static void device_attributes(int dev, int& max_shared_mem, int& major, int& minor) {
  static std::mutex mu;
  static int cache[16][3] = {};
  const bool cached = dev >= 0 && dev < 16 && fllm_opt(kFllmOptStepCache) != 0;
  if (cached) {
    std::lock_guard<std::mutex> lock(mu);
    if (cache[dev][0] != 0) {
      max_shared_mem = cache[dev][0];
      major = cache[dev][1];
      minor = cache[dev][2];
      return;
    }
  }
  cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
  if (cached && max_shared_mem > 0) {
    std::lock_guard<std::mutex> lock(mu);
    cache[dev][0] = max_shared_mem;
    cache[dev][1] = major;
    cache[dev][2] = minor;
  }
}

static void set_max_dynamic_smem(const void* kernel, int dev, int bytes) {
  static std::mutex mu;
  static std::map<std::pair<const void*, int>, int> set;
  if (fllm_opt(kFllmOptStepCache) != 0) {
    std::lock_guard<std::mutex> lock(mu);
    auto it = set.find({kernel, dev});
    if (it != set.end() && it->second == bytes) return;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    set[{kernel, dev}] = bytes;
    return;
  }
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
}

int marlin_mm(const void* A, const void* B, void* C, void* C_tmp, void* b_bias,
               void* a_s, void* b_s, void* g_s, void* zp, int prob_m,
               int prob_n, int prob_k, int lda, void* workspace,
               vllm::ScalarType const& a_type, vllm::ScalarType const& b_type,
               vllm::ScalarType const& c_type, vllm::ScalarType const& s_type,
               bool has_bias, bool has_zp, int num_groups, int group_size,
               int dev, cudaStream_t stream, int thread_k_init,
               int thread_n_init, int sms, bool use_atomic_add,
               bool use_fp32_reduce, bool is_zp_float) {
  bool is_a_8bit = a_type.size_bits() == 8;
  STD_TORCH_CHECK(prob_m > 0 && prob_n > 0 && prob_k > 0, "Invalid MNK = [",
                  prob_m, ", ", prob_n, ", ", prob_k, "]");

  int group_blocks;
  if (group_size == -1) {
    group_blocks = -1;
  } else {
    group_blocks = group_size / 16;
    STD_TORCH_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                    " is not divisible by group_blocks = ", group_blocks);
  }

  int num_bits = b_type.size_bits();
  const int4* A_ptr = (const int4*)A;
  const int4* B_ptr = (const int4*)B;
  int4* C_ptr = (int4*)C;
  int4* C_tmp_ptr = (int4*)C_tmp;

  const int4* bias_ptr = (const int4*)b_bias;
  const float* a_s_ptr = (const float*)a_s;
  const int4* b_s_ptr = (const int4*)b_s;
  const float* g_s_ptr = (const float*)g_s;

  const int4* zp_ptr = (const int4*)zp;
  int* locks = (int*)workspace;

  // fLLM: the device attributes never change; they are queried once per device (three driver calls per GEMM otherwise)
  int max_shared_mem = 0;
  int major_capability, minor_capability;
  device_attributes(dev, max_shared_mem, major_capability, minor_capability);
  STD_TORCH_CHECK(max_shared_mem > 0);
  STD_TORCH_CHECK(major_capability * 10 + minor_capability >= 75,
                  "marlin kernel only support Turing or newer GPUs.");
  int stages = 4;
  if (major_capability == 7 && minor_capability == 5) {
    stages = 2;
    STD_TORCH_CHECK(a_type == vllm::kFloat16 || a_type == vllm::kS8,
                    "Turing only support FP16 or INT8 activation.");
  }
  if (a_type == vllm::kFE4M3fn) {
    STD_TORCH_CHECK(major_capability * 10 + minor_capability >= 89,
                    "FP8 only support Ada Lovelace or newer GPUs.");
    STD_TORCH_CHECK(
        major_capability * 10 + minor_capability == 89 ||
            major_capability == 12,
        "Marlin W4A8-FP8 only support SM89 or SM12x device (It is slower than "
        "Marlin W4A16 on other devices).");
  }

  int max_par = 16;
  if (prob_n <= 4096) max_par = 16 * 8;
  int max_shared_mem_new = max_shared_mem;
  int rest_m = prob_m;
  int max_thread_m_blocks = 4;
  while (rest_m) {
    int par_count = rest_m / (max_thread_m_blocks * 16);
    if (par_count > max_par) par_count = max_par;
    int prob_m_split =
        par_count > 0 ? (par_count * (max_thread_m_blocks * 16)) : rest_m;

    int thread_k = thread_k_init;
    int thread_n = thread_n_init;

    int thread_m_blocks = min(div_ceil(prob_m_split, 16), max_thread_m_blocks);
    int m_block_size_8 = prob_m_split <= 8 && a_type.size_bits() == 16;

    // Set thread config
    exec_config_t exec_cfg;
    thread_config_t thread_tfg;
    if (thread_k != -1 && thread_n != -1) {
      thread_tfg = thread_config_t{thread_k, thread_n, default_threads};
      exec_cfg = exec_config_t{1, thread_tfg};
      STD_TORCH_CHECK(prob_n % thread_n == 0, "prob_n = ", prob_n,
                      " is not divisible by thread_n = ", thread_n);
      STD_TORCH_CHECK(prob_k % thread_k == 0, "prob_k = ", prob_k,
                      " is not divisible by thread_k = ", thread_k);
    } else {
      // Auto config
      exec_cfg = determine_exec_config(
          a_type, b_type, c_type, s_type, prob_m_split, prob_n, prob_k,
          thread_m_blocks, m_block_size_8, num_bits, group_size, has_zp,
          is_zp_float, is_a_8bit, stages, max_shared_mem, sms);
      thread_tfg = exec_cfg.tb_cfg;
      if (thread_tfg.thread_n != -1) {
        if (prob_n / thread_tfg.thread_n *
                div_ceil(prob_m_split, thread_m_blocks * 16) * 4 <=
            sms) {
          if (is_valid_config({128, 64, 128}, thread_m_blocks, prob_m_split,
                              prob_n, prob_k, num_bits, group_size, has_zp,
                              is_zp_float, is_a_8bit, stages,
                              max_shared_mem_new)) {
            thread_tfg = {128, 64, 128};
            exec_cfg = {1, thread_tfg};
          }
        }
      }

      const thread_config_t forced = fllm_forced_config();
      if (forced.thread_k > 0 &&
          is_valid_config(forced, thread_m_blocks, prob_m_split, prob_n, prob_k,
                          num_bits, group_size, has_zp, is_zp_float, is_a_8bit,
                          stages, max_shared_mem - 512) &&
          get_marlin_kernel(a_type, b_type, c_type, s_type, thread_m_blocks,
                            forced.thread_n / 16, forced.thread_k / 16,
                            m_block_size_8, has_zp, group_blocks,
                            forced.num_threads, is_zp_float,
                            stages) != MarlinDefault) {
        thread_tfg = forced;
        exec_cfg = {1, thread_tfg};
      }

      if (thread_tfg.thread_k == -1 && max_thread_m_blocks > 1) {
        max_thread_m_blocks--;
        continue;
      }
    }

    int num_threads = thread_tfg.num_threads;
    thread_k = thread_tfg.thread_k;
    thread_n = thread_tfg.thread_n;
    int blocks = sms * exec_cfg.blocks_per_sm;
    if (exec_cfg.blocks_per_sm > 1)
      max_shared_mem_new = max_shared_mem / exec_cfg.blocks_per_sm - 1024;

    int thread_k_blocks = thread_k / 16;
    int thread_n_blocks = thread_n / 16;

    STD_TORCH_CHECK(
        is_valid_config(thread_tfg, thread_m_blocks, prob_m_split, prob_n,
                        prob_k, num_bits, group_size, has_zp, is_zp_float,
                        is_a_8bit, stages, max_shared_mem_new),
        "Invalid thread config: thread_m_blocks = ", thread_m_blocks,
        ", thread_k = ", thread_tfg.thread_k,
        ", thread_n = ", thread_tfg.thread_n,
        ", num_threads = ", thread_tfg.num_threads, " for MKN = [", prob_m,
        ", ", prob_k, ", ", prob_n, "] and num_bits = ", num_bits,
        ", prob_m_split = ", prob_m_split, ", group_size = ", group_size,
        ", has_zp = ", has_zp, ", is_zp_float = ", is_zp_float,
        ", stages = ", stages, ", max_shared_mem_new = ", max_shared_mem_new);

    auto kernel = get_marlin_kernel(
        a_type, b_type, c_type, s_type, thread_m_blocks, thread_n_blocks,
        thread_k_blocks, m_block_size_8, has_zp, group_blocks, num_threads,
        is_zp_float, stages);

    if (kernel == MarlinDefault) {
      STD_TORCH_CHECK(
          false, "Unsupported shapes: MNK = [", prob_m, ", ", prob_n, ", ",
          prob_k, "]", ", num_groups = ", num_groups,
          ", group_size = ", group_size, ", prob_m_split = ", prob_m_split,
          ", thread_m_blocks = ", thread_m_blocks,
          ", thread_n_blocks = ", thread_n_blocks,
          ", thread_k_blocks = ", thread_k_blocks,
          ", num_threads = ", num_threads, ", num_bits = ", num_bits);
    }

    set_max_dynamic_smem(reinterpret_cast<const void*>(kernel), dev, max_shared_mem_new);

    bool part_use_atomic_add =
        use_atomic_add && div_ceil(prob_m_split, 64) * prob_n <= 2048;

    // avoid ">>>" being formatted to "> > >"
    // clang-format off
    kernel<<<blocks, num_threads, max_shared_mem_new, stream>>>(
        A_ptr, B_ptr, C_ptr, C_tmp_ptr, bias_ptr, a_s_ptr, b_s_ptr, g_s_ptr, zp_ptr,
        prob_m_split, prob_n, prob_k, lda, locks, has_bias, part_use_atomic_add,
        use_fp32_reduce, max_shared_mem_new);
    // clang-format on

    bool is_a_8bit = a_type.size_bits() == 8;
    A_ptr += prob_m_split * (lda / (is_a_8bit ? 16 : 8));
    a_s_ptr += prob_m_split;
    C_ptr += prob_m_split * (prob_n / 8);
    rest_m -= prob_m_split;
  }
  return 0;
}

}  // namespace marlin

// Scratch sizes (bytes) the caller must provide. `workspace` must be zero when the first call starts; the kernels reset it.
extern "C" size_t fllm_vllm_marlin_workspace_bytes(int n) { return (size_t)(n / marlin::min_thread_n) * marlin::max_par * sizeof(int); }
extern "C" size_t fllm_vllm_marlin_ctmp_bytes(int m, int sms) {
    int mb = (m + 15) / 16 * 16;
    if (mb > 64) mb = 64;
    return (size_t)sms * mb * marlin::max_thread_n * sizeof(float);
}

// FP16 activations A [m, k] (row stride lda), FP16 output C [m, n]. `scales` and `zeros` in the Marlin layout (see
// marlin_vllm_prep.h). kind: 0 = AWQ (uint4 with zero points), 1 = GPTQ symmetric (uint4b8), 2 = uint8b128, 3 = NVFP4 (e2m1 weights,
// e4m3 scales per 16 values, `global_scale` is the per-tensor factor in the form of vLLM's nvfp4_marlin_process_global_scale).
// Returns 0 on success, -1 when a shape or a configuration is not supported.
static int fllm_marlin_gemm_plain(const void* A, const void* B, void* C, void* C_tmp, const void* scales, const void* zeros,
                                  const void* bias, const void* global_scale, int m, int n, int k, int lda, void* workspace, int kind,
                                  int group_size, int dev, int sms, cudaStream_t stream, int bf16 = 0) {
    const vllm::ScalarType b_type =
        kind == 0 ? vllm::kU4 : (kind == 1 ? vllm::kU4B8 : (kind == 3 ? vllm::kFE2M1f : (kind == 4 ? vllm::kFE4M3fn : vllm::kU8B128)));
    // bf16: BF16 activations, scales and result (the kernels exist only if generate_kernels.py was run with them, see FLLM_B_TYPES / A_C_TYPES there)
    const vllm::ScalarType ac_type = bf16 ? vllm::kBFloat16 : vllm::kFloat16;
    const vllm::ScalarType s_type = kind == 3 ? vllm::kFE4M3fn : ac_type;
    const bool has_zp = kind == 0;
    const int num_groups = group_size == -1 ? 1 : k / group_size;
    static const bool use_atomic_add = [] {
        const char* v = getenv("FLLM_MARLIN_USE_ATOMIC_ADD");
        if (!v) v = getenv("VLLM_MARLIN_USE_ATOMIC_ADD");
        return v && (!strcmp(v, "1") || !strcmp(v, "true") || !strcmp(v, "on"));
    }();
    return marlin::marlin_mm(A, B, C, C_tmp, (void*)bias, nullptr, (void*)scales, (void*)global_scale, (void*)zeros, m, n, k, lda,
                             workspace, ac_type, b_type, ac_type, s_type, bias != nullptr, has_zp, num_groups,
                             group_size, dev, stream, -1, -1, sms, use_atomic_add, C_tmp != nullptr, false);
}

// ---- small M ------------------------------------------------------------------------------------------------------------------------------
// FLLM_MARLIN_SMALL_M=auto|0|1 (default auto): products of 1 to 4 rows with AWQ / GPTQ weights run through the GEMV of fllm_marlin_gemv.cu instead
// of the Marlin GEMM. 0: never. 1: whenever the shape is supported. auto: the first call of every (device, n, k, m, kind, group) runs both,
// compares the results (the GEMV is dropped for the shape when they differ by more than 1 % of the largest value) and times them (the GEMV is kept when
// it is at least 5 % faster); the decision is cached. During CUDA graph capture a shape without a decision uses Marlin. The first call of a shape
// therefore costs a few launches and one stream synchronisation.
extern "C" int fllm_marlin_gemv_supported(int m, int n, int k, int kind, int group);
extern "C" int fllm_marlin_gemv(const void* A, const void* B, void* C, void* part, size_t part_bytes, void* tickets, const void* scales,
                                const void* zeros, int m, int n, int k, int lda, int kind, int group, int bf16, int sms, cudaStream_t stream);
extern "C" int fllm_marlin_gemv_diff(const void* a, const void* b, size_t count, float* host_out, cudaStream_t s, int bf16);

static std::atomic<int> g_small_m_override{-1};
// 0 off, 1 on, 2 auto
static int small_m_mode() {
  const int o = g_small_m_override.load(std::memory_order_relaxed);
  if (o >= 0) return o;
  static const int env = [] {
    const char* v = getenv("FLLM_MARLIN_SMALL_M");
    if (!v || !*v || !strcmp(v, "auto")) return 2;
    if (!strcmp(v, "0") || !strcmp(v, "off") || !strcmp(v, "false") || !strcmp(v, "no")) return 0;
    return 1;
  }();
  return env;
}
// selftest-gpu: mode 0, 1 or 2 (auto); negative returns to the environment
extern "C" void fllm_marlin_small_m_set(int mode) { g_small_m_override.store(mode > 2 ? 2 : mode, std::memory_order_relaxed); }

static int run_gemv(const void* A, const void* B, void* C, void* C_tmp, const void* scales, const void* zeros, int m, int n, int k, int lda,
                    void* workspace, int kind, int group, int sms, cudaStream_t stream, int bf16) {
  return fllm_marlin_gemv(A, B, C, C_tmp, C_tmp ? fllm_vllm_marlin_ctmp_bytes(m, sms) : 0, workspace, scales, zeros, m, n, k, lda, kind, group,
                          bf16, sms, stream);
}

extern "C" int fllm_vllm_marlin_gemm_t(const void* A, const void* B, void* C, void* C_tmp, const void* scales, const void* zeros,
                                      const void* bias, const void* global_scale, int m, int n, int k, int lda, void* workspace, int kind,
                                      int group_size, int dev, int sms, cudaStream_t stream, int bf16) {
  auto marlin = [&] {
    return fllm_marlin_gemm_plain(A, B, C, C_tmp, scales, zeros, bias, global_scale, m, n, k, lda, workspace, kind, group_size, dev, sms, stream, bf16);
  };
  auto gemv = [&](void* out) {
    return run_gemv(A, B, out, C_tmp, scales, zeros, m, n, k, lda, workspace, kind, group_size, sms, stream, bf16);
  };
  const int mode = (m <= 4 && bias == nullptr && (kind == 0 || kind == 1)) ? small_m_mode() : 0;
  if (mode == 0 || !fllm_marlin_gemv_supported(m, n, k, kind, group_size)) return marlin();
  if (mode == 1) return gemv(C) == 0 ? 0 : marlin();
  // auto
  static std::mutex mu;
  static std::map<std::tuple<int, int, int, int, int>, int> decided;  // 1 GEMV, 0 Marlin
  const auto key = std::make_tuple(dev, n, k, m * 4 + kind * 2 + bf16, group_size);
  int known = -1;
  {
    std::lock_guard<std::mutex> lock(mu);
    auto it = decided.find(key);
    if (it != decided.end()) known = it->second;
  }
  if (known == 1) return gemv(C) == 0 ? 0 : marlin();
  if (known == 0) return marlin();
  cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
  cudaStreamIsCapturing(stream, &cap);
  if (cap != cudaStreamCaptureStatusNone) return marlin();
  // decide: Marlin's result is the reference, the GEMV runs last when it is kept (C then holds its result)
  static void* ref = nullptr;
  static size_t ref_bytes = 0;
  const size_t cbytes = (size_t)m * n * 2;  // FP16 and BF16 are both two bytes
  int rc = marlin();
  if (rc != 0) return rc;
  bool keep = false, same = false;
  double t_marlin = 0, t_gemv = 0;
  float h[2] = {0.f, 0.f};
  void* reference = nullptr;
  {
    std::lock_guard<std::mutex> lock(mu);
    if (ref_bytes < cbytes) {
      cudaFree(ref);
      ref = nullptr;
      ref_bytes = 0;
      if (cudaMalloc(&ref, cbytes * 2) == cudaSuccess) ref_bytes = cbytes * 2;
      cudaGetLastError();
    }
    reference = ref_bytes >= cbytes ? ref : nullptr;
  }
  // `reference` receives the GEMV result; it is compared with Marlin's (in C)
  if (reference && gemv(reference) == 0 && fllm_marlin_gemv_diff(C, reference, (size_t)m * n, h, stream, bf16) == 0) {
    same = h[0] <= 0.01f * h[1] + 1e-6f;
    if (same) {
      cudaEvent_t e[3];
      for (auto& ev : e) cudaEventCreate(&ev);
      for (int i = 0; i < 2; i++) {
        marlin();
        gemv(C);
      }
      cudaEventRecord(e[0], stream);
      for (int i = 0; i < 8; i++) marlin();
      cudaEventRecord(e[1], stream);
      for (int i = 0; i < 8; i++) gemv(C);
      cudaEventRecord(e[2], stream);
      cudaEventSynchronize(e[2]);
      float a = 0.f, b = 0.f;
      cudaEventElapsedTime(&a, e[0], e[1]);
      cudaEventElapsedTime(&b, e[1], e[2]);
      t_marlin = a * 1000.0 / 8;
      t_gemv = b * 1000.0 / 8;
      keep = t_gemv < 0.95 * t_marlin;
      for (auto& ev : e) cudaEventDestroy(ev);
    }
    if (getenv("FLLM_CUDA_TUNE_LOG") || getenv("FLLM_MARLIN_SMALL_M_LOG"))
      fprintf(stderr, "[fllm-rs] Marlin small M: m=%d n=%d k=%d kind=%d group=%d: marlin %.1f us, gemv %.1f us, max diff %.3g of %.3g -> %s\n", m, n,
              k, kind, group_size, t_marlin, t_gemv, h[0], h[1], keep ? "gemv" : (same ? "marlin" : "marlin (results differ)"));
  }
  cudaGetLastError();
  {
    std::lock_guard<std::mutex> lock(mu);
    decided[key] = keep ? 1 : 0;
  }
  return keep ? 0 : marlin();
}

extern "C" int fllm_vllm_marlin_gemm(const void* A, const void* B, void* C, void* C_tmp, const void* scales, const void* zeros,
                                    const void* bias, const void* global_scale, int m, int n, int k, int lda, void* workspace, int kind,
                                    int group_size, int dev, int sms, cudaStream_t stream) {
  return fllm_vllm_marlin_gemm_t(A, B, C, C_tmp, scales, zeros, bias, global_scale, m, n, k, lda, workspace, kind, group_size, dev, sms, stream, 0);
}
