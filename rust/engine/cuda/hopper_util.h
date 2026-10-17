// Helpers shared by the kernel files of the Hopper / Blackwell work (fp8_*.cu, int8_gemm.cu, fa3_*.cu, fi_*.cu, cascade_attn.cu, mla_*.cu,
// nvfp4_cutlass_sm100.cu, cublaslt_gemm.cu).
#pragma once
#include <cstdlib>
#include <cstring>

// A switch with the values auto | 0 | 1 (also off / on / false / true): -1 = auto (unset, empty or "auto"), 0 = off, 1 = on.
static inline int fllm_env_tri(const char *name) {
    const char *v = getenv(name);
    if (v == nullptr || *v == 0 || strcmp(v, "auto") == 0) return -1;
    if (strcmp(v, "0") == 0 || strcmp(v, "off") == 0 || strcmp(v, "false") == 0) return 0;
    return 1;
}
// One of `n` names (index), `dflt` when unset or not one of them: FLLM_ATTN_BACKEND=auto|fa2|fa3 -> 0|1|2.
static inline int fllm_env_choice(const char *name, const char *const *names, int n, int dflt) {
    const char *v = getenv(name);
    if (v == nullptr || *v == 0) return dflt;
    for (int i = 0; i < n; i++)
        if (strcmp(v, names[i]) == 0) return i;
    return dflt;
}
// FLLM_SELFTEST_BENCH=1: the Hopper / Blackwell self-test groups add large-shape timing cases (each path timed against the one it replaces).
static inline bool fllm_selftest_bench() {
    const char *v = getenv("FLLM_SELFTEST_BENCH");
    return v != nullptr && *v != 0 && strcmp(v, "0") != 0;
}
