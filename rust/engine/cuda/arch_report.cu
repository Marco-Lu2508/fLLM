// fLLM CUDA backend: what this binary can do on a device, for `fllm-rs devices` (backend/cuda/hopper_arch.rs).
//
// build.rs compiles this file last and passes FLLM_BUILT, the description of the build:
//   archs=75.80....   plain targets (sm_75 cubin ... plus the PTX of the newest)
//   fp4a=120.121      architecture-specific FP4 targets (sm_120a, sm_121a)
//   sm100a=100.103    architecture-specific tcgen05 targets (sm_100a, sm_103a, sm_110a)
//   hopper=1          sm_90a kernels (wgmma, TMA) were built
//   cutlass, flashinfer   the headers were found
//   files=a+b         kernel files of the Hopper / Blackwell work that were built as real kernels (not as stubs)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common.cuh"
#include "fllm_cuda_internal.h"

#define FLLM_STR2(...) #__VA_ARGS__
#define FLLM_STR(x) FLLM_STR2(x)
#ifndef FLLM_BUILT
#define FLLM_BUILT unknown
#endif

extern "C" int fllm_cuda_nvfp4_sm100_native(int dev);

namespace {

std::vector<std::string> split(const std::string &s, char sep) {
    std::vector<std::string> out;
    size_t at = 0;
    while (at <= s.size()) {
        const size_t next = s.find(sep, at);
        out.push_back(s.substr(at, next == std::string::npos ? std::string::npos : next - at));
        if (next == std::string::npos) break;
        at = next + 1;
    }
    return out;
}
std::vector<int> numbers(const std::string &s) {
    std::vector<int> v;
    for (const std::string &p : split(s, '.'))
        if (!p.empty()) v.push_back(atoi(p.c_str()));
    return v;
}
bool contains(const std::vector<int> &v, int x) {
    for (int a : v)
        if (a == x) return true;
    return false;
}
struct Build {
    std::vector<int> archs, fp4a, sm100a;
    bool hopper = false, cutlass = false, flashinfer = false;
    std::vector<std::string> files;
    bool has(const char *f) const {
        for (const auto &x : files)
            if (x == f) return true;
        return false;
    }
};
Build parse_build() {
    Build b;
    for (const std::string &item : split(FLLM_STR(FLLM_BUILT), '/')) {
        if (item.rfind("archs=", 0) == 0) b.archs = numbers(item.substr(6));
        else if (item.rfind("fp4a=", 0) == 0) b.fp4a = numbers(item.substr(5));
        else if (item.rfind("sm100a=", 0) == 0) b.sm100a = numbers(item.substr(7));
        else if (item == "hopper=1") b.hopper = true;
        else if (item == "cutlass") b.cutlass = true;
        else if (item == "flashinfer") b.flashinfer = true;
        else if (item.rfind("files=", 0) == 0) b.files = split(item.substr(6), '+');
    }
    return b;
}
// compute capability as the plain number the build uses (9.0 -> 90, 10.3 -> 103)
std::string sm(int cc) { return "sm_" + std::to_string(cc); }

}  // namespace

// Writes two lines to `buf`: "kernels: ..." (which cubin of this binary the device loads) and "paths: a; b; c" (the fast paths available on it).
// Returns 0 on success.
extern "C" int fllm_cuda_fast_paths(int dev, char *buf, size_t len) {
    if (buf == nullptr || len == 0) return 1;
    buf[0] = 0;
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) {
        cudaGetLastError();
        return fllm_set_error("cannot query device %d", dev);
    }
    const int cc = p.major * 10 + p.minor;
    const Build b = parse_build();
    std::string kernels;
    // a plain cubin serves every device of the same major version with an equal or higher minor version
    int best = 0;
    for (int a : b.archs)
        if (a / 10 == cc / 10 && a <= cc && a > best) best = a;
    if (best == cc) kernels = sm(cc) + " cubin";
    else if (best > 0) kernels = sm(best) + " cubin (runs on " + sm(cc) + ")";
    else if (!b.archs.empty() && b.archs.back() <= cc) kernels = "PTX of " + sm(b.archs.back()) + " compiled at start-up (slow first run)";
    else kernels = "none for " + sm(cc) + ": rebuild with FLLM_CUDA_ARCHS including " + std::to_string(cc);
    std::vector<std::string> paths;
    if (best > 0) {
        if (cc >= 75) paths.push_back("Marlin 4-bit / FP8 / FP4 weights (W4A16, W8A16)");
        if (b.flashinfer && cc >= 75) paths.push_back("FlashInfer prompt attention (f16 KV)");
        if (b.has("fp8_gemm") && cc >= 89) paths.push_back("FP8 W8A8 tensor-core GEMM (mma.sync e4m3)");
        if (b.has("fp8_cutlass_sm90") && cc == 90 && b.hopper) paths.push_back("FP8 W8A8 CUTLASS wgmma GEMM");
        if (b.has("int8_gemm") && cc >= 75) paths.push_back("int8 W8A8 tensor-core GEMM");
        if (b.has("cublaslt_gemm") && cc >= 70) paths.push_back("cublasLt dense GEMM (loaded at run time when present)");
        if (b.has("fa3_prefill") && cc == 90 && b.hopper) paths.push_back("FA3 prompt attention (wgmma, TMA)");
        if (b.has("fi_attn") && cc >= 75) paths.push_back("FlashInfer prefill variants (sliding window, soft cap, sinks, ragged batches)");
        if (b.has("fi_decode") && cc >= 80) paths.push_back("tensor-core decode attention");
        if (b.has("cascade_attn")) paths.push_back("cascade attention (FLLM_CASCADE_ATTN=1)");
        if ((b.has("mla_latent") || b.has("mla_decode")) && cc >= 80) paths.push_back("latent-cache MLA (FLLM_MLA=latent)");
    }
    if (cc / 10 == 12 && contains(b.fp4a, cc)) paths.push_back(std::string("NVFP4 W4A4 tensor cores (") + (b.cutlass ? "mma.sync + CUTLASS Sm120)" : "mma.sync)"));
    else if (cc / 10 == 12) paths.push_back("NVFP4 W4A16 (Marlin) only: no sm_" + std::to_string(cc) + "a target in this binary");
    if (cc == 100 || cc == 103 || cc == 110) {
        if (contains(b.sm100a, cc)) paths.push_back(fllm_cuda_nvfp4_sm100_native(dev) ? "NVFP4 W4A4 tcgen05 (CUTLASS Sm100), on" : "NVFP4 W4A4 tcgen05 (CUTLASS Sm100), off: FLLM_NVFP4_SM100=1");
        else paths.push_back("NVFP4 W4A16 (Marlin) only: no sm_" + std::to_string(cc) + "a target in this binary");
    }
    std::string text = "kernels: " + kernels + "\npaths: ";
    for (size_t i = 0; i < paths.size(); i++) text += (i ? "; " : "") + paths[i];
    if (paths.empty()) text += "none";
    snprintf(buf, len, "%s", text.c_str());
    return 0;
}
