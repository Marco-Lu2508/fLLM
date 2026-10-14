// fLLM HIP backend: properties of the current AMD device and run-time switches for the kernels in this directory.
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include "../cuda/common.cuh"
#include "fllm_hip.h"

namespace {

bool starts_with(const std::string &s, const char *prefix) { return s.rfind(prefix, 0) == 0; }

FllmHipDevice query(int dev) {
    FllmHipDevice d{};
    d.arch = FLLM_HIP_ARCH_OTHER;
    d.wave = 64;
    d.lds_bytes = 64 * 1024;
    hipDeviceProp_t p;
    if (hipGetDeviceProperties(&p, dev) != hipSuccess) return d;
    snprintf(d.gcn, sizeof d.gcn, "%s", p.gcnArchName);
    const std::string arch = p.gcnArchName;
    d.cus = p.multiProcessorCount;
    if (p.sharedMemPerBlock > 0) d.lds_bytes = (int)p.sharedMemPerBlock;
    if (starts_with(arch, "gfx908") || starts_with(arch, "gfx90a") || starts_with(arch, "gfx940") || starts_with(arch, "gfx941") ||
        starts_with(arch, "gfx942") || starts_with(arch, "gfx950")) {
        d.arch = FLLM_HIP_ARCH_CDNA;
        d.wave = 64;
    } else if (starts_with(arch, "gfx11")) {
        d.arch = FLLM_HIP_ARCH_RDNA3;
        d.wave = 32;
    } else if (starts_with(arch, "gfx120")) {
        d.arch = FLLM_HIP_ARCH_RDNA4;
        d.wave = 32;
    } else if (starts_with(arch, "gfx10")) {
        d.wave = 32;
    }
    // v_dot2_f32_f16: gfx90a and newer CDNA (not gfx908), every gfx11 / gfx12
    d.dot2 = starts_with(arch, "gfx90a") || starts_with(arch, "gfx94") || starts_with(arch, "gfx950") || starts_with(arch, "gfx11") ||
             starts_with(arch, "gfx12");
    d.gfx1151 = starts_with(arch, "gfx1151");
    return d;
}

}  // namespace

const FllmHipDevice &fllm_hip_device() {
    static std::mutex mu;
    static FllmHipDevice cache[16];
    static bool have[16] = {false};
    static FllmHipDevice fallback{};
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 16) return fallback;
    std::lock_guard<std::mutex> lock(mu);
    if (!have[dev]) {
        cache[dev] = query(dev);
        have[dev] = true;
    }
    return cache[dev];
}

bool fllm_hip_enabled(const char *name) {
    char var[96];
    snprintf(var, sizeof var, "FLLM_HIP_NO_%s", name);
    const char *v = getenv(var);
    return !(v && *v && strcmp(v, "0") != 0);
}
