// fLLM HIP backend: what the host side knows about an AMD GPU from its name (gcnArchName, e.g. "gfx1100" or "gfx90a:sramecc+:xnack-").
//
// Plain C++ without any HIP include, so that it can be compiled and tested on any machine (rust/engine/hip/hip_arch_test.cpp). The data is
// hip/arch_table.csv; hip_arch_test.cpp checks that this table and the csv agree, and scripts/hip_arch_check.sh checks the csv against the
// macros the compiler really defines for every target (cuda/hip_compat.h). The Rust side (src/backend/hip_arch.rs) reads the same csv.
//
// Names that are not in the table (a chip newer than this source) get the conservative class: no matrix cores, no dot product, the wave
// size of their generation. The run-time probe in hip_device.cu can still promote such a device when its self-check passes.
#pragma once
#include <cstring>
#include <string>

namespace fllm_hip_arch {

enum Use { USE_OTHER = 0, USE_CDNA = 1, USE_RDNA3 = 2, USE_RDNA4 = 3 };  // the matrix-core kernel set fLLM runs; USE_OTHER = plain kernels
enum Dot4 { DOT4_NONE = 0, DOT4_SDOT4 = 1, DOT4_SUDOT4 = 2 };

struct Info {
    bool known;      // listed in the table
    Use use;
    int wave;        // 64 on GCN / CDNA, 32 on RDNA; 0 when nothing is known
    Dot4 dot4;
    bool dot2;       // FLLM_HIP_DOT2 of hip_compat.h
};

struct Row {
    const char *gfx;
    Use use;
    int wave;
    Dot4 dot4;
    bool dot2;
};

// the order is the order of arch_table.csv
inline const Row *table(int *count) {
    static const Row rows[] = {
        {"gfx900", USE_OTHER, 64, DOT4_NONE, false},   {"gfx902", USE_OTHER, 64, DOT4_NONE, false},   {"gfx904", USE_OTHER, 64, DOT4_NONE, false},
        {"gfx906", USE_OTHER, 64, DOT4_SDOT4, false},  {"gfx909", USE_OTHER, 64, DOT4_NONE, false},   {"gfx90c", USE_OTHER, 64, DOT4_NONE, false},
        {"gfx908", USE_CDNA, 64, DOT4_SDOT4, false},   {"gfx90a", USE_CDNA, 64, DOT4_SDOT4, true},    {"gfx940", USE_CDNA, 64, DOT4_SDOT4, true},
        {"gfx941", USE_CDNA, 64, DOT4_SDOT4, true},    {"gfx942", USE_CDNA, 64, DOT4_SDOT4, true},    {"gfx950", USE_CDNA, 64, DOT4_SDOT4, true},
        {"gfx1010", USE_OTHER, 32, DOT4_NONE, false},  {"gfx1011", USE_OTHER, 32, DOT4_SDOT4, false}, {"gfx1012", USE_OTHER, 32, DOT4_SDOT4, false},
        {"gfx1013", USE_OTHER, 32, DOT4_NONE, false},  {"gfx1030", USE_OTHER, 32, DOT4_SDOT4, false}, {"gfx1031", USE_OTHER, 32, DOT4_SDOT4, false},
        {"gfx1032", USE_OTHER, 32, DOT4_SDOT4, false}, {"gfx1033", USE_OTHER, 32, DOT4_SDOT4, false}, {"gfx1034", USE_OTHER, 32, DOT4_SDOT4, false},
        {"gfx1035", USE_OTHER, 32, DOT4_SDOT4, false}, {"gfx1036", USE_OTHER, 32, DOT4_SDOT4, false}, {"gfx1100", USE_RDNA3, 32, DOT4_SUDOT4, true},
        {"gfx1101", USE_RDNA3, 32, DOT4_SUDOT4, true}, {"gfx1102", USE_RDNA3, 32, DOT4_SUDOT4, true}, {"gfx1103", USE_RDNA3, 32, DOT4_SUDOT4, true},
        {"gfx1150", USE_RDNA3, 32, DOT4_SUDOT4, true}, {"gfx1151", USE_RDNA3, 32, DOT4_SUDOT4, true}, {"gfx1152", USE_RDNA3, 32, DOT4_SUDOT4, true},
        {"gfx1153", USE_RDNA3, 32, DOT4_SUDOT4, true}, {"gfx1154", USE_RDNA3, 32, DOT4_SUDOT4, true}, {"gfx1170", USE_OTHER, 32, DOT4_SUDOT4, true},
        {"gfx1171", USE_OTHER, 32, DOT4_SUDOT4, true}, {"gfx1172", USE_OTHER, 32, DOT4_SUDOT4, true}, {"gfx1200", USE_RDNA4, 32, DOT4_SUDOT4, true},
        {"gfx1201", USE_RDNA4, 32, DOT4_SUDOT4, true}, {"gfx1250", USE_OTHER, 32, DOT4_SUDOT4, false}, {"gfx1251", USE_OTHER, 32, DOT4_SUDOT4, false},
    };
    *count = (int)(sizeof rows / sizeof rows[0]);
    return rows;
}

// "gfx90a:sramecc+:xnack-" -> "gfx90a"
inline std::string base_name(const std::string &gcn) {
    const size_t colon = gcn.find(':');
    return colon == std::string::npos ? gcn : gcn.substr(0, colon);
}

inline Info classify(const std::string &gcn) {
    const std::string name = base_name(gcn);
    int n = 0;
    const Row *rows = table(&n);
    for (int i = 0; i < n; i++)
        if (name == rows[i].gfx) return Info{true, rows[i].use, rows[i].wave, rows[i].dot4, rows[i].dot2};
    // unknown: the conservative class
    int wave = 0;
    if (name.compare(0, 4, "gfx9") == 0) wave = 64;
    else if (name.compare(0, 4, "gfx1") == 0) wave = 32;
    return Info{false, USE_OTHER, wave, DOT4_NONE, false};
}

}  // namespace fllm_hip_arch
