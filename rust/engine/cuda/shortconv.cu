// fLLM CUDA backend: LFM2 / LFM2.5 gated short convolution on the device.
//
// Mirrors `shortconv_forward` in rust/engine/src/rust_decoder.rs (the reference): the input projection yields [b | c | x] per token;
// current = b * x goes through a depthwise causal convolution (the last kernel - 1 inputs of every channel are the layer's state in
// the sequence's device buffer) and the result is multiplied by c. One thread per channel walks the tokens of a run. The output
// projection is an ordinary GEMM afterwards. Plain FP32 code, runs unchanged on Turing.
#include "common.cuh"
#include "fllm_cuda_internal.h"

// Rows of one sequence that are handled together (mirrors DeltaRun in backend/mod.rs).
struct ConvRun {
    unsigned long long state;
    int row0;
    int n;
    int reset;
    int pad;
};

#define CONV_MAX_KERNEL 8

int fllm_shortconv_supported(int hidden, int ksz) { return hidden > 0 && ksz >= 2 && ksz <= CONV_MAX_KERNEL; }

// grid (ceil(hidden / 128), runs), block 128
__global__ void __launch_bounds__(128) shortconv_kernel(const float *__restrict__ proj, const float *__restrict__ convw,
                                                        const float *__restrict__ convb, const ConvRun *__restrict__ runs,
                                                        float *__restrict__ out, int hidden, int ksz, size_t state_off) {
    const ConvRun run = runs[blockIdx.y];
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= hidden) return;
    float *hist = reinterpret_cast<float *>(reinterpret_cast<char *>(run.state) + state_off) + (size_t)c * (ksz - 1);
    float w[CONV_MAX_KERNEL], h[CONV_MAX_KERNEL - 1];
    for (int k = 0; k < ksz; k++) w[k] = convw[(size_t)c * ksz + k];
    const float bias = convb ? convb[c] : 0.f;
    for (int j = 0; j < ksz - 1; j++) h[j] = run.reset ? 0.f : hist[j];
    for (int t = 0; t < run.n; t++) {
        const float *row = proj + (size_t)(run.row0 + t) * 3 * hidden;
        const float cur = row[c] * row[2 * hidden + c];
        float value = bias;
        for (int j = 0; j < ksz - 1; j++) value += w[j] * h[j];
        value += w[ksz - 1] * cur;
        out[(size_t)(run.row0 + t) * hidden + c] = row[hidden + c] * value;
        for (int j = 0; j + 1 < ksz - 1; j++) h[j] = h[j + 1];
        h[ksz - 2] = cur;
    }
    for (int j = 0; j < ksz - 1; j++) hist[j] = h[j];
}

int fllm_shortconv_launch(cudaStream_t s, const void *runs, int nruns, int hidden, int ksz, size_t state_off, const float *proj,
                          const float *convw, const float *convb, float *out) {
    if (!fllm_shortconv_supported(hidden, ksz)) return fllm_set_error("shortconv: unsupported shape");
    if (nruns <= 0) return 0;
    shortconv_kernel<<<dim3((hidden + 127) / 128, nruns), 128, 0, s>>>(proj, convw, convb, static_cast<const ConvRun *>(runs), out, hidden, ksz,
                                                                       state_off);
    return fllm_check_launch("shortconv");
}
