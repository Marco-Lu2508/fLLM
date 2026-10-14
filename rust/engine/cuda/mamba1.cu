// fLLM CUDA backend: Mamba-1 selective state space layer (Mamba, Falcon-Mamba, Jamba) on the device.
//
// Mirrors `mamba1_forward` in rust/engine/src/rust_decoder.rs (the reference). The input projection, x_proj, dt_proj and the output
// projection run as ordinary GEMMs; the kernels in between:
//   1. the depthwise causal convolution with SiLU (fllm_ssm_conv_launch, mamba2.cu), history in the sequence's state buffer;
//   2. mamba1_prep_kernel: optional RMS normalisation of the dt / B / C parts of x_proj's output (Falcon-Mamba) and the compact copy of
//      the dt part that feeds dt_proj;
//   3. mamba1_scan_kernel<N>: one thread per channel keeps its N state values in registers while it walks the tokens of a run:
//      s = s * exp(step * A) + step * B * x;  y = (s . C + D * x) * silu(gate).
// State per layer and sequence: [inner][N] FP32, then the convolution history. Plain FP32 CUDA-core code, runs unchanged on Turing.
#include "common.cuh"
#include "fllm_cuda_internal.h"

// Rows of one sequence that are handled together (mirrors DeltaRun in backend/mod.rs).
struct SsmRun1 {
    unsigned long long state;
    int row0;
    int n;
    int reset;
    int pad;
};

int fllm_mamba1_supported(int inner, int state, int rank, int ksz) {
    return inner > 0 && rank > 0 && ksz >= 2 && ksz <= 8 && (state == 16 || state == 32);
}

// grid rows, block 128. split: rows x (rank + 2 state); dt_in: rows x rank. eps <= 0: no normalisation.
__global__ void __launch_bounds__(128) mamba1_prep_kernel(float *__restrict__ split, float *__restrict__ dt_in, int rank, int state, float eps) {
    __shared__ float scratch[32];
    const size_t row = blockIdx.x;
    float *r = split + row * (size_t)(rank + 2 * state);
    const int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int seg = 0; seg < 3; seg++) {
        const int off = seg == 0 ? 0 : (seg == 1 ? rank : rank + state);
        const int len = seg == 0 ? rank : state;
        float inv = 1.0f;
        if (eps > 0.f) {
            float sum = 0.f;
            for (int i = threadIdx.x; i < len; i += blockDim.x) sum += r[off + i] * r[off + i];
            sum = warp_sum(sum);
            __syncthreads();
            if (lane == 0) scratch[warp] = sum;
            __syncthreads();
            float total = 0.f;
            for (int i = 0; i < warps; i++) total += scratch[i];
            inv = 1.0f / sqrtf(total / (float)len + eps);
        }
        for (int i = threadIdx.x; i < len; i += blockDim.x) {
            const float v = r[off + i] * inv;
            r[off + i] = v;
            if (seg == 0) dt_in[row * rank + i] = v;
        }
        __syncthreads();
    }
}

// grid (ceil(inner / 128), runs), block 128
template <int N>
__global__ void __launch_bounds__(128) mamba1_scan_kernel(const float *__restrict__ split, int split_w, int rank, const float *__restrict__ dt,
                                                          const float *__restrict__ mixed, const float *__restrict__ proj,
                                                          const float *__restrict__ decay, const float *__restrict__ skip,
                                                          const SsmRun1 *__restrict__ runs, float *__restrict__ y, int inner,
                                                          size_t state_off) {
    const SsmRun1 run = runs[blockIdx.y];
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= inner) return;
    float *rec = reinterpret_cast<float *>(reinterpret_cast<char *>(run.state) + state_off) + (size_t)c * N;
    float s[N], a[N];
    for (int k = 0; k < N; k++) {
        s[k] = run.reset ? 0.f : rec[k];
        a[k] = decay[(size_t)c * N + k];
    }
    const float sk = skip[c];
    for (int t = 0; t < run.n; t++) {
        const size_t row = (size_t)(run.row0 + t);
        const float raw = dt[row * inner + c];
        const float step = raw > 20.f ? raw : log1pf(expf(raw));
        const float xc = mixed[row * inner + c];
        const float *b = split + row * split_w + rank;
        const float *cv = b + N;
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < N; k++) {
            s[k] = s[k] * expf(step * a[k]) + step * b[k] * xc;
            acc += s[k] * cv[k];
        }
        const float z = proj[row * 2 * (size_t)inner + inner + c];
        y[row * inner + c] = (acc + sk * xc) * (z / (1.0f + expf(-z)));
    }
    for (int k = 0; k < N; k++) rec[k] = s[k];
}

int fllm_mamba1_conv_launch(cudaStream_t s, const void *runs, int nruns, int inner, int ksz, size_t state_off, size_t rec_bytes, const float *proj,
                            const float *convw, const float *convb, float *mixed) {
    if (nruns <= 0) return 0;
    return fllm_ssm_conv_launch(s, runs, nruns, proj, 2 * inner, 0, convw, convb, mixed, inner, ksz, state_off, rec_bytes);
}

int fllm_mamba1_prep_launch(cudaStream_t s, float *split, float *dt_in, int rows, int rank, int state, float eps) {
    if (rows <= 0) return 0;
    mamba1_prep_kernel<<<rows, 128, 0, s>>>(split, dt_in, rank, state, eps);
    return fllm_check_launch("mamba1_prep");
}

int fllm_mamba1_scan_launch(cudaStream_t s, const void *runs, int nruns, int inner, int state, int rank, size_t state_off, const float *split,
                            const float *dt, const float *mixed, const float *proj, const float *decay, const float *skip, float *out) {
    if (nruns <= 0) return 0;
    const SsmRun1 *r = static_cast<const SsmRun1 *>(runs);
    const dim3 grid((inner + 127) / 128, nruns);
    if (state == 16) mamba1_scan_kernel<16><<<grid, 128, 0, s>>>(split, rank + 2 * state, rank, dt, mixed, proj, decay, skip, r, out, inner, state_off);
    else if (state == 32) mamba1_scan_kernel<32><<<grid, 128, 0, s>>>(split, rank + 2 * state, rank, dt, mixed, proj, decay, skip, r, out, inner, state_off);
    else return fllm_set_error("mamba1: unsupported state size %d", state);
    return fllm_check_launch("mamba1_scan");
}
