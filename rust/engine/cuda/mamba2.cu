// fLLM CUDA backend: Mamba-2 selective state space layer (Nemotron-H, Granite-Hybrid, Bamba, Falcon-H1 ...) on the device.
//
// Mirrors `mamba_forward` in rust/engine/src/rust_decoder.rs (the reference). The input projection and the output projection run as
// ordinary GEMMs; three kernels per layer sit between them:
//   1. mamba_conv_kernel: depthwise causal convolution over the channels [x | B | C] with SiLU; the last (kernel - 1) inputs of every
//      channel live in the sequence's state buffer.
//   2. mamba_scan_kernel<N>: one block per (run, head), one thread per head channel p; the state row S[h][p][0..N) lives in registers
//      while the block walks the tokens of the run: S = S * exp(dt * A) + dt * x * B; y = S . C + D * x.
//   3. mamba_norm_kernel: y * silu(gate), RMS-normalised in `norm_groups` groups, times the norm weight.
// The recurrent state is FP32, [head][p][n], followed by the convolution history, one slice per layer in the sequence's state buffer
// (same bookkeeping as the DeltaNet layers, see rust_decoder.rs). Plain FP32 CUDA-core code (shared memory): runs unchanged on Turing.
#include "common.cuh"
#include "fllm_cuda_internal.h"

// Rows of one sequence that are handled together (mirrors DeltaRun in backend/mod.rs).
struct SsmRun {
    unsigned long long state;  // device address of the sequence's state buffer
    int row0;                  // first row of the run in the activations
    int n;                     // number of rows (consecutive positions)
    int reset;                 // 1: the sequence starts here, the old state is ignored
    int pad;
};

#define SSM_MAX_KERNEL 8

int fllm_mamba2_supported(int inner, int state, int heads, int groups, int ksz, int norm_groups) {
    if (heads <= 0 || groups <= 0 || inner % heads != 0 || heads % groups != 0 || norm_groups <= 0 || inner % norm_groups != 0) return 0;
    const int head_dim = inner / heads;
    if (head_dim > 256 || head_dim < 1) return 0;
    if (ksz < 2 || ksz > SSM_MAX_KERNEL) return 0;
    return state == 16 || state == 32 || state == 64 || state == 128;
}

// grid (ceil(conv_dim / 128), runs), block 128
__global__ void __launch_bounds__(128) mamba_conv_kernel(const float *__restrict__ proj, int width, int xbc_off, const float *__restrict__ convw,
                                                         const float *__restrict__ convb, const SsmRun *__restrict__ runs,
                                                         float *__restrict__ mixed, int conv_dim, int ksz, size_t state_off,
                                                         size_t rec_bytes) {
    const SsmRun run = runs[blockIdx.y];
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    float *hist = reinterpret_cast<float *>(reinterpret_cast<char *>(run.state) + state_off + rec_bytes) + (size_t)c * (ksz - 1);
    float w[SSM_MAX_KERNEL], h[SSM_MAX_KERNEL - 1];
    for (int k = 0; k < ksz; k++) w[k] = convw[(size_t)c * ksz + k];
    const float bias = convb ? convb[c] : 0.f;
    for (int j = 0; j < ksz - 1; j++) h[j] = run.reset ? 0.f : hist[j];
    for (int t = 0; t < run.n; t++) {
        const float x = proj[(size_t)(run.row0 + t) * width + xbc_off + c];
        float acc = w[ksz - 1] * x + bias;
        for (int j = 0; j < ksz - 1; j++) acc += w[j] * h[j];
        for (int j = 0; j + 1 < ksz - 1; j++) h[j] = h[j + 1];
        h[ksz - 2] = x;
        mixed[(size_t)(run.row0 + t) * conv_dim + c] = acc / (1.0f + expf(-acc));
    }
    for (int j = 0; j < ksz - 1; j++) hist[j] = h[j];
}

// grid (heads, runs), block head_dim threads
template <int N>
__global__ void __launch_bounds__(256) mamba_scan_kernel(const float *__restrict__ proj, int width, int dt_off, const float *__restrict__ mixed,
                                                         const float *__restrict__ dt_bias, const float *__restrict__ decay,
                                                         const float *__restrict__ skip, const SsmRun *__restrict__ runs,
                                                         float *__restrict__ y, int inner, int conv_dim, int heads, int groups,
                                                         float step_min, size_t state_off) {
    __shared__ float bsh[N], csh[N];
    const SsmRun run = runs[blockIdx.y];
    const int h = blockIdx.x, p = threadIdx.x;
    const int head_dim = blockDim.x;
    const int g = h / (heads / groups);
    float *rec = reinterpret_cast<float *>(reinterpret_cast<char *>(run.state) + state_off) + ((size_t)h * head_dim + p) * N;
    float cell[N];
    for (int n = 0; n < N; n++) cell[n] = run.reset ? 0.f : rec[n];
    const float a = decay[h], dtb = dt_bias[h], sk = skip[h];
    for (int t = 0; t < run.n; t++) {
        const size_t row = (size_t)(run.row0 + t);
        const float *mx = mixed + row * conv_dim;
        for (int i = p; i < N; i += head_dim) {
            bsh[i] = mx[inner + g * N + i];
            csh[i] = mx[inner + groups * N + g * N + i];
        }
        __syncthreads();
        const float raw = proj[row * width + dt_off + h] + dtb;
        float dt = raw > 20.f ? raw : log1pf(expf(raw));
        dt = fmaxf(dt, step_min);
        const float da = expf(dt * a);
        const float xp = mx[h * head_dim + p];
        float acc = 0.f;
#pragma unroll
        for (int n = 0; n < N; n++) {
            cell[n] = cell[n] * da + dt * xp * bsh[n];
            acc += cell[n] * csh[n];
        }
        y[row * inner + h * head_dim + p] = acc + sk * xp;
        __syncthreads();
    }
    for (int n = 0; n < N; n++) rec[n] = cell[n];
}

// grid (rows, norm_groups), block 256: y = rmsnorm_group(y * silu(gate)) * weight
__global__ void __launch_bounds__(256) mamba_norm_kernel(const float *__restrict__ proj, int width, float *__restrict__ y,
                                                         const float *__restrict__ weight, int inner, int group_size, float eps) {
    __shared__ float scratch[32];
    const size_t row = blockIdx.x;
    const int base = blockIdx.y * group_size;
    float* yr = y + row * inner;
    const float *gate = proj + row * width;
    float sum = 0.f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        const float g = gate[base + i];
        const float v = yr[base + i] * (g / (1.0f + expf(-g)));
        sum += v * v;
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
    sum = warp_sum(sum);
    if (lane == 0) scratch[warp] = sum;
    __syncthreads();
    float total = 0.f;
    for (int i = 0; i < warps; i++) total += scratch[i];
    const float inv = 1.0f / sqrtf(total / (float)group_size + eps);
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        const float g = gate[base + i];
        const float v = yr[base + i] * (g / (1.0f + expf(-g)));
        yr[base + i] = v * inv * weight[base + i];
    }
}

// depthwise causal convolution + SiLU over `channels` channels of rows of `width` floats starting at column `off` (shared with mamba1.cu)
int fllm_ssm_conv_launch(cudaStream_t s, const void *runs, int nruns, const float *proj, int width, int off, const float *convw, const float *convb,
                         float *mixed, int channels, int ksz, size_t state_off, size_t rec_bytes) {
    if (nruns <= 0) return 0;
    mamba_conv_kernel<<<dim3((channels + 127) / 128, nruns), 128, 0, s>>>(proj, width, off, convw, convb, static_cast<const SsmRun *>(runs), mixed,
                                                                         channels, ksz, state_off, rec_bytes);
    return fllm_check_launch("ssm_conv");
}

int fllm_mamba2_launch(cudaStream_t s, const void *runs, int nruns, int rows, int inner, int state, int heads, int groups, int ksz,
                       int norm_groups, float step_min, float eps, size_t state_off, size_t rec_bytes, const float *proj,
                       const float *convw, const float *convb, const float *dt_bias, const float *decay, const float *skip,
                       const float *norm, float *mixed, float *out) {
    if (!fllm_mamba2_supported(inner, state, heads, groups, ksz, norm_groups)) return fllm_set_error("mamba2: unsupported shape");
    if (nruns <= 0 || rows <= 0) return 0;
    const int conv_dim = inner + 2 * groups * state;
    const int width = inner + conv_dim + heads;
    const int head_dim = inner / heads;
    const SsmRun *r = static_cast<const SsmRun *>(runs);
    if (int e = fllm_ssm_conv_launch(s, runs, nruns, proj, width, inner, convw, convb, mixed, conv_dim, ksz, state_off, rec_bytes)) return e;
    const dim3 grid(heads, nruns);
    switch (state) {
        case 16:
            mamba_scan_kernel<16><<<grid, head_dim, 0, s>>>(proj, width, inner + conv_dim, mixed, dt_bias, decay, skip, r, out, inner, conv_dim, heads, groups, step_min, state_off);
            break;
        case 32:
            mamba_scan_kernel<32><<<grid, head_dim, 0, s>>>(proj, width, inner + conv_dim, mixed, dt_bias, decay, skip, r, out, inner, conv_dim, heads, groups, step_min, state_off);
            break;
        case 64:
            mamba_scan_kernel<64><<<grid, head_dim, 0, s>>>(proj, width, inner + conv_dim, mixed, dt_bias, decay, skip, r, out, inner, conv_dim, heads, groups, step_min, state_off);
            break;
        default:
            mamba_scan_kernel<128><<<grid, head_dim, 0, s>>>(proj, width, inner + conv_dim, mixed, dt_bias, decay, skip, r, out, inner, conv_dim, heads, groups, step_min, state_off);
            break;
    }
    if (int e = fllm_check_launch("mamba_scan")) return e;
    mamba_norm_kernel<<<dim3(rows, norm_groups), 256, 0, s>>>(proj, width, out, norm, inner, inner / norm_groups, eps);
    return fllm_check_launch("mamba_norm");
}
