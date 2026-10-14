// Activation prepass of the quantized linear kernels (package p06a): column gather for act-order GPTQ weights (Q4ZP*).
// With a permutation table `perm` (one u32 per column, DeviceWeight::perm) the weights are stored in permuted column order, so
// y[tok][c] = x[tok][perm[c]] is computed once and the ordinary Q4Z kernels read y instead of x.
//
// Verification level: L0 (clang-shim syntax check and CPU emulation only).

struct q_permute_params {
    uint batch;     // activation rows
    uint cols;      // columns
    uint x_stride;  // floats between two rows of x
    uint pad;
};

// Buffers: 0 x (f32), 1 y (f32, batch * cols, dense), 2 perm (u32 * cols), 3 q_permute_params.
// Grid: ceil(batch * cols / 256) threadgroups of 256 threads (one thread per output element).
kernel void fllm_q_permute_x(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                                                              device const uint *perm [[buffer(2)]],
                                                              constant q_permute_params &p [[buffer(3)]],
                                                              uint gid [[thread_position_in_grid]]) {
    if (gid >= p.batch * p.cols) return;
    const uint tok = gid / p.cols;
    const uint c = gid - tok * p.cols;
    y[gid] = x[ulong(tok) * p.x_stride + perm[c]];
}
