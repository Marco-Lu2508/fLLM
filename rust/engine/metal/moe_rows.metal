// FLLM-KERNEL: moe_gather_rows | constants: n,row_len,has_weights | buffers: 1 dst(w), 2 src(r), 3 idx(r), 4 weights(r) | threads: 256,1,1
// FLLM-KERNEL: moe_scatter_add_rows | constants: n,row_len,has_weights | buffers: 1 dst(rw), 2 src(r), 3 idx(r), 4 weights(r) | threads: 256,1,1
// (the constants struct is buffer 0; idx is int32, weights float; grid: [min(ceil(n*row_len/256), 65535), 1, 1] threadgroups)
// Row permutation helpers of the MoE token grouping (Stream::gather_rows and Stream::scatter_add_rows), grid-stride loops over n rows of row_len floats:
//   moe_gather_rows:       dst[i] = src[idx[i]] * weights[i]            (dst written, src read; weights optional: has_weights = 0 means 1)
//   moe_scatter_add_rows:  dst[idx[i]] += src[i] * weights[i]           (dst read and written; the rows of one call have distinct destinations
//                          (Stream contract), so a plain read-modify-write is enough)
// idx is int32 (checked >= 0 on the host); idx and weights are small host arrays that the stream puts into a buffer; with has_weights = 0 the weights
// binding is not read (bind any small buffer). Ports of shaders/vulkan/moe_gather_rows.comp and moe_scatter_add_rows.comp (package 3).

struct moe_rows_params {
    uint n;
    uint row_len;
    uint has_weights;
};

kernel void moe_gather_rows(
    constant moe_rows_params &p [[buffer(0)]],
    device float *dst [[buffer(1)]],
    device const float *src [[buffer(2)]],
    device const int *idx [[buffer(3)]],
    device const float *weights [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint grid_items [[threads_per_grid]]) {
    const ulong total = ulong(p.n) * p.row_len;
    for (ulong i = gid; i < total; i += grid_items) {
        const ulong r = i / p.row_len;
        const ulong c = i - r * p.row_len;
        const float v = src[ulong(idx[r]) * p.row_len + c];
        dst[i] = p.has_weights != 0u ? v * weights[r] : v;
    }
}

kernel void moe_scatter_add_rows(
    constant moe_rows_params &p [[buffer(0)]],
    device float *dst [[buffer(1)]],
    device const float *src [[buffer(2)]],
    device const int *idx [[buffer(3)]],
    device const float *weights [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint grid_items [[threads_per_grid]]) {
    const ulong total = ulong(p.n) * p.row_len;
    for (ulong i = gid; i < total; i += grid_items) {
        const ulong r = i / p.row_len;
        const ulong c = i - r * p.row_len;
        const float w = p.has_weights != 0u ? weights[r] : 1.0f;
        dst[ulong(idx[r]) * p.row_len + c] += w * src[i];
    }
}
