// FLLM-KERNEL: fllm_argmax_f32 | buffers: logits, out | params: o_argmax_params | grid: [rows, 1, 1] x [threads, 1, 1]
// Greedy sampling (Stream::argmax_rows): out[r] = index of the largest of the `vocab` logits of row r (int32); the lowest index wins a tie and a
// NaN never wins; an all -inf / all NaN row gives 0 (the CPU `sampling_logits` behaves the same). One threadgroup per row. The values are
// compared as integer sort keys (`o_order_key`), so fast math cannot change the result. threads: a power of two, 32..=512 (halving tree in
// threadgroup memory; no simd intrinsics, so the SIMD width of the device does not matter).
#include <metal_stdlib>
using namespace metal;

kernel void fllm_argmax_f32(
    device const float *logits [[buffer(1)]],
    device int *out [[buffer(2)]],
    constant o_argmax_params &p [[buffer(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
    threadgroup uint s_key[512];
    threadgroup uint s_idx[512];
    device const float *l = logits + ulong(row) * p.vocab;
    uint best_key = 0u;
    uint best_idx = 0u;
    for (uint i = tid; i < p.vocab; i += tg_size) {
        const uint k = o_order_key(l[i]);
        if (k > best_key) {
            best_key = k;
            best_idx = i;
        }
    }
    s_key[tid] = best_key;
    s_idx[tid] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2u; s > 0u; s >>= 1) {
        if (tid < s) {
            const uint ok = s_key[tid + s];
            const uint oi = s_idx[tid + s];
            if (ok > s_key[tid] || (ok == s_key[tid] && oi < s_idx[tid])) {
                s_key[tid] = ok;
                s_idx[tid] = oi;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) out[row] = int(s_idx[0]);
}
