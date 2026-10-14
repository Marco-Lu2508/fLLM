// FLLM-KERNEL: fllm_embed_f32, fllm_embed_f16, fllm_embed_bf16 | buffers: weights, ids, out | params: o_embed_params | grid: [ceil(cols / threads), n_ids, 1] x [threads, 1, 1]
// Embedding lookup (Stream::embedding_lookup) for float tables: out[t][c] = table[ids[t]][c] for t < n_ids, c < cols (= hidden size). The table is
// row-major as in the file: element (row, c) at index row * cols + c, f32 / binary16 / bfloat16 (bfloat16 is widened by a shift, no `bfloat`
// type needed). ids are token ids (uint32, checked against the vocabulary on the host). `row_bytes` is not used.
#include <metal_stdlib>
using namespace metal;

template <int KIND>
[[kernel]] void o_embed_float(
    device const uchar *weights [[buffer(1)]],
    device const uint *ids [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant o_embed_params &p [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x;
    if (c >= p.cols) return;
    const ulong t = gid.y;
    const ulong idx = ulong(ids[t]) * p.cols + c;
    float v;
    if (KIND == 0) {
        v = ((device const float *)weights)[idx];
    } else if (KIND == 1) {
        v = float(((device const half *)weights)[idx]);
    } else {
        v = as_type<float>(uint(((device const ushort *)weights)[idx]) << 16);
    }
    out[t * p.cols + c] = v;
}

template [[host_name("fllm_embed_f32")]] [[kernel]] decltype(o_embed_float<0>) o_embed_float<0>;
template [[host_name("fllm_embed_f16")]] [[kernel]] decltype(o_embed_float<1>) o_embed_float<1>;
template [[host_name("fllm_embed_bf16")]] [[kernel]] decltype(o_embed_float<2>) o_embed_float<2>;
