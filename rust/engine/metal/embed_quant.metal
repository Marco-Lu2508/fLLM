// FLLM-KERNEL: fllm_embed_q4_0, fllm_embed_q4_1, fllm_embed_q5_0, fllm_embed_q5_1, fllm_embed_q8_0, fllm_embed_q2_k, fllm_embed_q3_k, fllm_embed_q4_k, fllm_embed_q5_k, fllm_embed_q6_k, fllm_embed_iq4_nl, fllm_embed_mxfp4, fllm_embed_mxfp4_planar | buffers: weights, ids, out | params: o_embed_params | grid: [ceil(cols / threads), n_ids, 1] x [threads, 1, 1]
// Embedding lookup (Stream::embedding_lookup) for block-quantized tables in the GGUF layout of the file (block formats of `QuantType`, decoded by
// the `o_deq_*` functions of common_ops.metal, which transcribe `compute::dequantize`): out[t][c] = dequant(table row ids[t])[c]. One thread per
// output element: block = c / QK, element = c % QK; the row starts at ids[t] * row_bytes (row_bytes = cols / QK * block bytes; the planner checks
// cols % QK == 0). Block bytes are read bytewise (blocks start at arbitrary offsets, e.g. 17-byte MXFP4 blocks).
#include <metal_stdlib>
using namespace metal;

template <int FMT>
[[kernel]] void o_embed_quant(
    device const uchar *weights [[buffer(1)]],
    device const uint *ids [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant o_embed_params &p [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x;
    if (c >= p.cols) return;
    const ulong t = gid.y;
    constexpr uint qk = o_fmt_qk(FMT);
    constexpr uint bytes = o_fmt_bytes(FMT);
    device const uchar *block = weights + ulong(ids[t]) * p.row_bytes + ulong(c / qk) * bytes;
    out[t * p.cols + c] = o_deq<FMT>(block, c % qk);
}

template [[host_name("fllm_embed_q4_0")]] [[kernel]] decltype(o_embed_quant<O_FMT_Q4_0>) o_embed_quant<O_FMT_Q4_0>;
template [[host_name("fllm_embed_q4_1")]] [[kernel]] decltype(o_embed_quant<O_FMT_Q4_1>) o_embed_quant<O_FMT_Q4_1>;
template [[host_name("fllm_embed_q5_0")]] [[kernel]] decltype(o_embed_quant<O_FMT_Q5_0>) o_embed_quant<O_FMT_Q5_0>;
template [[host_name("fllm_embed_q5_1")]] [[kernel]] decltype(o_embed_quant<O_FMT_Q5_1>) o_embed_quant<O_FMT_Q5_1>;
template [[host_name("fllm_embed_q8_0")]] [[kernel]] decltype(o_embed_quant<O_FMT_Q8_0>) o_embed_quant<O_FMT_Q8_0>;
template [[host_name("fllm_embed_q2_k")]] [[kernel]] decltype(o_embed_quant<O_FMT_Q2_K>) o_embed_quant<O_FMT_Q2_K>;
template [[host_name("fllm_embed_q3_k")]] [[kernel]] decltype(o_embed_quant<O_FMT_Q3_K>) o_embed_quant<O_FMT_Q3_K>;
template [[host_name("fllm_embed_q4_k")]] [[kernel]] decltype(o_embed_quant<O_FMT_Q4_K>) o_embed_quant<O_FMT_Q4_K>;
template [[host_name("fllm_embed_q5_k")]] [[kernel]] decltype(o_embed_quant<O_FMT_Q5_K>) o_embed_quant<O_FMT_Q5_K>;
template [[host_name("fllm_embed_q6_k")]] [[kernel]] decltype(o_embed_quant<O_FMT_Q6_K>) o_embed_quant<O_FMT_Q6_K>;
template [[host_name("fllm_embed_iq4_nl")]] [[kernel]] decltype(o_embed_quant<O_FMT_IQ4_NL>) o_embed_quant<O_FMT_IQ4_NL>;
template [[host_name("fllm_embed_mxfp4")]] [[kernel]] decltype(o_embed_quant<O_FMT_MXFP4>) o_embed_quant<O_FMT_MXFP4>;

// MXFP4 in the planar device layout of package p06a (weights/layout.rs): plane 0 = the 16 code bytes of every block (block = row * cols / 32 + c / 32),
// plane 1 at byte `plane1` = the E8M0 scale byte of every block.
kernel void fllm_embed_mxfp4_planar(
    device const uchar *weights [[buffer(1)]],
    device const uint *ids [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant o_embed_params &p [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x;
    if (c >= p.cols) return;
    const ulong t = gid.y;
    const ulong block = ulong(ids[t]) * (p.cols / 32u) + (c / 32u);
    const uint i = c % 32u;
    out[t * p.cols + c] = o_mxfp4_value(weights[p.plane1 + block], weights[block * 16ul + (i & 15u)], i);
}
