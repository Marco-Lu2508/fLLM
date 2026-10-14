// FLLM-KERNEL: fllm_kv_write_f16, fllm_kv_write_q8 | buffers: src, slots, pool | params: o_kvw_params | grid: f16: [ceil(kv / threads), rows, 1] x [threads, 1, 1]; q8: [kv / group, rows, 1] x [32, 1, 1]
// Stream::kv_cache_write, one pool (keys or values) per dispatch; the layout is that of `DeviceKvCache` (backend/mod.rs). Row r of `src` (kv = kv_heads *
// dim f32 values) goes to slot slots[r]. The pool is bound as `device char *` and viewed as 16-bit elements (f16) or bytes (q8): a layer may start
// at an odd 16-bit word, which a half pointer handles naturally. Different threads store to neighbouring 16-bit / 8-bit locations, which Metal
// does without read-modify-write, so no atomics are needed (unlike the Vulkan shaders). Slots of one call are distinct (Stream contract).
//   f16: half(x) at element  base + slot * kv + c  (round to nearest even like the CPU conversion).
//   q8:  one threadgroup of 32 threads per group of `group` values of a row, like the CPU encoder `encode_q8_cache_group`:
//        d = amax / 127, scale = half(d) at element  base + scale_base + slot * (kv / group) + g,
//        q = round-half-away-from-zero(x / d) clamped to +-127 (all zero for d = 0) at byte  2 * base + slot * kv + g * group + j.
//        The division is `precise::divide`, so values on a rounding boundary land where the CPU puts them.
#include <metal_stdlib>
using namespace metal;

kernel void fllm_kv_write_f16(
    device const float *src [[buffer(1)]],
    device const uint *slots [[buffer(2)]],
    device char *pool [[buffer(3)]],
    constant o_kvw_params &p [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x;
    if (c >= p.kv) return;
    const ulong r = gid.y;
    device half *ph = (device half *)pool;
    ph[p.base + ulong(slots[r]) * p.kv + c] = half(src[r * p.kv + c]);
}

kernel void fllm_kv_write_q8(
    device const float *src [[buffer(1)]],
    device const uint *slots [[buffer(2)]],
    device char *pool [[buffer(3)]],
    constant o_kvw_params &p [[buffer(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],  // scalar like sg/lane; tgid is a vector, and MSL rejects a scalar thread_position_in_threadgroup next to it
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float red[32];
    const uint g = tgid.x;
    const ulong r = tgid.y;
    const ulong slot = slots[r];
    const uint groups = p.kv / p.group;
    device const float *x = src + r * p.kv + ulong(g) * p.group;

    float amax = 0.0f;
    for (uint j = tid; j < p.group; j += 32u) amax = max(amax, abs(x[j]));
    amax = o_tg_max(amax, red, sg, lane, nsg);
    const float d = amax / 127.0f;

    device half *ph = (device half *)pool;
    device char *qb = pool + 2ul * p.base + slot * p.kv + ulong(g) * p.group;
    if (tid == 0u) ph[p.base + p.scale_base + slot * groups + g] = half(d);
    for (uint j = tid; j < p.group; j += 32u) {
        float q = 0.0f;
        if (d != 0.0f) q = clamp(round(precise::divide(x[j], d)), -127.0f, 127.0f);
        qb[j] = char(int(q));
    }
}
