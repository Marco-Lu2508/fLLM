// FLLM Vulkan shaders, package 3: workgroup reductions in shared memory (no subgroup operations, so they run on every Vulkan 1.0 device).
// Requires `#define WG <power of two>` equal to local_size_x before the include. Every invocation of the workgroup must call these
// functions (they contain barriers) from uniform control flow. The result is returned to all invocations.
#ifndef FLLM_OPS_REDUCE
#define FLLM_OPS_REDUCE

shared float g_red[WG];

float wg_sum(float v) {
    const uint lane = gl_LocalInvocationID.x;
    g_red[lane] = v;
    barrier();
    for (uint s = WG / 2u; s > 0u; s >>= 1u) {
        if (lane < s) g_red[lane] += g_red[lane + s];
        barrier();
    }
    float r = g_red[0];
    barrier();  // g_red may be reused right away
    return r;
}

float wg_max(float v) {
    const uint lane = gl_LocalInvocationID.x;
    g_red[lane] = v;
    barrier();
    for (uint s = WG / 2u; s > 0u; s >>= 1u) {
        if (lane < s) g_red[lane] = max(g_red[lane], g_red[lane + s]);
        barrier();
    }
    float r = g_red[0];
    barrier();
    return r;
}

#endif
