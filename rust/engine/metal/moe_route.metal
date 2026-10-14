// FLLM-KERNEL: moe_topk | constants: rows,experts,k,normalize_scores,scale | buffers: 1 logits(r), 2 ids(w), 3 weights(w) | threads: 128,1,1
// (the constants struct is buffer 0; grid: [rows, 1, 1] threadgroups)
// Softmax routing of mixture-of-experts layers (Stream::moe_topk and Stream::moe_route; semantics of the CPU `moe_route`): for every row of `logits`
// (rows x experts) the k best experts (ties to the lower index) as ids (int32) with their softmax scores (over ALL experts) as weights, renormalised over
// the selected ones when `normalize_scores` != 0 (divided by max(sum, 1e-8)) and then multiplied by `scale`. moe_route is normalize = 1, scale = 1.
// ids[r * k + j] / weights[r * k + j] are in descending order of score. One threadgroup (128 threads) per row; experts <= 2048, k <= 32 (the planner checks).
// Port of shaders/vulkan/moe_topk.comp (package 3); the threadgroup uses 9 KiB of threadgroup memory.

#define MOE_TOPK_WG 128u
#define MOE_TOPK_MAX_EXPERTS 2048u
#define MOE_TOPK_MAX_K 32u

struct moe_topk_params {
    uint rows;
    uint experts;
    uint k;
    uint normalize_scores;
    float scale;
};

kernel void moe_topk(
    constant moe_topk_params &p [[buffer(0)]],
    device const float *logits [[buffer(1)]],
    device int *ids [[buffer(2)]],
    device float *weights [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float s_logit[MOE_TOPK_MAX_EXPERTS];
    threadgroup float s_red[MOE_TOPK_WG];
    threadgroup float s_val[MOE_TOPK_WG];
    threadgroup uint s_idx[MOE_TOPK_WG];
    threadgroup uint s_chosen[MOE_TOPK_MAX_K];
    threadgroup float s_score[MOE_TOPK_MAX_K];

    if (row >= p.rows) return;  // uniform: the whole threadgroup leaves
    const ulong base = ulong(row) * p.experts;
    float m = R_NEG;
    for (uint e = lane; e < p.experts; e += MOE_TOPK_WG) {
        const float v = logits[base + e];
        s_logit[e] = v;
        m = max(m, v);
    }
    m = r_tg_max<MOE_TOPK_WG>(m, s_red, lane);  // its barriers also publish s_logit
    float sum = 0.0f;
    for (uint e = lane; e < p.experts; e += MOE_TOPK_WG) sum += precise::exp(s_logit[e] - m);
    sum = r_tg_sum<MOE_TOPK_WG>(sum, s_red, lane);

    for (uint j = 0; j < p.k; j++) {
        float best = R_NEG;
        uint besti = 0x7FFFFFFFu;
        for (uint e = lane; e < p.experts; e += MOE_TOPK_WG) {
            const float v = s_logit[e];
            if (v > best || (v == best && e < besti)) {
                best = v;
                besti = e;
            }
        }
        s_val[lane] = best;
        s_idx[lane] = besti;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = MOE_TOPK_WG / 2u; s > 0u; s >>= 1u) {
            if (lane < s) {
                const float ov = s_val[lane + s];
                const uint oi = s_idx[lane + s];
                if (ov > s_val[lane] || (ov == s_val[lane] && oi < s_idx[lane])) {
                    s_val[lane] = ov;
                    s_idx[lane] = oi;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0u) {
            uint bi = s_idx[0];
            if (bi >= p.experts) bi = 0u;  // only for degenerate input (NaN logits)
            s_chosen[j] = bi;
            s_score[j] = precise::exp(s_val[0] - m) / sum;
            s_logit[bi] = -3.4e38f;  // below R_NEG: never selected again
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        float total = 0.0f;
        for (uint j = 0; j < p.k; j++) total += s_score[j];
        const float denom = max(total, 1e-8f);
        for (uint j = 0; j < p.k; j++) {
            ids[ulong(row) * p.k + j] = int(s_chosen[j]);
            const float sc = p.normalize_scores != 0u ? s_score[j] / denom : s_score[j];
            weights[ulong(row) * p.k + j] = sc * p.scale;
        }
    }
}
