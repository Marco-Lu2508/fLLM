// fLLM CUDA backend: sampling on the device.
//
// One block per row takes a row of logits through the steps of vLLM's sampler (vllm/v1/sample): logit bias, repetition / presence /
// frequency penalties, an optional grammar bitmask, temperature, top-k, min-p, typical-p, top-p, then draws the token and extracts
// the log-probabilities of the chosen token and the top alternatives. The draw is a Gumbel-max over the surviving tokens
// (argmax of logit / T + G, G = -ln(-ln u)), with u from Philox-4x32-10 keyed by the request seed and indexed by (draw, token id).
// The draw therefore depends on nothing but (seed, draw number, logits): not on the batch, the row position or the thread count.
// `sampling::device_sample_reference` in the Rust sources computes the same function on the host, with a sorting implementation of
// the thresholds instead of the bisections used here; selftest-gpu compares the two.
//
// Semantics where the device and the host sampler (`sampling::sample_with_state`) differ on purpose:
//  * ties at a top-k / top-p / typical-p threshold are all kept (vLLM's torch implementation keeps them as well);
//  * the token is drawn by Gumbel-max, the host sampler inverts the cumulative distribution with one uniform number.
//
// Compiled without --use_fast_math (build.rs): expf / logf must be the accurate versions for the comparison with the host.
#include "common.cuh"
#include "fllm_cuda_internal.h"

namespace {

constexpr int kBlock = 1024;
constexpr int kMaxTop = 20;

struct Philox4 {
    uint32_t x, y, z, w;
};

__device__ __forceinline__ Philox4 philox4x32_10(Philox4 c, uint32_t k0, uint32_t k1) {
    for (int i = 0; i < 10; i++) {
        const uint32_t hi0 = __umulhi(0xD2511F53u, c.x), lo0 = 0xD2511F53u * c.x;
        const uint32_t hi1 = __umulhi(0xCD9E8D57u, c.z), lo1 = 0xCD9E8D57u * c.z;
        c = Philox4{hi1 ^ c.y ^ k0, lo1, hi0 ^ c.w ^ k1, lo0};
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return c;
}

// Same function as `sampling::philox_uniform`: a number in the open interval (0, 1).
__device__ __forceinline__ float philox_uniform(uint64_t seed, uint64_t draw, uint32_t index) {
    const Philox4 r = philox4x32_10(Philox4{index, 0u, (uint32_t)draw, (uint32_t)(draw >> 32)}, (uint32_t)seed, (uint32_t)(seed >> 32));
    return ((float)(r.x >> 8) + 0.5f) / 16777216.0f;
}

// Order-preserving map of floats to unsigned integers (larger float, larger key).
__device__ __forceinline__ uint32_t f2key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

struct Shared {
    double red[32];
    float fred[32];
    float bval[32];
    int bidx[32];
    unsigned hist[256];
    uint32_t prefix;
    int remaining;
    int flag;
    double scalar;
};

__device__ __forceinline__ double block_sum_d(double v, Shared &sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh.red[threadIdx.x >> 5] = v;
    __syncthreads();
    double total = 0.0;
    for (int i = 0; i < (int)(blockDim.x >> 5); i++) total += sh.red[i];  // fixed order: deterministic
    __syncthreads();
    return total;
}

__device__ __forceinline__ float block_max_f(float v, Shared &sh) {
    v = warp_max(v);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh.fred[threadIdx.x >> 5] = v;
    __syncthreads();
    float m = -INFINITY;
    for (int i = 0; i < (int)(blockDim.x >> 5); i++) m = fmaxf(m, sh.fred[i]);
    __syncthreads();
    return m;
}

// Largest value; the lowest index wins a tie. An index of INT_MAX means no candidate.
__device__ __forceinline__ void block_argmax(float v, int idx, Shared &sh, float &best, int &besti) {
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        if (ov > v || (ov == v && oi < idx)) {
            v = ov;
            idx = oi;
        }
    }
    __syncthreads();
    if ((threadIdx.x & 31) == 0) {
        sh.bval[threadIdx.x >> 5] = v;
        sh.bidx[threadIdx.x >> 5] = idx;
    }
    __syncthreads();
    best = sh.bval[0];
    besti = sh.bidx[0];
    for (int i = 1; i < (int)(blockDim.x >> 5); i++) {
        if (sh.bval[i] > best || (sh.bval[i] == best && sh.bidx[i] < besti)) {
            best = sh.bval[i];
            besti = sh.bidx[i];
        }
    }
    __syncthreads();
}

__device__ __forceinline__ float sanitize(float v) { return isfinite(v) ? v : -INFINITY; }

// Key of the k-th largest value of w (counting every element, -inf included) by a radix select over the order-preserving keys.
__device__ uint32_t kth_largest_key(const float *w, int vocab, int k, Shared &sh) {
    if (threadIdx.x == 0) {
        sh.prefix = 0;
        sh.remaining = k;
    }
    __syncthreads();
    for (int pass = 0; pass < 4; pass++) {
        const int shift = 24 - 8 * pass;
        for (int i = threadIdx.x; i < 256; i += blockDim.x) sh.hist[i] = 0;
        __syncthreads();
        const uint32_t prefix = sh.prefix;
        for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
            const uint32_t key = f2key(w[i]);
            const bool match = pass == 0 || (key >> (shift + 8)) == (prefix >> (shift + 8));
            if (match) atomicAdd(&sh.hist[(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int acc = 0, rem = sh.remaining;
            uint32_t bin = 0;
            for (int b = 255; b >= 0; b--) {
                if (acc + (int)sh.hist[b] >= rem) {
                    bin = (uint32_t)b;
                    break;
                }
                acc += (int)sh.hist[b];
            }
            sh.prefix = prefix | (bin << shift);
            sh.remaining = rem - acc;
        }
        __syncthreads();
    }
    return sh.prefix;
}

}  // namespace

struct FllmSampleRow {
    float temperature, top_p, min_p, typical_p;
    int32_t top_k;
    float repetition_penalty, presence_penalty, frequency_penalty;
    uint64_t seed, draw;
    int32_t pen_off, pen_len, bias_off, bias_len, mask_row, logprobs_mode, top_n, pad;
};
static_assert(sizeof(FllmSampleRow) == 80, "FllmSampleRow must match the Rust struct SampleRowParams");

namespace {

// Smallest key t such that the mass of {key >= t} is still at least `target`: bisection on the 32-bit key space. `keyfn(i)` gives the
// key of element i (0 for an element that is gone), `mass(i)` its weight. Returns 0 (keep everything) when even the full mass is short.
template <typename KeyFn, typename MassFn>
__device__ uint32_t mass_threshold(int vocab, double target, Shared &sh, KeyFn keyfn, MassFn massfn) {
    uint32_t lo = 0, hi = 0xFFFFFFFFu;  // invariant: mass(lo) >= target (assumed), find the largest such key
    while (lo < hi) {
        const uint32_t mid = lo + ((hi - lo) >> 1) + 1;
        double part = 0.0;
        for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
            const uint32_t k = keyfn(i);
            if (k >= mid) part += massfn(i);
        }
        const double total = block_sum_d(part, sh);
        if (total >= target) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

__global__ void __launch_bounds__(kBlock)
sample_rows_kernel(const float *__restrict__ logits, float *work, int vocab, const FllmSampleRow *__restrict__ rows,
                   const int32_t *__restrict__ pen_ids, const int32_t *__restrict__ pen_counts,
                   const int32_t *__restrict__ bias_ids, const float *__restrict__ bias_vals, const uint32_t *__restrict__ mask,
                   int mask_words, int32_t *__restrict__ out_ids, float *__restrict__ out_logprob,
                   int32_t *__restrict__ top_ids, float *__restrict__ top_logprob, int top_stride) {
    __shared__ Shared sh;
    const int row = blockIdx.x;
    const FllmSampleRow p = rows[row];
    const float *src = logits + (size_t)row * vocab;
    float *w = work + (size_t)row * vocab;
    const bool in_place = (const float *)w == src;

    for (int i = threadIdx.x; i < vocab; i += blockDim.x) w[i] = sanitize(src[i]);
    __syncthreads();

    // log-sum-exp of the raw distribution, for the raw log-probabilities
    float raw_max = 0.f;
    double raw_lse = 0.0;
    if (p.logprobs_mode == 1 && !in_place) {
        float m = -INFINITY;
        for (int i = threadIdx.x; i < vocab; i += blockDim.x) m = fmaxf(m, w[i]);
        raw_max = block_max_f(m, sh);
        double s = 0.0;
        for (int i = threadIdx.x; i < vocab; i += blockDim.x) s += (double)expf(w[i] - raw_max);
        raw_lse = log(block_sum_d(s, sh));
    }

    // penalties: repetition first, then presence / frequency, then the bias (the order of sampling::apply_penalties)
    for (int j = threadIdx.x; j < p.pen_len; j += blockDim.x) {
        const int id = pen_ids[p.pen_off + j];
        if (id < 0 || id >= vocab) continue;
        float v = w[id];
        if (fabsf(p.repetition_penalty - 1.f) >= 1e-5f) v = v > 0.f ? v / p.repetition_penalty : v * p.repetition_penalty;
        v -= p.presence_penalty + p.frequency_penalty * (float)pen_counts[p.pen_off + j];
        w[id] = v;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < p.bias_len; j += blockDim.x) {
        const int id = bias_ids[p.bias_off + j];
        if (id >= 0 && id < vocab) w[id] += bias_vals[p.bias_off + j];  // ids of one row are unique
    }
    __syncthreads();
    if (p.mask_row >= 0) {
        const uint32_t *bits = mask + (size_t)p.mask_row * mask_words;
        for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
            if (!((bits[i >> 5] >> (i & 31)) & 1u)) w[i] = -INFINITY;
        }
    }
    __syncthreads();

    float m = -INFINITY;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        const float v = w[i];
        m = fmaxf(m, isfinite(v) ? v : -INFINITY);
    }
    m = block_max_f(m, sh);
    int chosen = -1;
    double lnz = 0.0;  // log of the normalizer of the final distribution, when it is needed
    const bool greedy = p.temperature <= 1e-5f;
    const float inv_t = greedy ? 1.f : (float)(1.0 / (double)p.temperature);

    if (!(m > -INFINITY)) {
        chosen = -1;  // no token left: the host reports it
    } else if (greedy) {
        float best = -INFINITY;
        int besti = 0x7fffffff;
        for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
            if (w[i] > best) {
                best = w[i];
                besti = i;
            }
        }
        float b;
        int bi;
        block_argmax(best, besti, sh, b, bi);
        chosen = bi;
    } else {
        // top-k
        if (p.top_k > 0 && p.top_k < vocab) {
            const uint32_t t = kth_largest_key(w, vocab, p.top_k, sh);
            for (int i = threadIdx.x; i < vocab; i += blockDim.x)
                if (f2key(w[i]) < t) w[i] = -INFINITY;
            __syncthreads();
        }
        // normalizer of the survivors
        auto e_of = [&](int i) -> float { return isfinite(w[i]) ? expf((w[i] - m) * inv_t) : 0.f; };
        double z = 0.0;
        {
            double part = 0.0;
            for (int i = threadIdx.x; i < vocab; i += blockDim.x) part += (double)e_of(i);
            z = block_sum_d(part, sh);
        }
        // min-p: relative to the largest probability (e = 1 at the maximum)
        if (p.min_p > 0.f) {
            const float thr = p.min_p;
            for (int i = threadIdx.x; i < vocab; i += blockDim.x)
                if (e_of(i) < thr) w[i] = -INFINITY;
            __syncthreads();
            double part = 0.0;
            for (int i = threadIdx.x; i < vocab; i += blockDim.x) part += (double)e_of(i);
            z = block_sum_d(part, sh);
        }
        // typical-p: keep the tokens whose surprise is closest to the entropy until their mass reaches typical_p
        if (p.typical_p < 1.f) {
            double part = 0.0;
            for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
                const float e = e_of(i);
                if (e > 0.f) {
                    const float pr = e / (float)z;
                    part += (double)(-pr * logf(fmaxf(pr, 1e-30f)));
                }
            }
            const float entropy = (float)block_sum_d(part, sh);
            auto keyfn = [&](int i) -> uint32_t {
                const float e = e_of(i);
                if (!(e > 0.f)) return 0u;
                const float pr = e / (float)z;
                return f2key(-fabsf(-logf(fmaxf(pr, 1e-30f)) - entropy));
            };
            auto massfn = [&](int i) -> double { return (double)e_of(i); };
            const uint32_t t = mass_threshold(vocab, (double)p.typical_p * z, sh, keyfn, massfn);
            for (int i = threadIdx.x; i < vocab; i += blockDim.x)
                if (e_of(i) > 0.f && keyfn(i) < t) w[i] = -INFINITY;
            __syncthreads();
            part = 0.0;
            for (int i = threadIdx.x; i < vocab; i += blockDim.x) part += (double)e_of(i);
            z = block_sum_d(part, sh);
        }
        // top-p
        if (p.top_p > 0.f && p.top_p < 1.f) {
            auto keyfn = [&](int i) -> uint32_t { return e_of(i) > 0.f ? f2key(w[i]) : 0u; };
            auto massfn = [&](int i) -> double { return (double)e_of(i); };
            const uint32_t t = mass_threshold(vocab, (double)p.top_p * z, sh, keyfn, massfn);
            for (int i = threadIdx.x; i < vocab; i += blockDim.x)
                if (e_of(i) > 0.f && f2key(w[i]) < t) w[i] = -INFINITY;
            __syncthreads();
            double part = 0.0;
            for (int i = threadIdx.x; i < vocab; i += blockDim.x) part += (double)e_of(i);
            z = block_sum_d(part, sh);
        }
        lnz = log(z);
        // Gumbel-max draw over the survivors
        float best = -INFINITY;
        int besti = 0x7fffffff;
        for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
            if (!isfinite(w[i])) continue;
            const float u = philox_uniform(p.seed, p.draw, (uint32_t)i);
            const float g = -logf(-logf(u));
            const float score = (w[i] - m) * inv_t + g;
            if (score > best) {
                best = score;
                besti = i;
            }
        }
        float b;
        int bi;
        block_argmax(best, besti, sh, b, bi);
        chosen = bi;
    }

    // log-probabilities
    if (threadIdx.x == 0) {
        out_ids[row] = chosen;
        float lp = 0.f;
        if (chosen >= 0 && p.logprobs_mode == 1) lp = (in_place ? 0.f : sanitize(src[chosen]) - raw_max - (float)raw_lse);
        else if (chosen >= 0 && p.logprobs_mode == 2) lp = greedy ? 0.f : (w[chosen] - m) * inv_t - (float)lnz;
        out_logprob[row] = lp;
    }
    // top alternatives: N passes, each takes the largest element ordered after the previous pick (value descending, id ascending)
    const int top_n = p.logprobs_mode != 0 ? min(p.top_n, min(kMaxTop, top_stride)) : 0;
    if (top_n > 0) {
        const float *base = p.logprobs_mode == 1 ? (in_place ? w : src) : w;
        float prev_v = INFINITY;
        int prev_i = -1;
        for (int n = 0; n < top_n; n++) {
            float best = -INFINITY;
            int besti = 0x7fffffff;
            for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
                const float v = p.logprobs_mode == 1 ? sanitize(base[i]) : base[i];
                if (!isfinite(v)) continue;
                const bool after = v < prev_v || (v == prev_v && i > prev_i);
                if (after && (v > best || (v == best && i < besti))) {
                    best = v;
                    besti = i;
                }
            }
            float b;
            int bi;
            block_argmax(best, besti, sh, b, bi);
            if (threadIdx.x == 0) {
                if (bi == 0x7fffffff) {
                    top_ids[(size_t)row * top_stride + n] = -1;
                    top_logprob[(size_t)row * top_stride + n] = -INFINITY;
                } else {
                    top_ids[(size_t)row * top_stride + n] = bi;
                    top_logprob[(size_t)row * top_stride + n] =
                        p.logprobs_mode == 1 ? b - raw_max - (float)raw_lse : (greedy ? (n == 0 ? 0.f : -INFINITY) : (b - m) * inv_t - (float)lnz);
                }
            }
            prev_v = b;
            prev_i = bi;
            if (bi == 0x7fffffff) {
                // nothing left: the remaining slots of the row are empty too (the device buffer is not initialised)
                if (threadIdx.x == 0) {
                    for (int r = n + 1; r < top_n; r++) {
                        top_ids[(size_t)row * top_stride + r] = -1;
                        top_logprob[(size_t)row * top_stride + r] = -INFINITY;
                    }
                }
                break;
            }
        }
    }
}

}  // namespace

int fllm_sample_launch(cudaStream_t s, const float *logits, float *work, int rows, int vocab, const void *params,
                       const int32_t *pen_ids, const int32_t *pen_counts, const int32_t *bias_ids, const float *bias_vals,
                       const uint32_t *mask, int mask_words, int32_t *out_ids, float *out_logprob, int32_t *top_ids,
                       float *top_logprob, int top_stride) {
    if (rows <= 0 || vocab <= 0) return 0;
    sample_rows_kernel<<<rows, kBlock, 0, s>>>(logits, work, vocab, static_cast<const FllmSampleRow *>(params), pen_ids, pen_counts,
                                               bias_ids, bias_vals, mask, mask_words, out_ids, out_logprob, top_ids, top_logprob,
                                               top_stride);
    return fllm_check_launch("sample");
}

// Device-side speculative acceptance and rewinding kernel
__global__ void spec_verify_kernel(const int32_t *__restrict__ draft,
                                    const int32_t *__restrict__ target,
                                    int num_draft,
                                    int32_t *__restrict__ out_accepted,
                                    int32_t *__restrict__ out_next,
                                    int64_t *__restrict__ seq_length) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int accepted = 0;
        for (int i = 0; i < num_draft; ++i) {
            if (draft[i] == target[i]) {
                accepted++;
            } else {
                break;
            }
        }
        *out_accepted = accepted;
        *out_next = target[accepted];
        if (seq_length) {
            *seq_length -= (num_draft - accepted);
        }
    }
}

int fllm_spec_verify_launch(cudaStream_t s, const int32_t *draft, const int32_t *target,
                            int num_draft, int32_t *out_accepted, int32_t *out_next,
                            int64_t *seq_length) {
    if (num_draft <= 0) return 0;
    spec_verify_kernel<<<1, 1, 0, s>>>(draft, target, num_draft, out_accepted, out_next, seq_length);
    return fllm_check_launch("spec_verify");
}
