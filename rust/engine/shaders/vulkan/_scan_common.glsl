// FLLM Vulkan shaders: chunk-parallel scan of the recurrent layers. Shared by the scan_* shaders; included after `_ops_common.glsl`.
//
// One linear recurrence per head covers Gated DeltaNet, Kimi Delta Attention, RWKV-7, RWKV-6 and Mamba-2:
//     P_t = (diag(w_t) - b_t a_t^T) P_{t-1} + k_t v_t^T,      o_t = P_t^T q_t + c_t v_t,      w_t = exp(lw_t), lw_t <= 0
// (rust/engine/src/chunked_scan.rs proves the chunked form equal to it and documents the derivation; cuda/scan.cu is the CUDA version of these shaders).
// A prompt run is cut into chunks of SC = 64 tokens; scan_cum, scan_pairs, scan_solve and scan_out work on all chunks at once, scan_state walks the chunks in
// order. A per-architecture `*_prep` shader writes the inputs of the run, a `*_post` shader applies the output norm / gate.
//
// The scratch buffer (one per operation, `sc` in the shaders) holds, in this order, the inputs [row][head][width] of the run's rows (row = token index inside
// the dispatch's window, 64 * slots rows, slots = ceil(n / 64)) and the workspace of every (chunk, head); all offsets are floats and derived from the same few
// push constants by scan_layout (the same arithmetic as scan_plan in cuda/scan.cu and `ScanLayout` in ops/scan.rs).
#ifndef FLLM_SCAN_COMMON
#define FLLM_SCAN_COMMON

const uint SC = 64u;        // tokens per chunk
const uint SCW = 32u;       // value columns of the state per workgroup of scan_state
const uint SF_RANK1 = 1u;   // rank-one terms a, b present
const uint SF_BSAME = 2u;   // b equals kp (stored once)
const uint SF_LWV = 4u;     // the decay is a vector (hk values) per token and head, else one value
const uint SF_QPREV = 8u;   // the readout sees the state before the token (RWKV-6)
const uint SF_HASC = 16u;   // c_t v_t term present
const float SLW_MIN = -30.0;  // log decays are clamped here (exp(-30) = 1e-13 is zero for a state)

uint up4(uint x) { return (x + 3u) & ~3u; }

struct ScanLay {
    uint gw;     // width of a row of G: hk (SF_LWV) or 1
    uint chunk;  // floats of workspace per (chunk, head)
    uint o_g, o_bm, o_xi, o_qk, o_qb, o_r0, o_wr, o_qe, o_o0, o_kd, o_bd, o_de;  // parts of a chunk's workspace
    uint q, kp, a, b, lw, c, v, o;  // inputs / output of the run
    uint ws;                         // workspace of chunk slot 0, head 0
};

ScanLay scan_layout(uint hk, uint hv, uint heads, uint flags, uint slots) {
    ScanLay y;
    const bool r1 = (flags & SF_RANK1) != 0u;
    y.gw = (flags & SF_LWV) != 0u ? hk : 1u;
    uint c = 0u;
    y.o_g = c;  c += up4(SC * y.gw);
    y.o_bm = c; c += up4(r1 ? SC * SC : 0u);
    y.o_xi = c; c += up4(r1 ? SC * SC : 0u);
    y.o_qk = c; c += up4(SC * SC);
    y.o_qb = c; c += up4(r1 ? SC * SC : 0u);
    y.o_r0 = c; c += up4(r1 ? SC * hv : 0u);
    y.o_wr = c; c += up4(r1 ? SC * hk : 0u);
    y.o_qe = c; c += up4(SC * hk);
    y.o_o0 = c; c += up4(SC * hv);
    y.o_kd = c; c += up4(SC * hk);
    y.o_bd = c; c += up4(r1 ? SC * hk : 0u);
    y.o_de = c; c += up4(hk);
    y.chunk = c;
    const uint rh = slots * SC * heads;
    uint off = 0u;
    y.q = off;  off += up4(rh * hk);
    y.kp = off; off += up4(rh * hk);
    y.a = off;  off += up4(r1 ? rh * hk : 0u);
    const uint ob = off;
    off += up4((r1 && (flags & SF_BSAME) == 0u) ? rh * hk : 0u);
    y.b = (flags & SF_BSAME) != 0u ? y.kp : ob;
    y.lw = off; off += up4(rh * y.gw);
    y.c = off;  off += up4(rh);
    y.v = off;  off += up4(rh * hv);
    y.o = off;  off += up4(rh * hv);
    y.ws = off;
    return y;
}

// Workgroup barrier that also publishes the buffer and shared-memory writes of the invocations.
void wg_sync() {
    memoryBarrierBuffer();
    memoryBarrierShared();
    barrier();
}

#endif
