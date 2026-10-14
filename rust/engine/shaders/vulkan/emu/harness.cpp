// Executes a transformed quantized-linear shader (namespace K, optionally the activation prepass in namespace PRE) on the CPU and compares
// the output with an f64 reference built from the packed weights' dequantized values. Build flags: -DMV or -DMM, -DIDOT, -DSPEC_1 -DSPEC_2 -DWG_SIZE.
#include "shim.h"
#include "sh_k.inc"
#ifdef IDOT
#include "sh_pre.inc"
#endif
#if defined(PERM) && !defined(IDOT)
#include "sh_pp.inc"
#endif

static uint rng_state = 12345u;
static float rnd() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((rng_state >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
}

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    FILE* fp = fopen(argv[1], "rb");
    if (!fp) { perror("open"); return 2; }
    uint batch = (uint)atoi(argv[2]);
    uint hdr[8];
    if (fread(hdr, 4, 8, fp) != 8) return 2;
    const uint rows = hdr[0], cols = hdr[1], nb = hdr[3];
    std::vector<uint> buf(hdr[7] / 4);
    if (fread(buf.data(), 4, buf.size(), fp) != buf.size()) return 2;
    std::vector<float> ref((size_t)rows * cols);
    if (fread(ref.data(), 4, ref.size(), fp) != ref.size()) return 2;

    const uint ROW_BASE = 2, TOTAL_ROWS = rows + 5;
    const float scale = 0.75f;
    std::vector<float> x((size_t)batch * cols), bias(TOTAL_ROWS), out((size_t)batch * TOTAL_ROWS, -12345.0f);
    for (auto& v : x) v = rnd();
    for (auto& v : bias) v = rnd();
    std::vector<uint> perm(cols);
    for (uint c = 0; c < cols; ++c) perm[c] = c;
#ifdef PERM
    for (uint c = cols - 1; c > 0; --c) { uint j = (uint)((rnd() * 0.5f + 0.5f) * (c + 0.999f)); std::swap(perm[c], perm[j]); }
#endif
    // activations as the weights see them: x'[c] = x[perm[c]]
    std::vector<float> xr(x.size());
    for (uint b = 0; b < batch; ++b)
        for (uint c = 0; c < cols; ++c) xr[(size_t)b * cols + c] = x[(size_t)b * cols + perm[c]];

    std::vector<double> want((size_t)batch * rows), mag((size_t)batch * rows);
    for (uint b = 0; b < batch; ++b)
        for (uint r = 0; r < rows; ++r) {
            double s = 0, m = 0;
            for (uint k = 0; k < cols; ++k) {
                double t = (double)xr[(size_t)b * cols + k] * ref[(size_t)r * cols + k];
                s += t; m += fabs(t);
            }
            want[(size_t)b * rows + r] = scale * s + bias[ROW_BASE + r];
            mag[(size_t)b * rows + r] = scale * m + fabs(bias[ROW_BASE + r]);
        }

    const uint xq_d = batch * cols / 4;
    K::p.rows = rows; K::p.cols = cols; K::p.batch = batch; K::p.flags = 1; K::p.scale = scale; K::p.nb = nb;
    K::p.off1 = hdr[4]; K::p.off2 = hdr[5]; K::p.off3 = hdr[6];
    K::p.out_stride = TOTAL_ROWS; K::p.row_base = ROW_BASE; K::p.xq_d = xq_d;
    K::w = buf.data();
    K::outv = out.data();
    K::biasv = bias.data();
    std::vector<uint> xq;
#ifdef IDOT
    xq.assign(xq_d + (size_t)batch * cols / 32, 0xdeadbeefu);
    PRE::p.cols = cols; PRE::p.batch = batch; PRE::p.xq_d = xq_d;
    PRE::x = x.data();
    PRE::xq = xq.data();
#ifdef PERM
    PRE::perm = perm.data();
#endif
    {
        uint blocks = batch * cols / 32, n = (blocks + 127) / 128, gx = std::min(n, 3u), gy = (n + gx - 1) / gx;
        dispatch(PRE::main_, gx, gy, 128);
    }
    K::xq = xq.data();
#elif defined(PERM)
    std::vector<float> y(x.size());
    PP::p.cols = cols; PP::p.total = batch * cols;
    PP::x = x.data(); PP::y = y.data(); PP::perm = perm.data();
    {
        uint n = (batch * cols + 255) / 256, gx = std::min(n, 3u), gy = (n + gx - 1) / gx;
        dispatch(PP::main_, gx, gy, 256);
    }
    K::xv = (vec4*)y.data();
#else
    K::xv = (vec4*)x.data();
#endif
#ifdef MV
    {
        uint n = (rows + K::NROWS - 1) / K::NROWS, gx = std::min(n, 5u), gy = (n + gx - 1) / gx;   // small x forces the x/y group split
        dispatch(K::main_, gx, gy, K::gl_WorkGroupSize.x);
    }
#else
    dispatch(K::main_, (rows + 63) / 64, (batch + 63) / 64, K::gl_WorkGroupSize.x);
#endif
    size_t bad = 0;
    double worst = 0;
    const double rel = argc > 3 ? atof(argv[3]) : 2e-5;
    for (uint b = 0; b < batch; ++b)
        for (uint r = 0; r < TOTAL_ROWS; ++r) {
            float got = out[(size_t)b * TOTAL_ROWS + r];
            if (r < ROW_BASE || r >= ROW_BASE + rows) {
                if (got != -12345.0f) { if (bad < 5) printf("  wrote outside the row range: b%u r%u\n", b, r); ++bad; }
                continue;
            }
            size_t i = (size_t)b * rows + (r - ROW_BASE);
            double err = fabs((double)got - want[i]);
            worst = std::max(worst, err / (mag[i] + 1e-9));
            if (!(err <= rel * mag[i] + 1e-6)) { if (bad < 5) printf("  MISMATCH b%u r%u: got %g want %g (scale %g)\n", b, r, got, want[i], mag[i]); ++bad; }
        }
    printf("rows=%u cols=%u batch=%u mismatches=%zu worst_rel_err=%.3g\n", rows, cols, batch, bad, worst);
    return bad ? 1 : 0;
}
