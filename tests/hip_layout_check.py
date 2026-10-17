#!/usr/bin/env python3
"""Paper check of the matrix-core lane layouts used by the HIP kernels (test level L0, no AMD GPU involved).

What it does
  1. Models the three 16x16x16 FP16 matrix instructions the HIP backend uses (CDNA MFMA, RDNA3 WMMA, RDNA4 WMMA) from the
     lane/register layouts of AMD's matrix instruction calculator and the RDNA3/RDNA4 ISA guides, written as "matrix coordinate ->
     (lane, register element)".  This model is independent of the kernel code.
  2. Reads the index formulas the kernels use (mma_k0 / mma_row / fragment sizes in rust/engine/hip/hip_mma.cuh, the tile constants
     in rust/engine/hip/hip_gemm.cuh) out of the sources, so that editing a formula in the .cuh files is noticed here.
  3. Checks that the kernel formulas are the inverse of the instruction model (every matrix element is held by exactly one lane/element,
     and the kernel's A / B / D positions agree with the instruction), and
  4. emulates the complete tile loop of mma_gemm_kernel (wave grid, sub-tiles, fragment loads from the LDS tile, accumulator store)
     with the modelled instructions for ragged shapes and compares the result with a float64 reference.
  5. Checks the alignment of every LDS fragment load (the vector loads of hip_mma.cuh need 8 / 16 byte alignment).

Run:  python3 tests/hip_layout_check.py            (needs numpy)
A pass means the index arithmetic is consistent with the documented layouts. It does not prove the documentation was read correctly
(that needs silicon) and says nothing about timing.
"""
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIP = os.path.join(ROOT, "rust", "engine", "hip")

# ---- 1. instruction model: matrix coordinate -> (lane, element) -----------------------------------------------------------------


class Instr:
    """A 16x16x16 instruction: where A[i][k], B[k][j] and D[i][j] live (D is also the accumulator input C)."""

    def __init__(self, name, wave, a_frag, acc, a_pos, b_pos, d_pos):
        self.name, self.wave, self.a_frag, self.acc = name, wave, a_frag, acc
        self.a_pos, self.b_pos, self.d_pos = a_pos, b_pos, d_pos

    def run(self, a_regs, b_regs, c_regs):
        """a_regs[lane][e], b_regs[lane][e], c_regs[lane][e] -> new accumulator registers."""
        A = np.zeros((16, 16))
        B = np.zeros((16, 16))
        C = np.zeros((16, 16))
        for i in range(16):
            for k in range(16):
                l, e = self.a_pos(i, k)
                A[i][k] = a_regs[l][e]
                l, e = self.b_pos(k, i)  # b_pos(k, j)
                B[k][i] = b_regs[l][e]
            for j in range(16):
                l, e = self.d_pos(i, j)
                C[i][j] = c_regs[l][e]
        D = C + A @ B
        out = np.zeros_like(c_regs)
        for i in range(16):
            for j in range(16):
                l, e = self.d_pos(i, j)
                out[l][e] = D[i][j]
        return out


# v_mfma_f32_16x16x16_f16 (gfx908 and later): 64 lanes, 4 halves of A and B, 4 accumulators.
MFMA = Instr(
    "CDNA mfma_f32_16x16x16f16", 64, 4, 4,
    lambda i, k: (i + 16 * (k // 4), k % 4),
    lambda k, j: (j + 16 * (k // 4), k % 4),
    lambda i, j: (j + 16 * (i // 4), i % 4),
)
# v_wmma_f32_16x16x16_f16 (gfx11, wave32): A row i and B column j are replicated in lanes 16..31; accumulator element e of lane l
# holds D[2 e + l // 16][l % 16].
WMMA11 = Instr(
    "RDNA3 wmma_f32_16x16x16_f16_w32", 32, 16, 8,
    lambda i, k: (i, k),
    lambda k, j: (j, k),
    lambda i, j: (j + 16 * (i % 2), i // 2),
)
# v_wmma_f32_16x16x16_f16 (gfx12, wave32): 8 halves per lane, lanes 16..31 hold the upper 8 k.
WMMA12 = Instr(
    "RDNA4 wmma_f32_16x16x16_f16_w32_gfx12", 32, 8, 8,
    lambda i, k: (i + 16 * (k // 8), k % 8),
    lambda k, j: (j + 16 * (k // 8), k % 8),
    lambda i, j: (j + 16 * (i // 8), i % 8),
)
FAMILIES = {"CDNA": MFMA, "RDNA3": WMMA11, "RDNA4": WMMA12}

# ---- 2. read the kernels' formulas -------------------------------------------------------------------------------------------------


def c_expr(s):
    s = s.replace("(lane >> 4)", "(lane // 16)").replace("lane >> 4", "lane // 16")
    return s.strip()


def read_mma_header():
    text = open(os.path.join(HIP, "hip_mma.cuh"), encoding="utf-8").read()
    blocks = re.split(r"^#(?:if|elif) defined\(FLLM_HIP_(CDNA|RDNA3|RDNA4)\)\s*$", text, flags=re.M)
    out = {}
    for fam, body in zip(blocks[1::2], blocks[2::2]):
        wave = int(re.search(r"MMA_WAVE = (\d+);", body).group(1))
        frag = int(re.search(r"MMA_FRAG = (\d+);", body).group(1))
        acc = int(re.search(r"MMA_ACC = (\d+);", body).group(1))
        k0 = re.search(r"int mma_k0\(int(?: lane)?\) \{ return (.*?); \}", body).group(1)
        row = re.search(r"int mma_row\(int lane, int i\) \{ return (.*?); \}", body).group(1)
        out[fam] = dict(wave=wave, frag=frag, acc=acc, k0=eval("lambda lane: " + c_expr(k0)), row=eval("lambda lane, i: " + c_expr(row)))
    assert set(out) == set(FAMILIES), "hip_mma.cuh: expected CDNA, RDNA3 and RDNA4 blocks, found %s" % sorted(out)
    return out


def read_gemm_constants():
    text = open(os.path.join(HIP, "hip_gemm.cuh"), encoding="utf-8").read()
    c = {}
    for name in ("G_BM", "G_BN", "G_BK", "G_LD", "G_NT", "G_WGM"):
        m = re.search(r"\b%s = ([^,;]+)[,;]" % name, text)
        c[name] = m.group(1).strip()
    env = {}
    # constants may refer to each other (G_LD = G_BK + 8)
    for name in ("G_BM", "G_BN", "G_BK", "G_NT", "G_WGM", "G_LD"):
        env[name] = eval(c[name], {}, env)
    return env


# ---- 3. kernel formulas against the instruction model ------------------------------------------------------------------------------


def check_formulas(fam, hdr, ins):
    errs = []
    if hdr["wave"] != ins.wave or hdr["frag"] != ins.a_frag or hdr["acc"] != ins.acc:
        errs.append("%s: header says wave %d frag %d acc %d, instruction has %d %d %d" % (fam, hdr["wave"], hdr["frag"], hdr["acc"], ins.wave, ins.a_frag, ins.acc))
    # bijection of the model
    seen = set()
    for i in range(16):
        for j in range(16):
            p = ins.d_pos(i, j)
            if p in seen:
                errs.append("%s: model D position %s used twice" % (fam, p))
            seen.add(p)
    # D: accumulator element e of lane l is D[mma_row(l, e)][l % 16]
    for lane in range(ins.wave):
        for e in range(ins.acc):
            i, j = hdr["row"](lane, e), lane % 16
            if ins.d_pos(i, j) != (lane, e):
                errs.append("%s: accumulator (lane %d, e %d): kernel says D[%d][%d], instruction puts D[%d][%d] there" % (fam, lane, e, i, j, i, j))
    # A / B: lane l element e is A[l % 16][mma_k0(l) + e] and B[mma_k0(l) + e][l % 16] for e < frag.  RDNA3 lanes 16..31 repeat lanes 0..15.
    for lane in range(ins.wave):
        for e in range(ins.a_frag):
            k = hdr["k0"](lane) + e
            i = lane % 16
            pa, pb = ins.a_pos(i, k), ins.b_pos(k, i)
            if pa[1] != e or (pa[0] != lane and pa[0] != lane % 16) or pb[1] != e or (pb[0] != lane and pb[0] != lane % 16):
                errs.append("%s: fragment (lane %d, e %d) -> A[%d][%d] is at %s, B[%d][%d] at %s" % (fam, lane, e, i, k, pa, k, i, pb))
    return errs


# ---- 4. emulation of mma_gemm_kernel ------------------------------------------------------------------------------------------------


def emulate_gemm(fam, hdr, ins, g, rows, cols, batch, rng):
    BM, BN, BK, LD, NT, WGM = g["G_BM"], g["G_BN"], g["G_BK"], g["G_LD"], g["G_NT"], g["G_WGM"]
    WAVE, FRAG, ACC = hdr["wave"], hdr["frag"], hdr["acc"]
    WGN = NT // WAVE // WGM
    TM, TN = BM // WGM // 16, BN // WGN // 16
    assert WGM * TM * 16 == BM and WGN * TN * 16 == BN
    W = rng.standard_normal((rows, cols)).astype(np.float16).astype(np.float64)
    X = rng.standard_normal((batch, cols)).astype(np.float16).astype(np.float64)
    Y = np.zeros((batch, rows))
    ktiles = (cols + BK - 1) // BK
    for bx in range((rows + BM - 1) // BM):
        for by in range((batch + BN - 1) // BN):
            m0, n0 = bx * BM, by * BN
            # LDS tiles as the fetch / stash code fills them (zero outside the matrix)
            acc = np.zeros((NT // WAVE, TM, TN, WAVE, ACC))
            for kt in range(ktiles):
                As = np.zeros((BM, LD))
                Bs = np.zeros((BN, LD))
                for r in range(BM):
                    for kk in range(BK):
                        k = kt * BK + kk
                        if m0 + r < rows and k < cols:
                            As[r][kk] = W[m0 + r][k]
                        if n0 + r < batch and k < cols:
                            Bs[r][kk] = X[n0 + r][k]
                for wave in range(NT // WAVE):
                    wm, wn = wave % WGM, wave // WGM
                    for kk in range(0, BK, 16):
                        af = [np.zeros((WAVE, FRAG)) for _ in range(TM)]
                        bf = [np.zeros((WAVE, FRAG)) for _ in range(TN)]
                        for lane in range(WAVE):
                            fr, k0 = lane & 15, hdr["k0"](lane)
                            for i in range(TM):
                                off = (wm * TM * 16 + i * 16 + fr) * LD + kk + k0
                                assert (off * 2) % min(FRAG * 2, 16) == 0, "unaligned A fragment load"
                                af[i][lane] = As.reshape(-1)[off:off + FRAG]
                            for j in range(TN):
                                off = (wn * TN * 16 + j * 16 + fr) * LD + kk + k0
                                assert (off * 2) % min(FRAG * 2, 16) == 0, "unaligned B fragment load"
                                bf[j][lane] = Bs.reshape(-1)[off:off + FRAG]
                        for i in range(TM):
                            for j in range(TN):
                                acc[wave][i][j] = ins.run(af[i], bf[j], acc[wave][i][j])
            for wave in range(NT // WAVE):
                wm, wn = wave % WGM, wave // WGM
                for i in range(TM):
                    for j in range(TN):
                        for lane in range(WAVE):
                            n = n0 + wn * TN * 16 + j * 16 + (lane & 15)
                            for e in range(ACC):
                                m = m0 + wm * TM * 16 + i * 16 + hdr["row"](lane, e)
                                if m < rows and n < batch:
                                    Y[n][m] = acc[wave][i][j][lane][e]
    ref = X @ W.T
    err = np.abs(Y - ref)
    worst = tuple(int(x) for x in np.unravel_index(np.argmax(err), err.shape))
    return float(err.max()), worst, float(np.abs(ref).max())


def main():
    hdrs = read_mma_header()
    g = read_gemm_constants()
    rng = np.random.default_rng(1)
    failed = 0
    for fam, ins in FAMILIES.items():
        errs = check_formulas(fam, hdrs[fam], ins)
        for e in errs[:10]:
            print("FAIL", e)
        failed += len(errs)
        if errs:
            continue
        for (rows, cols, batch) in ((70, 100, 9), (64, 64, 64), (130, 200, 70)):
            if rows * cols * batch > 400000 and "--full" not in sys.argv:
                rows, cols, batch = 33, 72, 20
            worst, at, scale = emulate_gemm(fam, hdrs[fam], ins, g, rows, cols, batch, rng)
            ok = worst < 1e-9 * max(1.0, scale)
            print("%-5s %-38s GEMM %3dx%3d x %3d tokens: max error %.2e at (token, row) %s %s" % (fam, ins.name, rows, cols, batch, worst, at, "ok" if ok else "FAIL"))
            failed += 0 if ok else 1
    print("hip_layout_check:", "FAILED (%d problems)" % failed if failed else "all layout checks passed (L0: paper model only, no hardware)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
