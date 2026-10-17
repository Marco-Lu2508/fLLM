# Emulates hip_gemm.cuh (wave grid, LDS tile, fragment loads, accumulator mapping) on small shapes with the documented lane layouts of
# MFMA 16x16x16 (CDNA), WMMA gfx11 and WMMA gfx12, and compares with a plain matrix product.
import random
BM = BN = 64; BK = 64
ARCH = {
  'cdna':  dict(wave=64, frag=4,  accn=4, k0=lambda l: 4 * (l >> 4), row=lambda l, i: 4 * (l >> 4) + i),
  'rdna3': dict(wave=32, frag=16, accn=8, k0=lambda l: 0,            row=lambda l, i: 2 * i + (l >> 4)),
  'rdna4': dict(wave=32, frag=8,  accn=8, k0=lambda l: 8 * (l >> 4), row=lambda l, i: 8 * (l >> 4) + i),
}
def mma(arch, a_frags, b_frags, acc):
    # hardware: build A (m x k) and B (k x n) from the lane fragments, D += A*B, scatter D back to the lanes
    wave = arch['wave']; A = [[None] * 16 for _ in range(16)]; B = [[None] * 16 for _ in range(16)]
    for l in range(wave):
        for e in range(arch['frag']):
            k = arch['k0'](l) + e
            m = l % 16
            if A[m][k] is not None: assert A[m][k] == a_frags[l][e], 'A duplicate mismatch'
            A[m][k] = a_frags[l][e]
            if B[k][m] is not None: assert B[k][m] == b_frags[l][e], 'B duplicate mismatch'
            B[k][m] = b_frags[l][e]
    for m in range(16):
        for k in range(16): assert A[m][k] is not None and B[k][m] is not None
    D = [[sum(A[m][k] * B[k][n] for k in range(16)) for n in range(16)] for m in range(16)]
    for l in range(wave):
        for i in range(arch['accn']):
            acc[l][i] += D[arch['row'](l, i)][l % 16]
def run(name, rows, cols, batch):
    arch = ARCH[name]; wave = arch['wave']; NT = 256
    W = [[random.randint(-8, 8) for _ in range(cols)] for _ in range(rows)]
    X = [[random.randint(-8, 8) for _ in range(cols)] for _ in range(batch)]
    Y = [[0] * rows for _ in range(batch)]
    WGM = 2; WGN = NT // wave // WGM; TM = BM // WGM // 16; TN = BN // WGN // 16
    assert WGM * TM * 16 == BM and WGN * TN * 16 == BN
    for bx in range((rows + BM - 1) // BM):
        for by in range((batch + BN - 1) // BN):
            m0, n0 = bx * BM, by * BN
            As = [[0] * BK for _ in range(BM)]; Bs = [[0] * BK for _ in range(BN)]
            acc = {w: [[[[0] * arch['accn'] for _ in range(wave)] for _ in range(TN)] for _ in range(TM)] for w in range(NT // wave)}
            for kt in range((cols + BK - 1) // BK):
                # fetch + stash: thread tid decodes row tid // 4, run tid % 4
                for tid in range(NT):
                    lr, q = tid // 4, tid % 4
                    for j in range(16):
                        k = kt * BK + q * 16 + j
                        m = m0 + lr; n = n0 + lr
                        As[lr][q * 16 + j] = W[m][k] if m < rows and k < cols else 0
                        Bs[lr][q * 16 + j] = X[n][k] if n < batch and k < cols else 0
                for wv in range(NT // wave):
                    wm, wn = wv % WGM, wv // WGM
                    for kk in range(0, BK, 16):
                        af = [[[As[wm * TM * 16 + i * 16 + (l & 15)][kk + arch['k0'](l) + e] for e in range(arch['frag'])] for l in range(wave)] for i in range(TM)]
                        bf = [[[Bs[wn * TN * 16 + j * 16 + (l & 15)][kk + arch['k0'](l) + e] for e in range(arch['frag'])] for l in range(wave)] for j in range(TN)]
                        for i in range(TM):
                            for j in range(TN):
                                mma(arch, af[i], bf[j], acc[wv][i][j])
            for wv in range(NT // wave):
                wm, wn = wv % WGM, wv // WGM
                for i in range(TM):
                    for j in range(TN):
                        for l in range(wave):
                            n = n0 + wn * TN * 16 + j * 16 + (l & 15)
                            for e in range(arch['accn']):
                                m = m0 + wm * TM * 16 + i * 16 + arch['row'](l, e)
                                if m < rows and n < batch: Y[n][m] = acc[wv][i][j][l][e]
    bad = 0
    for n in range(batch):
        for m in range(rows):
            ref = sum(W[m][k] * X[n][k] for k in range(cols))
            if ref != Y[n][m]: bad += 1
    print(f"{name:6s} rows={rows} cols={cols} batch={batch}: mismatches={bad}")
for name in ARCH:
    run(name, 80, 96, 20)
