# Emulates (a) backend/q4z.rs marlin_arrays, (b) the device repack of hip/hip_q4z.cu (code words per output row, nibble order of the vLLM
# magic-number dequantization), (c) the scale/zero lookup of Q4zView, (d) the bit tricks of q4z_int4_sml_kernel, and compares the
# dequantized weights with the reference of compute.rs: (code - zero) * scale.
import random, struct
AWQ_SLOT = [0, 4, 1, 5, 2, 6, 3, 7]
INTERLEAVE = [0, 2, 4, 6, 1, 3, 5, 7]
def scale_perm():
    p = []
    for i in range(8):
        for j in range(8):
            p.append(i + 8 * j)
    return p
def marlin_arrays(raw, rows, cols, group):
    groups = cols // group; block = group // 2 + 3; row_bytes = groups * block; wpk = rows // 8
    awq = [[0] * wpk for _ in range(cols)]
    for n in range(rows):
        row = raw[n * row_bytes:(n + 1) * row_bytes]
        word, shift = n // 8, 4 * AWQ_SLOT[n % 8]
        for k in range(0, cols, 2):
            byte = row[(k // group) * block + (k % group) // 2]
            awq[k][word] |= (byte & 15) << shift
            awq[k + 1][word] |= (byte >> 4) << shift
    perm = scale_perm()
    scales = [0] * (groups * rows); plain = [0] * (groups * rows)
    for g in range(groups):
        for blk in range(rows // 64):
            for j in range(64):
                n = blk * 64 + perm[j]
                i = n * row_bytes + g * block + group // 2
                scales[g * rows + blk * 64 + j] = raw[i] | (raw[i + 1] << 8)
                plain[g * rows + blk * 64 + j] = raw[i + 2] & 15
    zeros = [0] * (groups * rows // 8)
    for w in range(len(zeros)):
        for i, src in enumerate(INTERLEAVE):
            zeros[w] |= plain[w * 8 + src] << (4 * i)
    return awq, scales, zeros
def f16(bits): return struct.unpack('<e', struct.pack('<H', bits & 0xffff))[0]
def nib_of(t): return ((t & 1) << 2) | ((t & 7) >> 1)
for rows, cols, group in [(128, 128, 32), (64, 256, 64), (192, 128, 128)]:
    groups = cols // group; block = group // 2 + 3
    raw = bytearray()
    for n in range(rows):
        for g in range(groups):
            raw += bytes(random.randrange(256) for _ in range(group // 2))
            raw += struct.pack('<e', random.uniform(0.001, 0.1)) + bytes([random.randrange(16)])
    ref = [[0.0] * cols for _ in range(rows)]
    row_bytes = groups * block
    for n in range(rows):
        for g in range(groups):
            b = raw[n * row_bytes + g * block: n * row_bytes + (g + 1) * block]
            scale = f16(b[group // 2] | (b[group // 2 + 1] << 8)); zero = b[group // 2 + 2] & 15
            for j in range(group // 2):
                ref[n][g * group + 2 * j] = ((b[j] & 15) - zero) * scale
                ref[n][g * group + 2 * j + 1] = ((b[j] >> 4) - zero) * scale
    awq, scales, zeros = marlin_arrays(bytes(raw), rows, cols, group)
    out = [[0] * (cols // 8) for _ in range(rows)]
    for n in range(rows):
        sh = 4 * AWQ_SLOT[n & 7]
        for w in range(cols // 8):
            word = 0
            for t in range(8):
                word |= ((awq[8 * w + t][n // 8] >> sh) & 15) << (4 * nib_of(t))
            out[n][w] = word
    def code_at(n, k): return (out[n][k >> 3] >> (4 * nib_of(k & 7))) & 15
    bad = 0
    for n in range(rows):
        for k in range(cols):
            g = k // group
            pos = (n & ~63) + 8 * (n & 7) + ((n & 63) >> 3)
            flat = g * rows + pos
            scale = f16(scales[flat]); s = pos & 7
            nib = 4 + (s >> 1) if (s & 1) else (s >> 1)
            zero = (zeros[flat >> 3] >> (4 * nib)) & 15
            if abs((code_at(n, k) - zero) * scale - ref[n][k]) > 1e-9: bad += 1
    # magic-number dequantization: halves (n0,n4),(n1,n5),(n2,n6),(n3,n7) pair with the activations (a0,a1)..(a6,a7)
    for n in range(0, rows, 29):
        for w in range(cols // 8):
            qa = out[n][w]
            lo0 = (qa & 0x000F000F) | 0x64006400; hi0 = (qa & 0x00F000F0) | 0x64006400
            q2 = qa >> 8
            lo1 = (q2 & 0x000F000F) | 0x64006400; hi1 = (q2 & 0x00F000F0) | 0x64006400
            c = []
            for reg, is_hi in ((lo0, False), (hi0, True), (lo1, False), (hi1, True)):
                for half in (0, 1):
                    v = f16(reg >> (16 * half))
                    c.append(v * (1.0 / 16.0) - 64.0 if is_hi else v - 1024.0)
            want = [code_at(n, 8 * w + t) for t in range(8)]   # element e of the 8 pairs with the activation of offset e
            if [int(x) for x in c] != want: bad += 1
    # GEMV step: 32 codes = 4 words, code j in word j >> 3 at nibble nib_of(j)
    for n in range(0, rows, 37):
        for st in range(cols // 32):
            words = out[n][st * 4:(st + 1) * 4]
            for j in range(32):
                if ((words[j >> 3] >> (4 * nib_of(j))) & 15) != code_at(n, st * 32 + j): bad += 1
    print(f"rows={rows} cols={cols} group={group}: mismatches={bad}")
