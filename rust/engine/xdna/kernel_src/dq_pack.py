"""Host reference of the packed 4-bit weight stream of ``fllm_dq.cc`` / ``whole_array_dq.py``.

``pack(codes, scales, mins, k, n)`` builds the byte array the NPU design reads as B; ``dequant`` is what the kernel computes.
The Rust engine (``backend/npu/xdna/native/dqpack.rs``) writes the same layout from GGUF Q4_K / Q4_0 / Q4_1 blocks.
"""
import numpy as np
from ml_dtypes import bfloat16

GROUP = 32


def payload_bytes(k, n):
    return 4 * (k // GROUP) * n + k * n // 2


def pack(codes, scales, mins, k=64, n=64):
    """codes: (N, K) uint8 in 0..15; scales, mins: (N, K // 32) float32 (rounded to bf16 here).

    Returns uint8 of shape (K // k, N // n, payload): payload (k step, n tile) = scales[g][n], mins[g][n], nibbles[n // 16][k][n % 16].
    """
    N, K = codes.shape
    assert K % k == 0 and N % n == 0 and k % GROUP == 0 and n % 16 == 0
    out = np.zeros((K // k, N // n, payload_bytes(k, n)), np.uint8)
    sc = scales.astype(bfloat16)
    mn = mins.astype(bfloat16)
    for ks in range(K // k):
        for nt in range(N // n):
            rows = slice(nt * n, (nt + 1) * n)
            g0 = ks * k // GROUP
            s_blk = sc[rows, g0 : g0 + k // GROUP].T.copy()  # (groups, n)
            m_blk = mn[rows, g0 : g0 + k // GROUP].T.copy()
            q = codes[rows, ks * k : (ks + 1) * k]  # (n, k)
            q = q.reshape(n // 16, 16, k).transpose(0, 2, 1)  # (n/16, k, 16)
            flat = q.reshape(-1)
            nib = (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8)
            out[ks, nt] = np.concatenate([s_blk.view(np.uint16).reshape(-1).view(np.uint8), m_blk.view(np.uint16).reshape(-1).view(np.uint8), nib])
    return out


def dequant(codes, scales, mins):
    """What the kernel computes per weight: bf16(min + scale * code), scale and min as bf16."""
    s = np.repeat(scales.astype(bfloat16).astype(np.float32), GROUP, axis=1)
    m = np.repeat(mins.astype(bfloat16).astype(np.float32), GROUP, axis=1)
    return (m + s * codes.astype(np.float32)).astype(bfloat16)
