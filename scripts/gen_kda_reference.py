#!/usr/bin/env python3
"""Reference vectors for the Kimi Delta Attention recurrence (rust/engine/testdata/kda_reference.json).

The numbers come from flash-linear-attention's pure-PyTorch implementations (`fla.ops.kda.naive`), which run on the CPU; the
chunked form is checked against the recurrent form here, so the vector file holds one set of outputs that both agree on:

    pip install flash-linear-attention torch
    python scripts/gen_kda_reference.py rust/engine/testdata/kda_reference.json

Inputs are the raw (pre-normalisation) queries and keys of a layer after its short convolution: the Rust side applies the
L2 normalisation and the `1/sqrt(K)` query scale itself, exactly as `use_qk_l2norm_in_kernel=True` does in the kernels.
"""

from __future__ import annotations

import json
import sys

import torch
from fla.ops.kda.naive import naive_chunk_kda, naive_recurrent_kda


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # same formula as the kernels and transformers: x / sqrt(sum(x^2) + eps)
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def main(path: str) -> None:
    torch.manual_seed(7)
    batch, tokens, heads, k_dim, v_dim, chunk = 1, 24, 3, 16, 16, 8
    q = torch.randn(batch, tokens, heads, k_dim)
    k = torch.randn(batch, tokens, heads, k_dim)
    v = torch.randn(batch, tokens, heads, v_dim)
    # per-channel log decay: -exp(A) * softplus(x) as the Kimi forget gate produces it (always negative)
    a = torch.exp(torch.empty(heads).uniform_(-1.0, 0.5))
    raw = torch.randn(batch, tokens, heads, k_dim) * 1.5 - 1.0
    g = -a[None, None, :, None] * torch.nn.functional.softplus(raw)
    beta = torch.sigmoid(torch.randn(batch, tokens, heads))
    qn, kn = l2norm(q), l2norm(k)
    scale = k_dim**-0.5

    out, state = naive_recurrent_kda(qn, kn, v, g, beta, scale=scale, output_final_state=True)
    chunked, state_chunked = naive_chunk_kda(qn, kn, v, g, beta, scale=scale, output_final_state=True, chunk_size=chunk)
    assert torch.allclose(out, chunked, atol=1e-4, rtol=1e-4), (out - chunked).abs().max()
    assert torch.allclose(state, state_chunked, atol=1e-4, rtol=1e-4), (state - state_chunked).abs().max()

    # the same run split in two, to test carrying the state
    split = 10
    first, mid = naive_recurrent_kda(qn[:, :split], kn[:, :split], v[:, :split], g[:, :split], beta[:, :split], scale=scale, output_final_state=True)
    second, last = naive_recurrent_kda(
        qn[:, split:], kn[:, split:], v[:, split:], g[:, split:], beta[:, split:], scale=scale, initial_state=mid, output_final_state=True
    )
    assert torch.allclose(torch.cat([first, second], 1), out, atol=1e-5)
    assert torch.allclose(last, state, atol=1e-5)

    def flat(t: torch.Tensor) -> list[float]:
        return [round(x, 7) for x in t.flatten().tolist()]

    doc = {
        "source": "fla.ops.kda.naive (naive_recurrent_kda / naive_chunk_kda), flash-linear-attention, MIT licence",
        "tokens": tokens, "heads": heads, "k_dim": k_dim, "v_dim": v_dim, "split": split,
        "q": flat(q[0]), "k": flat(k[0]), "v": flat(v[0]), "g": flat(g[0]), "beta": flat(beta[0]),
        "out": flat(out[0]),
        # state layout [head][k][v]
        "state": flat(state[0]), "state_at_split": flat(mid[0]),
    }
    with open(path, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    print(f"wrote {path}: {tokens} tokens, {heads} heads, K={k_dim}, V={v_dim}; max |recurrent - chunked| = {(out - chunked).abs().max():.2e}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rust/engine/testdata/kda_reference.json")
