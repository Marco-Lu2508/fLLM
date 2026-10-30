#!/usr/bin/env python3
"""Pure PyTorch reference of RWKV-6 ("Finch") and RWKV-7 ("Goose") and the tiny random checkpoints built from it.

transformers 5.22 has no RWKV-6/7 (its `rwkv` is version 4), so the golden logits come from this file: a token-by-token
transcription of the official recurrences (RWKV-LM `RWKV_Tmix_x060` / `RWKV_Tmix_x070`, the numpy reference
`rwkv_v7_numpy.py`, and the Hugging Face remote code `RWKV/v6-Finch-1B6-HF`), written independently of the Rust code.

    python scripts/rwkv_reference.py build rwkv7 OUT_DIR          # tiny checkpoint + golden.json (hf_golden.py calls this)
    python scripts/rwkv_reference.py generate DIR --ids 1,5,9 --new 12   # greedy continuation of a real or tiny checkpoint

Tensor names follow the Hugging Face layouts: RWKV-7 `RWKV7ForCausalLM` (flash-linear-attention: `model.layers.N.attn.*`,
`attn_norm`, `ffn_norm`, `ffn.*`, `pre_norm` in layer 0) and RWKV-6 `Rwkv6ForCausalLM` (`rwkv.blocks.N.attention.*`,
`ln1`, `ln2`, `feed_forward.*`, `pre_ln` in block 0).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

TOKENS = 12


def layer_norm(x, weight, bias, eps):
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def group_norm(x, heads, weight, bias, eps):
    """x: [C]; groups of C/heads values are normalised separately (torch GroupNorm on a single position)."""
    return F.group_norm(x.reshape(1, -1), heads, weight, bias, eps).reshape(-1)


class Rwkv:
    """Token-at-a-time reference; `state` holds the token-shift vectors and the WKV matrices of every layer."""

    def __init__(self, sd: dict, cfg: dict):
        self.sd = {k: v.float() for k, v in sd.items()}
        self.cfg = cfg
        self.version = 7 if cfg["model_type"] == "rwkv7" else 6
        self.layers = cfg["num_hidden_layers"]
        self.hidden = cfg["hidden_size"]
        if self.version == 7:
            self.head = cfg["head_dim"]
            self.eps = cfg.get("norm_eps", 1e-5)
            self.p = "model.layers.{}."
        else:
            self.head = cfg["head_size"]
            self.eps = cfg.get("layer_norm_epsilon", 1e-5)
        self.heads = self.hidden // self.head

    def new_state(self):
        z = lambda *s: torch.zeros(*s)
        return {
            "att": [z(self.hidden) for _ in range(self.layers)],
            "ffn": [z(self.hidden) for _ in range(self.layers)],
            "wkv": [z(self.heads, self.head, self.head) for _ in range(self.layers)],
        }

    # ---- RWKV-7 -----------------------------------------------------------------------------------------------
    def _layer7(self, i, x, st, v_first):
        g = lambda name: self.sd[f"model.layers.{i}.{name}"]
        has = lambda name: f"model.layers.{i}.{name}" in self.sd
        H, N = self.heads, self.head
        h = layer_norm(x, g("attn_norm.weight"), g("attn_norm.bias"), self.eps)
        xx = st["att"][i] - h
        st["att"][i] = h
        xr, xw, xk, xv, xa, xg = (h + xx * g(f"attn.x_{c}").reshape(-1) for c in "rwkvag")
        r = g("attn.r_proj.weight") @ xr
        lora = lambda n, v, act: (act(v @ g(f"attn.{n}_lora.lora.0.weight").T) @ g(f"attn.{n}_lora.lora.2.weight").T)
        w_logit = lora("w", xw, torch.tanh) + g("attn.w_lora.lora.2.bias")
        decay = torch.exp(-0.6065306597126334 * torch.sigmoid(w_logit))
        k = g("attn.k_proj.weight") @ xk
        v = g("attn.v_proj.weight") @ xv
        if i == 0:
            v_first = v
        else:
            mix = torch.sigmoid(lora("v", xv, lambda t: t) + g("attn.v_lora.lora.2.bias"))
            v = v + (v_first - v) * mix
        a = torch.sigmoid(lora("a", xa, lambda t: t) + g("attn.a_lora.lora.2.bias"))
        gate = lora("g", xg, torch.sigmoid)
        kk = F.normalize((k * g("attn.k_k")).reshape(H, N), dim=-1, p=2.0, eps=1e-12).reshape(-1)
        k = k * (1 + (a - 1) * g("attn.k_a"))
        S = st["wkv"][i]  # [H, value, key]
        kkh, ah = kk.reshape(H, N), a.reshape(H, N)
        sa = torch.einsum("hij,hj->hi", S, -kkh)
        S = S * decay.reshape(H, 1, N) + sa[:, :, None] * (kkh * ah)[:, None, :] + v.reshape(H, N, 1) * k.reshape(H, 1, N)
        st["wkv"][i] = S
        y = torch.einsum("hij,hj->hi", S, r.reshape(H, N)).reshape(-1)
        y = group_norm(y, H, g("attn.g_norm.weight"), g("attn.g_norm.bias"), N * self.eps)
        bonus = (r.reshape(H, N) * k.reshape(H, N) * g("attn.r_k")).sum(-1, keepdim=True) * v.reshape(H, N)
        y = y + bonus.reshape(-1)
        x = x + g("attn.o_proj.weight") @ (y * gate)
        h2 = layer_norm(x, g("ffn_norm.weight"), g("ffn_norm.bias"), self.eps)
        xx = st["ffn"][i] - h2
        st["ffn"][i] = h2
        kx = h2 + xx * g("ffn.x_k").reshape(-1)
        x = x + g("ffn.value.weight") @ torch.relu(g("ffn.key.weight") @ kx) ** 2
        return x, v_first

    # ---- RWKV-6 -----------------------------------------------------------------------------------------------
    def _layer6(self, i, x, st):
        g = lambda name: self.sd[f"rwkv.blocks.{i}.{name}"]
        H, N = self.heads, self.head
        h = layer_norm(x, g("ln1.weight"), g("ln1.bias"), self.eps)
        xx = st["att"][i] - h
        st["att"][i] = h
        a = lambda n: g(f"attention.{n}")
        xxx = h + xx * a("time_maa_x").reshape(-1)
        d = a("time_maa_w2").shape[1]
        lo = torch.tanh(xxx @ a("time_maa_w1")).reshape(5, 1, d)
        mix = torch.bmm(lo, a("time_maa_w2")).reshape(5, -1)  # w, k, v, r, g
        xw, xk, xv, xr, xg = (h + xx * (a(f"time_maa_{c}").reshape(-1) + mix[j]) for j, c in enumerate("wkvrg"))
        r = a("receptance.weight") @ xr
        k = a("key.weight") @ xk
        v = a("value.weight") @ xv
        gate = F.silu(a("gate.weight") @ xg)
        w = a("time_decay").reshape(-1) + torch.tanh(xw @ a("time_decay_w1")) @ a("time_decay_w2")
        decay = torch.exp(-torch.exp(w)).reshape(H, N)
        u = a("time_faaaa")  # [H, N]
        S = st["wkv"][i]  # [H, key, value]
        kv = k.reshape(H, N, 1) * v.reshape(H, 1, N)
        y = torch.einsum("hk,hkv->hv", r.reshape(H, N), u[:, :, None] * kv + S).reshape(-1)
        st["wkv"][i] = kv + decay[:, :, None] * S
        eps = 1e-5 * (self.cfg.get("head_size_divisor", 8) ** 2)
        y = group_norm(y, H, a("ln_x.weight"), a("ln_x.bias"), eps)
        x = x + a("output.weight") @ (y * gate)
        h2 = layer_norm(x, g("ln2.weight"), g("ln2.bias"), self.eps)
        xx = st["ffn"][i] - h2
        st["ffn"][i] = h2
        f = lambda n: g(f"feed_forward.{n}")
        xk = h2 + xx * f("time_maa_k").reshape(-1)
        xr = h2 + xx * f("time_maa_r").reshape(-1)
        kk = torch.relu(f("key.weight") @ xk) ** 2
        x = x + torch.sigmoid(f("receptance.weight") @ xr) * (f("value.weight") @ kk)
        return x

    def step(self, token: int, st):
        sd = self.sd
        if self.version == 7:
            x = sd["model.embeddings.weight"][token]
            if "model.layers.0.pre_norm.weight" in sd:
                x = layer_norm(x, sd["model.layers.0.pre_norm.weight"], sd["model.layers.0.pre_norm.bias"], self.eps)
            v_first = None
            for i in range(self.layers):
                x, v_first = self._layer7(i, x, st, v_first)
            x = layer_norm(x, sd["model.norm.weight"], sd["model.norm.bias"], self.eps)
            return sd["lm_head.weight"] @ x
        x = sd["rwkv.embeddings.weight"][token]
        if "rwkv.blocks.0.pre_ln.weight" in sd:
            x = layer_norm(x, sd["rwkv.blocks.0.pre_ln.weight"], sd["rwkv.blocks.0.pre_ln.bias"], self.eps)
        for i in range(self.layers):
            x = self._layer6(i, x, st)
        x = layer_norm(x, sd["rwkv.ln_out.weight"], sd["rwkv.ln_out.bias"], self.eps)
        return sd["head.weight"] @ x

    def logits(self, ids, st=None):
        st = st or self.new_state()
        return torch.stack([self.step(int(t), st) for t in ids]), st

    def generate(self, ids, new):
        out = []
        logits, st = self.logits(ids)
        for _ in range(new):
            token = int(logits[-1].argmax())
            out.append(token)
            logits, st = self.logits([token], st)
        return out


# ---- tiny checkpoints -----------------------------------------------------------------------------------------------

def tiny_config(version: int, vocab: int) -> dict:
    if version == 7:
        return {
            "architectures": ["RWKV7ForCausalLM"], "model_type": "rwkv7", "hidden_size": 64, "num_hidden_layers": 3, "head_dim": 16,
            "num_heads": None, "decay_low_rank_dim": 8, "a_low_rank_dim": 8, "v_low_rank_dim": 8, "gate_low_rank_dim": 16,
            "hidden_ratio": 4.0, "intermediate_size": 256, "norm_eps": 1e-5, "norm_bias": True, "norm_first": True,
            "hidden_act": "sqrelu", "vocab_size": vocab, "bos_token_id": 1, "eos_token_id": 2, "tie_word_embeddings": False,
            "max_position_embeddings": 2048, "torch_dtype": "float32",
        }
    return {
        "architectures": ["Rwkv6ForCausalLM"], "model_type": "rwkv6", "hidden_size": 64, "attention_hidden_size": 64, "num_hidden_layers": 3,
        "head_size": 16, "head_size_divisor": 8, "num_attention_heads": 4, "intermediate_size": 224, "layer_norm_epsilon": 1e-5,
        "rescale_every": 0, "vocab_size": vocab, "bos_token_id": 1, "eos_token_id": 2, "tie_word_embeddings": False,
        "torch_dtype": "float32",
    }


def random_state_dict(cfg: dict, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    rn = lambda *s, std=0.15: torch.randn(*s, generator=gen) * std
    ru = lambda *s, lo=0.0, hi=1.0: lo + (hi - lo) * torch.rand(*s, generator=gen)
    C, V, L = cfg["hidden_size"], cfg["vocab_size"], cfg["num_hidden_layers"]
    sd = {}
    ln = lambda name: (sd.__setitem__(f"{name}.weight", ru(C, lo=0.8, hi=1.2)), sd.__setitem__(f"{name}.bias", rn(C, std=0.1)))
    if cfg["model_type"] == "rwkv7":
        N = cfg["head_dim"]
        H = C // N
        sd["model.embeddings.weight"] = rn(V, C, std=1.0)
        sd["lm_head.weight"] = rn(V, C, std=0.3)
        ln("model.layers.0.pre_norm")
        ln("model.norm")
        for i in range(L):
            p = f"model.layers.{i}."
            ln(p + "attn_norm")
            ln(p + "ffn_norm")
            for c in "rwkvag":
                sd[p + f"attn.x_{c}"] = ru(1, 1, C)
            for n, rank, bias, has in (("w", cfg["decay_low_rank_dim"], True, True), ("a", cfg["a_low_rank_dim"], True, True),
                                       ("v", cfg["v_low_rank_dim"], True, i > 0), ("g", cfg["gate_low_rank_dim"], False, True)):
                if not has:
                    continue
                sd[p + f"attn.{n}_lora.lora.0.weight"] = rn(rank, C, std=0.3)
                sd[p + f"attn.{n}_lora.lora.2.weight"] = rn(C, rank, std=0.3)
                if bias:
                    sd[p + f"attn.{n}_lora.lora.2.bias"] = rn(C, std=0.5)
            sd[p + "attn.k_k"] = ru(C, lo=0.5, hi=1.5)
            sd[p + "attn.k_a"] = ru(C)
            sd[p + "attn.r_k"] = rn(H, N, std=0.3)
            for n in ("r", "k", "v", "o"):
                sd[p + f"attn.{n}_proj.weight"] = rn(C, C)
            ln(p + "attn.g_norm")
            sd[p + "ffn.x_k"] = ru(1, 1, C)
            sd[p + "ffn.key.weight"] = rn(cfg["intermediate_size"], C)
            sd[p + "ffn.value.weight"] = rn(C, cfg["intermediate_size"])
        return sd
    N = cfg["head_size"]
    H = C // N
    D, DD, F_ = 8, 12, cfg["intermediate_size"]
    sd["rwkv.embeddings.weight"] = rn(V, C, std=1.0)
    sd["head.weight"] = rn(V, C, std=0.3)
    ln("rwkv.blocks.0.pre_ln")
    ln("rwkv.ln_out")
    for i in range(L):
        p = f"rwkv.blocks.{i}."
        ln(p + "ln1")
        ln(p + "ln2")
        a = p + "attention."
        for c in "xwkvrg":
            sd[a + f"time_maa_{c}"] = ru(1, 1, C)
        sd[a + "time_maa_w1"] = rn(C, 5 * D, std=0.3)
        sd[a + "time_maa_w2"] = rn(5, D, C, std=0.3)
        sd[a + "time_decay"] = rn(1, 1, C, std=0.6) - 0.5
        sd[a + "time_decay_w1"] = rn(C, DD, std=0.3)
        sd[a + "time_decay_w2"] = rn(DD, C, std=0.3)
        sd[a + "time_faaaa"] = rn(H, N, std=0.3)
        for n in ("receptance", "key", "value", "gate", "output"):
            sd[a + f"{n}.weight"] = rn(C, C)
        ln(a + "ln_x")
        f = p + "feed_forward."
        sd[f + "time_maa_k"] = ru(1, 1, C)
        sd[f + "time_maa_r"] = ru(1, 1, C)
        sd[f + "key.weight"] = rn(F_, C)
        sd[f + "receptance.weight"] = rn(C, C)
        sd[f + "value.weight"] = rn(C, F_)
    return sd


def build(arch: str, out: Path, vocab: int = 320, seed: int = 1234) -> dict:
    from safetensors.torch import save_file

    sys.path.insert(0, str(Path(__file__).parent))
    import hf_golden

    out.mkdir(parents=True, exist_ok=True)
    tok = hf_golden.make_tokenizer(out, vocab)
    vocab = tok.get_vocab_size()
    cfg = tiny_config(7 if arch == "rwkv7" else 6, vocab)
    sd = random_state_dict(cfg, seed)
    save_file({k: v.contiguous() for k, v in sd.items()}, str(out / "model.safetensors"), metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    g = torch.Generator().manual_seed(seed + 1)
    ids = torch.randint(4, vocab, (TOKENS,), generator=g)
    with torch.no_grad():
        logits, _ = Rwkv(sd, cfg).logits(ids.tolist())
    golden = {"arch": arch, "ids": ids.tolist(), "vocab": vocab, "logits": logits.reshape(-1).tolist(), "shape": list(logits.shape)}
    (out / "golden.json").write_text(json.dumps(golden))
    return golden


def load_dir(directory: Path) -> Rwkv:
    from safetensors.torch import load_file

    cfg = json.loads((directory / "config.json").read_text())
    sd = {}
    for f in sorted(directory.glob("*.safetensors")):
        sd.update(load_file(str(f)))
    if not sd:
        sd = torch.load(directory / "pytorch_model.bin", map_location="cpu", weights_only=True)
    return Rwkv(sd, cfg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("arch", choices=["rwkv6", "rwkv7"])
    b.add_argument("output")
    b.add_argument("--vocab", type=int, default=320)
    g = sub.add_parser("generate")
    g.add_argument("directory")
    g.add_argument("--ids", required=True)
    g.add_argument("--new", type=int, default=12)
    args = ap.parse_args()
    if args.cmd == "build":
        golden = build(args.arch, Path(args.output), args.vocab)
        print(f"{args.arch}: {golden['shape']} logits written to {args.output}")
        return 0
    model = load_dir(Path(args.directory))
    ids = [int(t) for t in args.ids.split(",")]
    with torch.no_grad():
        print(json.dumps({"ids": ids, "generated": model.generate(ids, args.new)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
