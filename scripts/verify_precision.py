#!/usr/bin/env python3
"""Decide whether a logit mismatch of a tiny random fixture is a defect or a near-tie of the router.

A random MoE fixture can have a token whose two best experts score almost the same: rounding in the engine (an f16 KV cache, a
different summation order) then picks the other expert and the logits of that position change a lot, although every formula is right.
This script runs the same fixture through

* transformers in float32 and in float64 (experts in the eager implementation, which has no float64 grouped matmul),
* fLLM with an f32 KV cache (`FLLM_VERIFY_KV=f32`),

and prints, per position, how far float32 is from float64 and fLLM from float64, and the positions where the router of any layer
chooses other experts in float32 than in float64. If fLLM with the f32 cache agrees with float64 everywhere, the engine is right.

    TOKENS=45 python scripts/verify_precision.py deepseek_v32_sparse --fllm target/release/fllm-rs \
        --example target/release/examples/verify_checkpoint
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import hf_golden  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arch")
    parser.add_argument("--fllm", required=True)
    parser.add_argument("--example", required=True)
    args = parser.parse_args()
    import transformers as T

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        src = work / "hf"
        golden = hf_golden.build(args.arch, src)
        ids = torch.tensor([golden["ids"]])
        gguf = work / "m.gguf"
        subprocess.run([args.fllm, "convert", str(src), str(gguf), "--outtype", "f32"], check=True, capture_output=True)
        ref = work / "ids.json"
        ref.write_text(json.dumps({"ids": golden["ids"], "text": "hello"}))
        out = subprocess.run([args.example, str(gguf), str(ref)], check=True, capture_output=True, text=True, env={**os.environ, "FLLM_VERIFY_KV": "f32"})
        mine = np.array(json.loads(out.stdout.strip().splitlines()[-1])["logits"])
        loader = getattr(T, {"glm5_next": "Glm5NextForConditionalGeneration"}.get(args.arch, ""), None) or T.AutoModelForCausalLM
        results = {}
        for dtype in (torch.float32, torch.float64):
            model = loader.from_pretrained(src, dtype=dtype, experts_implementation="eager").eval()
            routes = {}
            for name, module in model.named_modules():
                if name.endswith(".mlp.gate"):
                    layer = int(name.split(".")[-3])
                    module.register_forward_hook(lambda m, a, o, layer=layer: routes.__setitem__(layer, o[2].detach().clone()))
            with torch.no_grad(), hf_golden.stable_topk():
                results[dtype] = (model(ids).logits[0].double().numpy(), routes)
    (l32, r32), (l64, r64) = results[torch.float32], results[torch.float64]
    mine = mine.reshape(l64.shape)
    span = float(l64.max() - l64.min())
    show = lambda a, b: [float(f"{v:.2g}") for v in np.abs(a - b).max(axis=1) / span]
    print("transformers float32 against float64, relative per position:", show(l32, l64))
    print("fLLM (f32 KV cache) against float64, relative per position:", show(mine, l64))
    for layer in sorted(r32):
        a, b = np.sort(r32[layer].numpy(), axis=1), np.sort(r64[layer].numpy(), axis=1)
        print(f"layer {layer}: positions whose chosen experts differ between float32 and float64:", np.where((a != b).any(axis=1))[0].tolist())
    worst = max(show(mine, l64))
    print("verdict:", "fLLM agrees with float64" if worst < 1e-4 else f"fLLM is {worst:.2g} away from float64")
    return 0 if worst < 1e-4 else 1


if __name__ == "__main__":
    sys.exit(main())
