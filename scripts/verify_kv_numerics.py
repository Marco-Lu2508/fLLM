#!/usr/bin/env python3
"""Tells numerics from bugs when the greedy text of a real checkpoint diverges from transformers.

    python scripts/verify_kv_numerics.py --fllm target/release/fllm-rs --model DIR --reference DIR [--tokens 24]

For every prompt of `verify_real_model.py` the script runs greedy decoding in transformers (float32) and in `fllm-rs serve` on the CPU
with the f16 KV cache and with the f32 KV cache. At the first token where an fLLM text differs from the reference it prints the
margin between the two candidate tokens in the reference logits (reference top-1 minus the logit of the token fLLM chose). A small
margin (a few 1e-2 for models whose logits span ten or more) means the flip is rounding: f16 keys/values or bf16 weights are enough to
move it. If the f32 cache removes the divergences, the f16 cache was the cause; if both diverge with a large margin, look for a bug.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_real_model as vrm  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fllm", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reference")
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    ref_dir = args.reference or args.model

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ref_dir)
    model = AutoModelForCausalLM.from_pretrained(ref_dir, dtype=torch.float32).eval()
    texts = {kind: vrm.fllm_texts(args.fllm, args.model, args.tokens, args.threads, ["--kv-cache-type", kind]) for kind in ("f16", "f32")}
    for index, prompt in enumerate(vrm.PROMPTS):
        ids = tok(prompt, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            gen = model.generate(input_ids=ids, max_new_tokens=args.tokens, do_sample=False)[0][ids.shape[1]:].tolist()
        want = tok.decode(gen, skip_special_tokens=True)
        row = {"prompt": prompt}
        for kind in ("f16", "f32"):
            mine = texts[kind][index]
            same = 0
            for a, b in zip(mine, want):
                if a != b:
                    break
                same += 1
            entry = {"identical": mine == want, "same_leading_chars": same, "of": len(want)}
            if mine != want:
                # position of the first differing token: encode the common prefix and compare token by token
                mine_ids = tok(prompt + mine, return_tensors="pt")["input_ids"][0][ids.shape[1]:].tolist()
                at = next((i for i, (a, b) in enumerate(zip(mine_ids, gen)) if a != b), None)
                if at is not None:
                    prefix = torch.tensor([ids[0].tolist() + gen[:at]])
                    with torch.no_grad():
                        logits = model(input_ids=prefix).logits[0, -1]
                    top = float(logits[gen[at]])
                    other = float(logits[mine_ids[at]])
                    entry.update({"token_index": at, "reference_top1_minus_fllm_choice": round(top - other, 4), "logit_span": round(float(logits.max() - logits.min()), 2)})
            row[kind] = entry
        print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
