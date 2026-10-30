#!/usr/bin/env python3
"""BitNet checkpoints in the Hugging Face `quant_method: "bitnet"` layout (uint8 weights with four ternary codes per byte plus a
`weight_scale`), converted by fllm-rs and compared with transformers.

A tiny random BitNet model (the `bitnet` fixture of hf_golden.py) gets ternary linear weights (`round(w / mean|w|)` times the scale);
transformers' own `pack_weights` writes the packed checkpoint, the reference logits come from the dense model holding the same
ternary values.

    python scripts/verify_bitnet_packed.py --fllm target/release/fllm-rs --example target/release/examples/verify_checkpoint
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).parent))
import hf_golden  # noqa: E402

LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fllm", required=True)
    parser.add_argument("--example", required=True)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM
    from transformers.integrations.bitnet import pack_weights

    work = Path(tempfile.mkdtemp(prefix="bitnet_packed_"))
    dense_dir, packed_dir = work / "dense", work / "packed"
    hf_golden.build("bitnet", dense_dir)
    model = AutoModelForCausalLM.from_pretrained(dense_dir, dtype=torch.float32)
    tensors = load_file(dense_dir / "model.safetensors")
    out = {}
    with torch.no_grad():
        for name, w in tensors.items():
            if name.endswith(".weight") and any(f".{lin}." in name for lin in LINEARS):
                scale = w.abs().mean().clamp(min=1e-5)
                codes = (w / scale).round().clamp(-1, 1)
                scale = torch.tensor(float(scale.to(torch.bfloat16)))  # the checkpoint stores the scale in bfloat16
                out[name] = pack_weights(codes.clone()).contiguous()
                out[name.replace(".weight", ".weight_scale")] = scale.reshape(1).to(torch.bfloat16)
                tensors[name] = codes * scale
            else:
                out[name] = w
        model.load_state_dict(tensors, strict=False)
        model.eval()
        golden = json.loads((dense_dir / "golden.json").read_text())
        ids = torch.tensor([golden["ids"]])
        want = model(ids).logits[0].float()
    packed_dir.mkdir()
    for f in dense_dir.iterdir():
        if f.name not in ("model.safetensors", "golden.json", "config.json"):
            (packed_dir / f.name).write_bytes(f.read_bytes())
    cfg = json.loads((dense_dir / "config.json").read_text())
    cfg["quantization_config"] = {"quant_method": "bitnet", "linear_class": "autobitlinear", "quantization_mode": "offline"}
    (packed_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    save_file(out, packed_dir / "model.safetensors", metadata={"format": "pt"})

    gguf = work / "bitnet_packed.gguf"
    subprocess.run([args.fllm, "convert", str(packed_dir), str(gguf), "--outtype", "f32"], check=True, capture_output=True, text=True)
    ref = work / "ids.json"
    ref.write_text(json.dumps({"ids": golden["ids"], "text": "hello"}))
    res = subprocess.run([args.example, str(gguf), str(ref)], check=True, capture_output=True, text=True)
    mine = torch.tensor(json.loads(res.stdout.strip().splitlines()[-1])["logits"]).reshape(want.shape)
    worst = float((mine - want).abs().max())
    span = float(want.max() - want.min())
    mismatches = int((mine.argmax(-1) != want.argmax(-1)).sum())
    print(json.dumps({"arch": "bitnet_packed", "max_abs": worst, "relative": worst / span, "argmax_mismatches": mismatches, "ok": worst / span < 2e-3 and mismatches == 0}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
