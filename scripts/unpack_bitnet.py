#!/usr/bin/env python3
"""Writes the dense (unpacked) equivalent of a Hugging Face packed BitNet checkpoint: every `uint8` weight becomes
`(code - 1) * weight_scale` in bfloat16, the `quantization_config` is removed. transformers loads the result as a plain BitNet model,
which is the reference for fLLM's packed-ternary reader (neither applies the 8-bit activation rounding of the packed variant).

    python scripts/unpack_bitnet.py PACKED_DIR DENSE_DIR
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.integrations.bitnet import unpack_weights


def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.suffix != ".safetensors" and f.name != "config.json" and f.is_file():
            shutil.copy(f, dst / f.name)
    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("quantization_config", None)
    cfg.pop("auto_map", None)
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    out = {}
    for shard in sorted(src.glob("*.safetensors")):
        with safe_open(shard, "pt") as f:
            names = list(f.keys())
            for name in names:
                if name.endswith(".weight_scale"):
                    continue
                t = f.get_tensor(name)
                if t.dtype == torch.uint8:
                    scale = f.get_tensor(name.replace(".weight", ".weight_scale")).float()
                    t = (unpack_weights(t, torch.float32) * scale).to(torch.bfloat16)
                out[name] = t.contiguous()
    save_file(out, dst / "model.safetensors", metadata={"format": "pt"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
