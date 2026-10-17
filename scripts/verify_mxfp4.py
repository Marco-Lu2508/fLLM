#!/usr/bin/env python3
"""Check the GPT-OSS MXFP4 import path.

Builds a tiny GPT-OSS checkpoint, then writes a second copy whose experts are stored as MXFP4
blocks (random codes and scales) and a third copy holding exactly the dequantised values as dense
tensors. Both are converted and run through fLLM; the logits must agree.

    python scripts/verify_mxfp4.py --fllm target/release/fllm-rs \
        --example target/release/examples/verify_checkpoint
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).parent))
import hf_golden  # noqa: E402

FP4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def random_blocks(experts, rows, nb, generator):
    codes = torch.randint(0, 16, (experts, rows, nb, 32), generator=generator)
    scales = torch.randint(122, 128, (experts, rows, nb), generator=generator, dtype=torch.int64)
    # HF layout: byte j holds elements 2j (low nibble) and 2j+1 (high nibble)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    table = torch.tensor(FP4)
    values = table[codes] * torch.pow(2.0, (scales - 127).float())[..., None]
    return packed, scales.to(torch.uint8), values.reshape(experts, rows, nb * 32)


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def logits(fllm, example, source, work, name, ids):
    gguf = work / f"{name}.gguf"
    run([fllm, "convert", str(source), str(gguf), "--outtype", "f32", "--overwrite"])
    request = work / f"{name}-ids.json"
    request.write_text(json.dumps({"ids": ids, "text": "x"}))
    return json.loads(run([example, str(gguf), str(request)]).strip().splitlines()[-1])["logits"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fllm", required=True)
    parser.add_argument("--example", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        golden = hf_golden.build("gpt_oss", work / "base")
        tensors = load_file(str(work / "base" / "model.safetensors"))
        g = torch.Generator().manual_seed(7)
        dense, packed = dict(tensors), dict(tensors)
        for key in list(tensors):
            if not key.endswith("mlp.experts.gate_up_proj"):
                continue
            prefix = key[: -len("gate_up_proj")]
            e, hidden, double = tensors[key].shape
            ff = tensors[prefix + "down_proj"].shape[1]
            blocks, scales, values = random_blocks(e, double, hidden // 32, g)
            dense[key] = values.transpose(1, 2).contiguous()  # [E, H, 2F]
            del packed[key]
            packed[prefix + "gate_up_proj_blocks"] = blocks
            packed[prefix + "gate_up_proj_scales"] = scales
            blocks, scales, values = random_blocks(e, hidden, ff // 32, g)
            dense[prefix + "down_proj"] = values.transpose(1, 2).contiguous()  # [E, F, H]
            del packed[prefix + "down_proj"]
            packed[prefix + "down_proj_blocks"] = blocks
            packed[prefix + "down_proj_scales"] = scales
        for name, content in (("dense", dense), ("packed", packed)):
            target = work / name
            shutil.copytree(work / "base", target)
            (target / "model.safetensors").unlink()
            (target / "golden.json").unlink()
            save_file({k: v.contiguous() for k, v in content.items()}, str(target / "model.safetensors"))
        a = logits(args.fllm, args.example, work / "dense", work, "dense", golden["ids"])
        b = logits(args.fllm, args.example, work / "packed", work, "packed", golden["ids"])
        worst = max(abs(x - y) for x, y in zip(a, b))
        span = max(a) - min(a)
        print(json.dumps({"max_abs": worst, "relative": worst / span}))
        # the packed path quantizes activations to 8 bits (about 0.4% per projection), so allow for that
        return 0 if worst / span < 2e-2 else 1


if __name__ == "__main__":
    sys.exit(main())
