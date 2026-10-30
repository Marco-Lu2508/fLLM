#!/usr/bin/env python3
"""Convert tiny reference checkpoints with fllm-rs and compare fLLM logits with transformers.

    python scripts/verify_architectures.py --fllm target/release/fllm-rs \
        --example target/release/examples/verify_checkpoint llama qwen3 ...

For each architecture the script builds a random checkpoint (hf_golden.py), converts it to f32 GGUF,
runs the same token sequence through fLLM and reports the largest logit difference relative to the
logit range and the number of positions whose arg-max differs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import hf_golden  # noqa: E402


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def verify(arch: str, fllm: str, example: str, workdir: Path, outtype: str, keep: bool) -> dict:
    src = workdir / f"{arch}-hf"
    golden = hf_golden.build(arch, src)
    gguf = workdir / f"{arch}.gguf"
    run([fllm, "convert", str(src), str(gguf), "--outtype", outtype])
    ref = workdir / f"{arch}-ids.json"
    ref.write_text(json.dumps({"ids": golden["ids"], "text": "hello"}))
    out = run([example, str(gguf), str(ref)])
    mine = json.loads(out.stdout.strip().splitlines()[-1])["logits"]
    want = golden["logits"]
    if len(mine) != len(want):
        return {"arch": arch, "ok": False, "error": f"logit count {len(mine)} != {len(want)}"}
    positions, vocab = golden["shape"]
    worst = 0.0
    worst_at = 0
    span = max(want) - min(want)
    mismatches = 0
    for p in range(positions):
        a = mine[p * vocab : (p + 1) * vocab]
        b = want[p * vocab : (p + 1) * vocab]
        position_worst = max(abs(x - y) for x, y in zip(a, b))
        if position_worst > worst:
            worst, worst_at = position_worst, p
        # a different arg-max only counts when the reference prefers it clearly (not a near-tie)
        mine_top = max(range(vocab), key=a.__getitem__)
        ref_top = max(range(vocab), key=b.__getitem__)
        if mine_top != ref_top and b[ref_top] - b[mine_top] > 4 * max(abs(x - y) for x, y in zip(a, b)):
            mismatches += 1
    result = {
        "arch": arch,
        "max_abs": worst,
        "relative": worst / span if span else 0.0,
        "argmax_mismatches": mismatches,
        "positions": positions,
        "worst_position": worst_at,
    }
    result["ok"] = result["relative"] < 2e-3 and mismatches == 0
    if not keep:
        for path in (gguf, ref):
            path.unlink(missing_ok=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archs", nargs="+")
    parser.add_argument("--fllm", required=True)
    parser.add_argument("--example", required=True)
    parser.add_argument("--outtype", default="f32")
    parser.add_argument("--workdir")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(args.workdir or tmp)
        work.mkdir(parents=True, exist_ok=True)
        for arch in args.archs:
            try:
                r = verify(arch, args.fllm, args.example, work, args.outtype, args.keep)
            except subprocess.CalledProcessError as e:
                r = {"arch": arch, "ok": False, "error": (e.stderr or e.stdout or str(e))[-600:]}
            failed += not r["ok"]
            print(json.dumps(r))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
