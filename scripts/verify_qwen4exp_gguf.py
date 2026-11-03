#!/usr/bin/env python3
"""Check fLLM's reader of the llama.cpp layout of Qwen3.8-Flash-Next (`qwen4exp` GGUF) on a tiny random model.

    python scripts/verify_qwen4exp_gguf.py --fllm target/release/fllm-rs --example target/release/examples/verify_checkpoint \
        [--llama-logits ./llama_cpp_logits] [--workdir DIR] [--keep]

One random checkpoint (the `qwen4_exp_gguf` fixture of hf_golden.py, with transformers' reference logits) goes three ways through fLLM:

1. `fllm-rs convert` (fLLM's own layout, GGUF `qwen3next` + `hc.*` / `indexer.*` / `ple.*`): the path that is checked against transformers;
2. `write_qwen4exp_gguf.py` (the llama.cpp layout: other metadata keys, other tensor names, tiled value heads, interleaved query / gate rows,
   `1 + w` norms, split indexer projections), unfused and `--merge` (fused shared-input weights), read by fLLM's `qwen4exp` reader;
3. optionally the same llama.cpp-layout file through llama.cpp itself (`llama_cpp_logits.cpp`), which shows that the file is a valid
   `qwen4exp` file and that its logits are the model's.

Every pair is compared position by position: the largest logit difference relative to the logit range and the arg-max mismatches.
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


def compare(name: str, have: list[float], want: list[float], positions: int, vocab: int) -> dict:
    if len(have) != len(want):
        return {"pair": name, "ok": False, "error": f"logit count {len(have)} != {len(want)}"}
    span = max(want) - min(want)
    worst, worst_at, mismatches = 0.0, 0, 0
    for p in range(positions):
        a, b = have[p * vocab:(p + 1) * vocab], want[p * vocab:(p + 1) * vocab]
        diff = max(abs(x - y) for x, y in zip(a, b))
        if diff > worst:
            worst, worst_at = diff, p
        top_a, top_b = max(range(vocab), key=a.__getitem__), max(range(vocab), key=b.__getitem__)
        if top_a != top_b and b[top_b] - b[top_a] > 4 * diff:
            mismatches += 1
    result = {"pair": name, "max_abs": worst, "relative": worst / span if span else 0.0, "argmax_mismatches": mismatches, "worst_position": worst_at}
    result["ok"] = result["relative"] < 2e-3 and mismatches == 0
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fllm", required=True)
    ap.add_argument("--example", required=True)
    ap.add_argument("--llama-logits", help="binary built from llama_cpp_logits.cpp")
    ap.add_argument("--gguf", help="compare fLLM with llama.cpp on this qwen4exp GGUF (for example a slice of a real file made by slice_qwen4exp_gguf.py) and nothing else; needs --llama-logits")
    ap.add_argument("--ids", default="17,2023,31,590,40117,5,992,13,271,80,1500,6", help="token ids for --gguf")
    ap.add_argument("--workdir")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    here = Path(__file__).parent
    if args.gguf:
        ids = [int(x) for x in args.ids.split(",")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ids.json"
            path.write_text(json.dumps({"ids": ids, "text": "hello"}))
            mine = json.loads(run([args.example, args.gguf, str(path)]).stdout.strip().splitlines()[-1])["logits"]
        ref = json.loads(run([args.llama_logits, args.gguf] + [str(i) for i in ids]).stdout.strip().splitlines()[-1])["logits"]
        result = compare("fLLM vs llama.cpp on " + Path(args.gguf).name, mine, ref, len(ids), len(ref) // len(ids))
        print(json.dumps(result))
        return 0 if result["ok"] else 1
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(args.workdir or tmp)
        work.mkdir(parents=True, exist_ok=True)
        golden = hf_golden.build("qwen4_exp_gguf", work / "hf")
        positions, vocab = golden["shape"]
        ids = work / "ids.json"
        ids.write_text(json.dumps({"ids": golden["ids"], "text": "hello"}))
        files = {"fllm layout (convert)": work / "fllm.gguf", "llama.cpp layout": work / "llamacpp.gguf", "llama.cpp layout, fused": work / "llamacpp-merged.gguf"}
        for path in files.values():
            path.unlink(missing_ok=True)
        run([args.fllm, "convert", str(work / "hf"), str(files["fllm layout (convert)"]), "--outtype", "f32"])
        run([sys.executable, str(here / "write_qwen4exp_gguf.py"), str(work / "hf"), str(files["llama.cpp layout"])])
        run([sys.executable, str(here / "write_qwen4exp_gguf.py"), str(work / "hf"), str(files["llama.cpp layout, fused"]), "--merge"])
        logits = {}
        for name, path in files.items():
            out = run([args.example, str(path), str(ids)])
            logits[name] = json.loads(out.stdout.strip().splitlines()[-1])["logits"]
        results = [compare(f"fLLM reads '{name}' vs transformers", logits[name], golden["logits"], positions, vocab) for name in files]
        results.append(compare("fLLM llama.cpp layout vs fLLM own layout", logits["llama.cpp layout"], logits["fllm layout (convert)"], positions, vocab))
        results.append(compare("fLLM fused vs unfused llama.cpp layout", logits["llama.cpp layout, fused"], logits["llama.cpp layout"], positions, vocab))
        if args.llama_logits:
            out = run([args.llama_logits, str(files["llama.cpp layout"])] + [str(i) for i in golden["ids"]])
            ref = json.loads(out.stdout.strip().splitlines()[-1])["logits"]
            results.append(compare("llama.cpp on the llama.cpp layout vs transformers", ref, golden["logits"], positions, vocab))
            results.append(compare("fLLM llama.cpp layout vs llama.cpp", logits["llama.cpp layout"], ref, positions, vocab))
        for r in results:
            print(json.dumps(r))
        return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
