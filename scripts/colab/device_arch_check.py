#!/usr/bin/env python3
"""GPU check of device paths that have no self test of their own: tiny random checkpoints of Nemotron-H with latent-space experts and of Apertus (xIELU),
scored token by token on the CPU and on the GPU with `fllm-rs ppl --dump-nll`, with and without the new device paths.

    python3 scripts/colab/device_arch_check.py --fllm ./fllm-rs [--device cuda] [--archs nemotron_h_latent,apertus]

Needs torch, transformers and tokenizers (Colab has torch: `pip install -U transformers tokenizers`). Exit status 1 when a run differs by more than --tol
(largest difference of the per-token log-probabilities) or fails.
"""
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hf_golden  # noqa: E402

VARIANTS = {
    "nemotron_h_latent": [("host (default)", {}), ("device", {"FLLM_LATENT": "device"})],
    "apertus": [("device xIELU (default)", {}), ("host xIELU", {"FLLM_XIELU_HOST": "1"})],
}


def sh(cmd, env=None):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run(cmd, capture_output=True, text=True, env=e)


def nll(fllm, gguf, ids, device, env, out):
    r = sh([fllm, "ppl", "--model", str(gguf), "--token-ids", str(ids), "--device", device, "--dump-nll", str(out)], env)
    if r.returncode != 0:
        return None, (r.stdout + r.stderr)[-1500:]
    vals = []
    for line in Path(out).read_text().splitlines():
        parts = line.split()
        if len(parts) == 3:
            vals.append(float(parts[2]))
    return vals, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fllm", default="./fllm-rs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--archs", default="nemotron_h_latent,apertus")
    ap.add_argument("--tol", type=float, default=5e-3)
    args = ap.parse_args()
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for arch in args.archs.split(","):
            src = work / f"{arch}-hf"
            golden = hf_golden.build(arch, src)
            gguf = work / f"{arch}.gguf"
            r = sh([args.fllm, "convert", str(src), str(gguf), "--outtype", "f32"])
            if r.returncode != 0:
                print(f"{arch}: convert failed\n{(r.stdout + r.stderr)[-1500:]}")
                bad += 1
                continue
            ids = work / f"{arch}-ids.txt"
            ids.write_text(" ".join(str(i) for i in golden["ids"]))
            want, err = nll(args.fllm, gguf, ids, "cpu", {}, work / "cpu.nll")
            if want is None:
                print(f"{arch}: CPU run failed\n{err}")
                bad += 1
                continue
            for label, env in VARIANTS.get(arch, [("default", {})]):
                env = dict(env, FLLM_CUDA_EXACT="1")
                have, err = nll(args.fllm, gguf, ids, args.device, env, work / "dev.nll")
                if have is None or len(have) != len(want):
                    print(f"{arch:20} {label:24} FAILED  {err or 'token count differs'}")
                    bad += 1
                    continue
                worst = max(abs(a - b) for a, b in zip(want, have))
                ok = worst <= args.tol
                bad += 0 if ok else 1
                print(f"{arch:20} {label:24} {'ok' if ok else 'DIFFERS'}  max |dlogp| = {worst:.2e}  (tolerance {args.tol:.0e}, {len(want)} tokens)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
