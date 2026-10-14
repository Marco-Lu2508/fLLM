#!/usr/bin/env python3
"""Parse and verify the StableHLO programs of the TPU backend with a real MLIR StableHLO parser (no TPU needed).

The unit tests of the backend run every operation on a backend without a device and, with FLLM_TPU_DUMP_DIR set, write each program
they build to a .mlir file. This script reads such a directory with the StableHLO dialect that ships in `jaxlib`, parses every file
and runs the dialect's verifiers (shapes, element types, attributes). It is a syntax and type check; it does not run anything.

    pip install jax                       # jaxlib contains the MLIR bindings
    FLLM_TPU_DUMP_DIR=/tmp/tpu-programs cargo test --release -p fllm-engine --features tpu-pjrt backend::tpu
    python scripts/colab/stablehlo_verify.py /tmp/tpu-programs [--print FILE]

Exit status 0: every program parsed and verified. With --print the canonical (pretty-printed) text of one file is shown, which is
easier to read than the generic form the emitter writes.
"""

import argparse
import glob
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", help="directory with .mlir files (searched recursively)")
    parser.add_argument("--print", dest="show", help="print this file in canonical form instead of verifying the directory")
    args = parser.parse_args()

    try:
        from jax._src.interpreters import mlir  # the context that has the StableHLO dialect registered
        from jax._src.lib.mlir import ir
    except ImportError as error:
        sys.exit(f"jax/jaxlib is needed for the MLIR bindings (pip install jax): {error}")

    context = mlir.make_ir_context()
    if args.show:
        with context:
            print(ir.Module.parse(open(args.show).read()))
        return

    files = sorted(glob.glob(os.path.join(args.directory, "**", "*.mlir"), recursive=True))
    if not files:
        sys.exit(f"no .mlir files below {args.directory}")
    failed = 0
    for path in files:
        try:
            with context:
                module = ir.Module.parse(open(path).read())
                if not module.operation.verify():
                    raise RuntimeError("verification failed")
        except Exception as error:  # noqa: BLE001 - report every file
            failed += 1
            print("FAIL", path, str(error)[:800].replace("\n", " | "))
    print(f"{len(files)} programs, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
