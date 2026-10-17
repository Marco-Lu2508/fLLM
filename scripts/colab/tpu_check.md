# TPU test on Google Colab (guide for `tpu_check.py`)

State: **L0**. The TPU backend (`--device tpu`, Cargo feature `tpu-pjrt`) is written and compiles, and the generated StableHLO programs
were checked by an MLIR StableHLO parser, but they have **never been executed on a TPU**. This script is the first real check:
it builds fLLM on Colab, lists the TPU cores, runs the self test against the CPU backend and (with a model) compares the
greedy text of TPU and CPU. A failure is expected and wanted; the report says where.

## Requirements

- A Colab runtime with a TPU: *Runtime -> Change runtime type -> TPU v5e-1* (or *v2-8*, if offered). Then **restart the runtime**
  so that no notebook kernel occupies the TPU yet (a TPU can be opened by only one client per process; an earlier `import jax` in the
  kernel is enough to lock it).
- The fLLM source on the VM: `git clone <repository URL> fllm` or an uploaded archive. The script uses the checkout it lives in
  (`--source DIR` for a different one).
- Internet access (rustup, `pip install libtpu`, optionally `fllm-rs pull hf://...`).

## Invocation (one notebook cell)

```python
!python fllm/scripts/colab/tpu_check.py \
    --model-uri hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf
```

Without `--model`/`--model-uri` only the last step is skipped. Useful switches: `--skip build` (the binary already exists, give it with
`--bin PATH`), `--libtpu PATH` (your own `libtpu.so`), `--jobs N` (Cargo jobs; Colab has few cores and little RAM),
`--prompt`, `--tokens`.

## What the steps do and what a failure means

| Step | Checks | Typical cause of failure |
|---|---|---|
| environment | TPU device files (`/dev/accel*`, `/dev/vfio`) | the runtime is not a TPU runtime |
| rust | `cargo` is present, otherwise `rustup` | no internet |
| libtpu | `libtpu.so` from the pip package `libtpu` (the path is set as `FLLM_TPU_LIBRARY`) | package cannot be installed; give the path with `--libtpu` |
| build | `cargo build --release -p fllm-engine --features tpu-pjrt` (10-25 min) | low memory: `--jobs 2`; compiler errors are in `tpu_check_build.log` |
| devices | `fllm-rs devices` lists cores with `tpu` in the line | plugin does not load (message `[fllm-rs] TPU backend: ...`), TPU occupied, PJRT API version mismatch |
| selftest | `selftest-gpu --device tpu --json`, exact arithmetic (`FLLM_TPU_EXACT=1`) | see below |
| model | server on the TPU, greedy text, then the same server on the CPU; the texts must be equal | TPU text differs -> level L2 not reached |

Expected weaknesses of the first run (known):

- **Compile time**: every new shape of a program is translated once by the TPU compiler (on the order of 0.1-1 s). The self test
  and the first request with a new prompt length are therefore slow; the second request of the same shape is not. With
  `FLLM_TPU_DEBUG=1` (the script sets it) every translation appears in the log.
- **Accuracy**: the ULP bounds of the self test assume exact f32 arithmetic. The script sets `FLLM_TPU_EXACT=1` (f32 weights,
  full matrix precision). If, for example, `swiglu`/`geglu` or `attention` still fail with an error just above the bound, the TPU
  approximations of `exp`/`tanh` are the cause and the bound should be relaxed, not the code.
- **q8 KV cache and DeltaNet** are not implemented on the TPU (`--kv-cache-type f16` is the default); the q8 cases of the
  self test (`kv_cache_write q8`, `attention q8`) report FAIL, the `delta_net` cases are skipped (the decoder then computes
  DeltaNet layers on the host).
- **One process per TPU**: the script steps run one after another; a hanging `fllm-rs` blocks the TPU for all following ones
  (`pkill fllm-rs`).

## What to send back

All files are next to the script (`scripts/colab/`):

- `tpu_check_report.json` (summary of all steps, environment, self-test failure lines, model comparison)
- `tpu_selftest.json` and `tpu_check_selftest.log` (every line of the self test, FAIL lines with the error text)
- `tpu_check_devices.log`, `tpu_check_server_tpu.log`, `tpu_check_server_cpu.log`, `tpu_check_build.log`

If the TPU compiler reports an error, the error message contains the beginning of the StableHLO program. With
`FLLM_TPU_DUMP_DIR=/path` the backend writes every translated program there as a `.mlir` file; the file name names the operation.

## Additional hardware tests (optional, no model)

The source contains two tests marked `#[ignore]` that need a TPU:

```bash
cd fllm && FLLM_TPU_EXACT=1 cargo test --release -p fllm-engine --features tpu-pjrt -- --ignored tpu_hardware
```

- `tpu_hardware_selftest`: the same as `selftest-gpu --device tpu`.
- `tpu_hardware_block_matches_reference`: a whole decoder block (`model.rs`, one program per layer) against the f32 reference.

## Backend settings

| Variable | Effect |
|---|---|
| `FLLM_TPU_LIBRARY` | path of the PJRT library (`libtpu.so`); otherwise searched in the Python package `libtpu` |
| `FLLM_TPU_EXACT=1` | f32 weights + full matrix precision (self test, parity runs) |
| `FLLM_TPU_WEIGHTS=bf16\|f32\|int8` | weight format on the device (default bf16; int8 with group scales) |
| `FLLM_TPU_NUMERICS=exact\|fast` | arithmetic precision, separate from the weights |
| `FLLM_TPU_DEBUG=1` | log every program translation |
| `FLLM_TPU_DUMP_DIR=DIR` | store every translated program as `.mlir` |

## Checking programs without a TPU (`stablehlo_verify.py`)

The unit tests of the backend build every program on a backend without a device (dry run) and check its structure themselves
(brackets, SSA names, operand types, shape rules). With `FLLM_TPU_DUMP_DIR` the programs are also stored as `.mlir` and
can be parsed and verified with the real StableHLO parser from `jaxlib` (no execution, only syntax and types):

```bash
pip install jax
FLLM_TPU_DUMP_DIR=/tmp/tpu-programs cargo test --release -p fllm-engine --features tpu-pjrt backend::tpu
python scripts/colab/stablehlo_verify.py /tmp/tpu-programs
python scripts/colab/stablehlo_verify.py /tmp --print /tmp/tpu-programs/exact_f32/0001_moduleattention.mlir   # readable form
```
