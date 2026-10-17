# Test levels

fLLM documents how far every backend, kernel family and model has been verified with four levels, L0 to L3. The levels exist so that a
statement such as "runs on AMD GPUs" always says what was actually done and on which hardware. They are used in the README Status
table, in the backend matrices ([backends.md](backends.md), [gpu-backends.md](gpu-backends.md)), in the verification procedures
([backends-verify.md](backends-verify.md)), in the research notes ([research/README.md](research/README.md)) and in pull requests and
hardware reports ([CONTRIBUTING.md](../CONTRIBUTING.md)).

| Level | Short name | What it proves | What it does not prove |
| --- | --- | --- | --- |
| **L0** | compiles | The code builds (and passes `clippy`) for the target; kernels that are text (GLSL, MSL, SYCL, HIP) parse or compile with the vendor tool. Unit tests that need no hardware and CPU emulations of a kernel also stay at L0. | That any instruction ran on the target device. A kernel can compile and still compute nonsense. |
| **L1** | self test passes on the device | `fllm-rs selftest-gpu --device <name>` (or `selftest-cpu` for CPU tiers, or the hardware tests of a backend) ran on real hardware, and every check matched the f64/CPU reference within its tolerance. | That a whole model works: the self test uses random data and single operations, not a loaded checkpoint, and not every operation is covered for every format. |
| **L2** | same text as the CPU | Greedy decoding (temperature 0) of a real model on the device produces the same text as the CPU reference, using the procedure of [backends-verify.md](backends-verify.md). | That it is fast, or that sampling-based output is unchanged. Small rounding differences can flip a near-tie token; the procedure says how to judge a difference. |
| **L3** | speed measured | Decode and prefill speed were measured on the same hardware, with a stated method, and (where a reference exists) compared with a reference engine on the same card. See [benchmarking.md](benchmarking.md). | That the numbers hold on other hardware, other drivers or other models. |

## How to read a level

* Levels are cumulative per target and per feature: L2 for a backend means L1 was met first. A table cell that says "L0" means "not run
  on the hardware", regardless of how complete the code looks.
* A level is a statement about a specific combination of backend, device generation, weight format and operation. "CUDA L2" is
  therefore always qualified, for example "Turing and Blackwell, Q4_K_M and NVFP4 models".
* A claim above L0 needs a record: the command, the version (commit or release), the device and driver, and the result. The
  hardware report issue template asks for exactly this.
* `SKIPPED (unsupported)` lines in a self test are not failures and not passes: they mark operations the backend does not implement.
  They lower the coverage that an L1 statement can make.
* Some tables add a "B" grade (CPU kernels): the kernels ran on real hardware by the project but without the full L1/L2 procedure
  of a GPU backend. The tables that use it define it.

## Producing evidence

| Level | Command (examples) |
| --- | --- |
| L0 | `cargo clippy --release --workspace --all-targets --locked -- -D warnings` with the backend feature, for example `--features vulkan` |
| L1 | `fllm-rs devices --json`, then `fllm-rs selftest-gpu --device <name> --json selftest.json`; for CPU tiers `fllm-rs selftest-cpu --tiers all` |
| L2 | the `scripts/verify-<backend>.sh` (or `.ps1`) script of the backend, which runs the self test, starts a server and compares greedy text with the CPU; `scripts/README.md` lists them |
| L3 | `fllm-rs bench` against a running server, or `scripts/colab/ab_matrix.py` for the CUDA switches; see [benchmarking.md](benchmarking.md) |

The CI workflows give L0 for every platform and L1 for the CPU tiers of the hosted runners; the `conformance` workflow gives L1 for a
GPU on a self-hosted runner ([ci.md](ci.md)).
