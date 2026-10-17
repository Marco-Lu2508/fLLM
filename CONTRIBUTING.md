# Contributing to fLLM

Thank you for helping. fLLM is a pure-Rust LLM inference engine; the maintainer is [Marco-Lu2508](https://github.com/Marco-Lu2508).
Bug reports, hardware reports, documentation fixes, new architectures, kernels and backends are all welcome. By taking part you agree
to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

* **Report a bug** with the bug report template. Include the exact command, the model, the output of `fllm-rs devices` and the version.
* **Report hardware results.** The project can only test what its maintainers own. A result on a GPU, NPU, TPU, CPU generation or OS
  that is marked untested in the README Status table is valuable. Use the hardware report template (see
  [Reporting hardware results](#reporting-hardware-results)).
* **Fix or extend the documentation.** The documentation index is [docs/index.md](docs/index.md).
* **Add a model architecture:** [docs/adding-a-model.md](docs/adding-a-model.md).
* **Add or extend a backend (GPU/NPU/accelerator):** [docs/adding-a-backend.md](docs/adding-a-backend.md).
* **Security problems** go through a private advisory, not a public issue: see [SECURITY.md](SECURITY.md).

For anything larger than a small fix, open an issue first so that the design can be agreed before you invest time.

## Building

The toolchain is stable Rust, a C compiler and Git; CUDA builds also need the CUDA toolkit (or use the CUDA Docker image).

```sh
cargo build --release --locked -p fllm-engine --features rust-gpu --bin fllm-rs   # same as: make build-rust
target/release/fllm-rs --help
```

Optional backends are Cargo features of `fllm-engine` (`cuda`, `vulkan`, `metal`, `hip`, `sycl`, `npu-*`, `tpu-pjrt`, `arm-simd`,
`avx512`); `docs/gpu-backends.md` and `docs/backends.md` say which toolchain each needs. Only build the features you can use.

## Testing

Run these before you open a pull request and say in the description which ones you ran:

```sh
cargo test --release --locked -p fllm-engine -p fllm-checkpoint -p fllm-gguf-probe          # make test runs the workspace tests
cargo clippy --release --workspace --all-targets --locked -- -D warnings                      # make lint also checks formatting
cargo clippy --release --workspace --all-targets --locked --features rust-gpu -- -D warnings  # and with every feature you touched
```

CI builds Linux (x86-64 and aarch64), macOS (Apple silicon) and Windows (MSVC). Code that is specific to one platform must be guarded
with `cfg`; remember that `c_char` is `u8` on aarch64 Linux and that GNU-only inline assembly does not compile with MSVC.

### Self tests

Kernels are checked by built-in self tests that compare a backend with the CPU and an f64 reference on random data:

```sh
fllm-rs selftest-cpu --tiers all                         # every CPU kernel tier this machine supports
fllm-rs devices                                          # what this build can use
fllm-rs selftest-gpu --device cuda --json selftest.json  # a GPU backend; --list shows the groups, --only narrows them
```

A new kernel or a new weight format needs `selftest-gpu` cases (see `rust/engine/src/backend/selftest*.rs`) whose failure messages
name the check, the largest error, the tolerance and the worst element. Parsers and importers need unit tests with tiny synthetic
fixtures. No test may download anything from the network.

### Model-level checks

The proof scripts in `scripts/` compare fLLM with `transformers` on tiny random checkpoints and check greedy generation, the
tokenizer, encoders and the API clients. `scripts/README.md` explains them; the common ones are `make test-arch`, `make test-model` and
`make test-encoder`. A new architecture needs a case in `scripts/hf_golden.py` and a passing `scripts/verify_architectures.py` run.

## Branches and commits

* Work on a topic branch of your fork and open the pull request against `main`. Keep a branch focused on one change.
* Commit messages are English and in the imperative: a short subject (about 70 characters), a blank line, then the reason for the
  change if it is not obvious. Prefix with the area when it helps (`cuda:`, `cpu:`, `server:`, `docs:`, `ci:`).
* One logical change per commit; every commit should build and pass the tests.
* Do not add `Co-Authored-By` trailers or mention tools used to write the code; commits are attributed to the person who submits them.
* Do not commit models, benchmark dumps over 1 MB, build output, credentials, private host names or e-mail addresses.
* A file's line-ending style must not change: some files use CRLF (check with `git diff --stat` that only your lines changed).

## Code style

* Follow the surrounding code: naming, error handling, comment density. `cargo fmt` is not enforced yet because the tree is not
  rustfmt-clean; do not reformat code you do not change.
* Every new behaviour that can change numerics or performance gets an environment switch with a documented default, and the switch
  goes into [docs/configuration.md](docs/configuration.md).
* Unsupported cases must return a clear error or fall back to an existing path. Never stub silently.
* Prefer safe Rust. `unsafe` blocks (FFI, SIMD, mmap) carry a `SAFETY:` comment that states the invariant.
* Documentation is English, plain and factual: no emojis, no marketing wording, no unmeasured performance claims.
* State test levels exactly as defined in [docs/test-levels.md](docs/test-levels.md) and never claim a result you did not obtain.

## Licence of contributions

fLLM is licensed under the [Apache License 2.0](LICENSE). Unless you state otherwise, every contribution you submit is licensed under
the same terms (section 5 of the licence). Only submit code you wrote or code whose licence allows redistribution under Apache-2.0.

Code adapted from another project (for example vLLM, SGLang, FlashInfer, CUTLASS or Marlin) keeps its licence header and needs an entry
in [THIRD_PARTY.md](THIRD_PARTY.md) (source, licence, what was adapted). **Do not copy code from llama.cpp or ggml**; reading its
file-format documentation to implement a layout is fine, copying source is not. GPL and AGPL material cannot be merged.

## Reporting hardware results

Open an issue with the "Hardware report" template and fill in:

1. Operating system, CPU, GPU/NPU and driver versions.
2. The full output of `fllm-rs devices`.
3. The summary of `fllm-rs selftest-gpu --device <name>` (or `selftest-cpu`), and the JSON file if it is small.
4. The model and quantization you ran and the result: whether the text is sensible and, if you can, identical to the CPU
   (`scripts/verify-<backend>.sh` does the comparison).
5. Speed numbers only with the method: the command, prompt length, number of users, and whether the machine was otherwise idle
   ([docs/benchmarking.md](docs/benchmarking.md)).

Say which level of [docs/test-levels.md](docs/test-levels.md) your result reaches. Failures are as useful as successes.

## Adding to the documentation

Concept pages (`docs/architecture.md`, `docs/adding-a-backend.md`, `docs/benchmarking.md`, `docs/roadmap.md`) avoid copying flag lists
and numbers that change; the CLI help (`fllm-rs --help`) and [docs/configuration.md](docs/configuration.md) are authoritative. Run
`python3 scripts/check_docs.py` before submitting documentation changes: it checks links, anchors, referenced paths and stray
non-English text.
