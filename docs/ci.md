# CI and releases (GitHub Actions)

All automation lives in `.github/workflows/`. Everything that needs no special hardware runs on GitHub-hosted runners;
the GPU and benchmark workflows are manual and need a self-hosted runner (see the last sections).

| Workflow | Trigger | What it does |
| --- | --- | --- |
| `ci` | push to `main`, pull requests, manual | `cargo clippy --locked --workspace --all-targets -- -D warnings` and `cargo test --release --locked -p fllm-engine -p fllm-checkpoint -p fllm-gguf-probe` on `ubuntu-24.04`, CPU build, no CUDA |
| `platforms` | push to `main`, pull requests, manual | build, unit tests, clippy, `devices --json`, `selftest-cpu` and `selftest-gpu --device cpu` on **Windows 2022 (MSVC)**, **macOS 14 (Apple silicon)**, **Linux aarch64** and **Linux x86_64**. The Linux and Windows runners have no GPU (compile and CPU level, L0 / L1-CPU of [test-levels.md](test-levels.md)); the Apple-silicon runner also runs the unit tests of the native Metal backend (`--features rust-gpu,metal`) and, as separate steps, the whole-model tests and `metal_quant_kernels` (every quantized-linear kernel against the CPU dequantization, one result line per format, kernel and batch size; this is the first hardware run of the MLX, block-wise FP8, bitsandbytes and EXL2 kernels), whose GPU kernel checks give the Metal backend its L1 statement. Manual only: `windows-cuda` (nvcc with the MSVC host compiler, compile and link) and `windows-package` (`scripts/build-windows.ps1` end to end) |
| `release` | tag `v*`, manual with a tag | builds the release binaries on hosted runners and attaches them to the GitHub release of the tag |
| `docker-image` | tag `v*`, manual | CPU image for `linux/amd64` and `linux/arm64` to `ghcr.io/<owner>/fllm` |
| `docker-image-cuda` | tag `v*`, manual | CUDA image (`linux/amd64` and `linux/arm64`) to `ghcr.io/<owner>/fllm:cuda` |
| `docs` | push to `main`, pull requests, manual | `python3 scripts/check_docs.py` (links, anchors, paths, language, wording, YAML syntax) |
| `msrv` | push to `main`, pull requests, manual | checks the workspace with the minimum supported Rust version (`Cargo.toml` `rust-version`) |
| `supply-chain` | push to `main`, pull requests, weekly, manual | `cargo deny`, `cargo audit`, CycloneDX SBOM (`cargo-cyclonedx`), Cosign signing, provenance attestation, and Trivy vulnerability scan |
| `coverage` | push to `main`, weekly, manual | `cargo llvm-cov` workspace coverage report and a weekly Rust nightly compiler check |
| `codeql` | push to `main`, pull requests, weekly, manual | CodeQL static analysis for Rust and GitHub Actions |
| `kernels` | manual | compile-only builds of the CUDA and ROCm/HIP kernels in the vendor containers |
| `conformance` | manual, **self-hosted runner** | `fllm-rs devices --json` and `fllm-rs selftest-gpu --json` on a runner with a real GPU |
| `streaming-benchmark` | manual, **self-hosted runner** | starts the CPU server with a GGUF model on the runner and runs `fllm-rs bench` against it |

There are no scheduled (cron) workflows: a scheduled run on a repository without the self-hosted runners would only fail.

`cargo fmt --check` is not enforced yet because the tree is not rustfmt-clean. After a `cargo fmt --all` commit, add a `fmt`
job to `ci.yml` (and `make lint` already runs it locally).

## Cutting a release

```sh
git tag v0.1.0
git push --tags
```

The tag starts three workflows in parallel: `release` (binaries), `docker-image` (CPU image) and `docker-image-cuda` (CUDA image).

1. `release` / `prepare` creates the GitHub release for the tag as a **pre-release** with automatically generated notes
   (the tag must already exist: `--verify-tag`).
2. One job per asset builds the binary and uploads it together with its `.sha256` file.
3. `finalize` checks that all required assets are on the release and sets the final state: tags `v0.*` and tags with a
   `-` suffix (`v1.0.0-rc1`) stay **pre-releases**, any other tag becomes a normal release and the "latest" release.
   If a required build fails, the release stays a pre-release without the missing asset; fix the cause and re-run the failed jobs
   (or run the workflow again from Actions -> release -> Run workflow with the tag; uploads replace existing assets).

`install.sh` and `install.ps1` find the newest release also while only pre-releases exist (they read the release list when
`/releases/latest` has no target), so the `v0.x` pre-release marking does not break `./install.sh`.

### Asset names (contract with `install.sh` / `install.ps1`)

| Asset | Runner | Notes |
| --- | --- | --- |
| `fllm-linux-x86_64-musl.tar.gz` | `ubuntu-24.04` | static musl build (`musl-tools`), feature `rust-gpu` |
| `fllm-linux-aarch64-musl.tar.gz` | `ubuntu-24.04-arm` | native ARM64 runner, same recipe |
| `fllm-macos-aarch64.tar.gz` | `macos-14` | Apple silicon, feature `rust-gpu` (wgpu uses Metal) |
| `fllm-macos-x86_64.tar.gz` | `macos-15-intel` | Intel Mac |
| `fllm-windows-x86_64-msvc.zip` | `windows-2022` | `scripts/build-windows.ps1`: MSVC, static C runtime, with the native Vulkan backend (Vulkan SDK installed in the job) |
| `fllm-windows-x86_64-msvc-cuda.zip` | `windows-2022` | plus the CUDA kernels (toolkit 12.8 installed in the job); **optional**: if the job fails the release is still published and `install.ps1` uses the plain Windows asset |

Every asset has a `<asset>.sha256` file with one line, `<hash>  <asset>`; the install scripts verify it and refuse a mismatch.
Not built: `fllm-windows-x86_64.zip` (MinGW cross build of old releases), `fllm-windows-x86_64-msvc-hip.zip` (needs the AMD HIP SDK),
`fllm-windows-aarch64-msvc.zip` (no hosted Windows ARM runner for this build). The install scripts try these names and fall back
to the next candidate, then to building from source.

The Linux and macOS archives contain `fllm-rs` and `LICENSE`, the Windows archives `fllm-rs.exe` and `LICENSE`. The release jobs
run `fllm-rs --help` as a smoke test (Windows: `fllm-rs devices` inside the script), use no build cache and run the unit tests in
`ci` / `platforms`, not again in `release` (a tag should be created from a commit whose `ci` run is green).

The feature set is `RUST_FEATURES` at the top of `release.yml` (`rust-gpu`, the same as `make build-rust`, `install.sh` and the
Dockerfile). The native `vulkan` / `metal` backends are not in the macOS and Linux assets; their verification level is in
[backends.md](backends.md).

## Docker images

`docker-image` builds `docker/Dockerfile` natively on an amd64 and an arm64 runner (no emulation), pushes each by digest and joins them
into one multi-arch tag. Tags for a release tag `v0.1.0`:

| Image | Tags |
| --- | --- |
| CPU, `linux/amd64` + `linux/arm64` | `ghcr.io/<owner>/fllm:latest` (not for tags with a `-` suffix), `:v0.1.0`, `:0.1.0`, `:sha-<short>` |
| CUDA, `linux/amd64` | `ghcr.io/<owner>/fllm:cuda` (not for tags with a `-` suffix), `:cuda-v0.1.0`, `:cuda-sha-<short>` |

The image name is the repository owner in lower case (the registry requires it). A manual run on a branch pushes only the `sha-` tags.
Login uses the job's `GITHUB_TOKEN` with `packages: write`; no personal token is stored.

The CUDA build compiles for the architectures `61;70;75;80;86;89;90;120` (the default of `docker/Dockerfile.cuda`; the manual run has
an input `cuda_archs`, e.g. `120` for a quick test). nvcc needs no GPU. On a 4-core hosted runner the full build takes several hours,
so the job has a 355-minute timeout, removes unused preinstalled toolchains to free disk space first and keeps a layer cache (`type=gha`, scope `cuda`) that makes
later builds with an unchanged `Cargo.toml` / `Cargo.lock` much shorter. GitHub's cache quota is 10 GB per repository; old entries are
evicted.

## Platform builds

`platforms.yml` is the "does it build and run on every OS" check. Its rows:

| Runner | Features |
| --- | --- |
| `windows-2022` | `rust-gpu,vulkan` (Vulkan SDK installed for `glslc`) |
| `macos-14` | `rust-gpu,metal` |
| `ubuntu-24.04-arm` | `rust-gpu,vulkan,arm-simd` (`glslc` from apt); the CPU self tests (`selftest-cpu`) run on the ARM hardware, so the NEON kernels are exercised, the opt-in SVE kernel is not |
| `ubuntu-24.04` | `rust-gpu,vulkan,avx512` (`glslc` from apt) |

Only the macOS row has a usable GPU (Apple silicon); `devices --json` and the self-test JSON are kept as artifacts of the run.

`kernels.yml` (manual) compiles the CUDA kernels (Ampere, Hopper, Blackwell, plus clippy with the `cuda` feature) in the
`nvidia/cuda` development container and the HIP kernels in the `rocm/dev-ubuntu` container. The containers are large; if a hosted
runner runs out of disk space, build with a smaller architecture list or use the Docker workflow, which frees disk space first.

## Running the GPU tests on your own machine (self-hosted runner)

GitHub-hosted runners have no GPU. `conformance.yml` and `bench.yml` therefore run on a runner you register yourself, selected by
the input `runner_labels` (a JSON array, default `["self-hosted", "linux", "gpu"]`).

1. On the machine with the GPU install the toolchain of the backend (NVIDIA: driver and CUDA toolkit; AMD: ROCm / HIP SDK with `hipcc`;
   Apple: Xcode command line tools; Vulkan: a driver and `glslc`; Intel: oneAPI DPC++), plus Rust (rustup), Git, a C compiler and
   CMake. Windows runners also need Git for Windows (the steps use `bash`).
2. Register the runner: repository **Settings -> Actions -> Runners -> New self-hosted runner**, follow the shown commands, and
   add custom labels that describe the machine (for example `gpu-nvidia`). Run it as an unprivileged user, ideally as a service.
3. Start the workflow: **Actions -> conformance -> Run workflow**. Set `runner_labels` to the labels of your runner, for example
   `["self-hosted", "linux", "gpu-nvidia"]`, and pick the `backend` (`nvidia-cuda`, `amd-rocm`, `apple-metal`, `vulkan`,
   `intel-sycl`). `only` and `quants` narrow the self test (see `fllm-rs selftest-gpu --help`). The job builds with the matching
   features, runs `fllm-rs devices --json` and `fllm-rs selftest-gpu --device <backend> --json` and uploads `devices.json`,
   `selftest.json` and `selftest.txt`; the step fails when a check fails.
4. `streaming-benchmark` needs a GGUF model on the runner: give its absolute path as the input `model` or set the repository variable
   `BENCH_MODEL`. It starts the CPU server on `127.0.0.1:18107`, runs `fllm-rs bench` and, if `bench/baseline.json` exists in the
   checkout, compares with it (10 % regression budget).

A green `conformance` job means level L1 of [gpu-backends.md](gpu-backends.md) for that backend and device: every operation and
format the backend lists matches the CPU reference within the tolerances (`SKIPPED (unsupported)` lines mark what it does not
implement). It does not mean that a model produces the same text (L2) or that it is fast (L3). Details:
[device-selection.md](device-selection.md); Windows: [windows.md](windows.md).

**A self-hosted runner executes the code of the workflow run on your machine.** Use it only for a private or trusted setup:
never attach it to a public repository without restricting who can run workflows, because a pull request from a fork could run
arbitrary code on it. Both workflows are `workflow_dispatch` only (they need write access to the repository to start), and the
repository setting **Actions -> General -> Fork pull request workflows** should require approval for all outside collaborators.

## Security notes

- Every third-party action is pinned to a full commit SHA with the version in a comment. Dependencies and actions are updated by hand
  (automated update pull requests are switched off).
- The default `GITHUB_TOKEN` permission of every workflow is `contents: read`. Jobs that publish get more, and only they:
  `contents: write` in `release` (create the release, upload assets), `packages: write` in the Docker workflows.
- Checkouts use `persist-credentials: false`. Values that come from outside (the tag name, workflow inputs) are passed to scripts
  through environment variables, never interpolated into the shell code, and the tag is checked against `^v[0-9][0-9A-Za-z._-]*$`.
- The release builds use no cache, so a published binary does not depend on a cache entry that a pull request could have written.
  The build caches of the other workflows are saved only on `main`.
- No secrets are needed: login and uploads use `GITHUB_TOKEN`. Do not echo tokens or model URLs with credentials in workflow logs.
- Static checks used on these files: `actionlint` and `zizmor` (both report nothing at the default level; `zizmor --persona pedantic`
  only notes that `dtolnay/rust-toolchain` could be replaced by `rustup` commands).

## One-time repository settings

- **Settings -> Actions -> General -> Workflow permissions**: leave "Read repository contents and packages permissions" (the
  workflows request what they need), and set "Fork pull request workflows" to require approval.
- After the first image push: **Packages -> fllm -> Package settings**: link it to the repository and change the visibility to
  **Public** (new ghcr.io packages start private), otherwise `docker pull` needs a login.
- **Settings -> Branches**: protect `main` and require the `ci` jobs (`clippy (CPU build)`, `unit tests (CPU build)`) and, if wanted,
  the four `platforms` jobs.
- **Settings -> Rules / Tags**: a tag ruleset for `v*` so that only maintainers can create release tags (the tag starts the
  publishing workflows).

## Verification status

The workflow files pass `actionlint` and `yamllint`. The workflows `ci`, `platforms`, `release`, `docker-image`, `docker-image-cuda`,
`docs`, `msrv`, `supply-chain`, `coverage`, `codeql`, `kernels`, `conformance` and `streaming-benchmark` exist. `ci`, `platforms`, `release` and the two Docker workflows run on GitHub-hosted runners (the v0.1.0 release was produced by them);
`kernels` is a manual compile check; `conformance` and `streaming-benchmark` need a self-hosted runner with a GPU (or a model) and are run on demand. The optional
`windows-cuda` and `windows-package` jobs of `platforms` are manual.
