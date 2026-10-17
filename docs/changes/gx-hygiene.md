# Release, Supply-Chain, CUDA Build, Packaging and Documentation Hygiene (`gx-hygiene`)

Base: combined hygiene branch merging `w-hygiene-a` (`02edef4`) into `w-hygiene-b` (`a157d5f`).

## 1. License Headers in `rust/engine/cuda`

Added explicit `SPDX-License-Identifier: Apache-2.0` and copyright/provenance notices referencing vLLM, Neural Magic, IST-DASLab (`Marlin`), ggml, `THIRD_PARTY.md`, `NOTICE`, and `licenses/` to:

- `rust/engine/cuda/marlin_vllm/dequant.h`
- `rust/engine/cuda/marlin_vllm/marlin_mma.h`
- `rust/engine/cuda/marlin_vllm/marlin_dtypes.cuh`
- `rust/engine/cuda/marlin_vllm/marlin.cuh`
- `rust/engine/cuda/marlin_vllm/kernel.h`
- `rust/engine/cuda/marlin_vllm/core/scalar_type.hpp`
- `rust/engine/cuda/marlin_vllm/marlin_template.h`
- `rust/engine/cuda/marlin_vllm/marlin_vllm.cu`
- `rust/engine/cuda/marlin_vllm/marlin_vllm_awq_repack.cu`
- `rust/engine/cuda/marlin_vllm/marlin_vllm_gptq_repack.cu`
- `rust/engine/cuda/marlin_q4k.cu`
- `rust/engine/cuda/iq_tables.cuh` (`SPDX-License-Identifier: Apache-2.0 AND MIT`)

## 2. Release Archives, Glibc `dlopen`, Linux/Windows CUDA & ARM64 Docker

- **Glibc dynamic Linux release archives**: Static `*-unknown-linux-musl` binaries cannot `dlopen` glibc-linked host GPU drivers (`libcuda.so.1`, `libvulkan.so.1`). `.github/workflows/release.yml` now builds `fllm-linux-x86_64-gnu.tar.gz` (`x86_64-unknown-linux-gnu`) and `fllm-linux-aarch64-gnu.tar.gz` (`aarch64-unknown-linux-gnu`) with `rust-gpu,vulkan` alongside the static `musl` CPU archives, and verifies dynamic `dlopen` support with `scripts/verify-dlopen.sh`. `install.sh` accepts `--variant cpu|gnu|musl|cuda`.
- **Linux x86_64 CUDA archive**: Added the `linux-cuda` job in `.github/workflows/release.yml` to build `fllm-linux-x86_64-cuda.tar.gz` (matching `install.sh --variant cuda`).
- **Legal files & release-note checksums**: Both `.github/workflows/release.yml` and `scripts/build-windows.ps1` now bundle `LICENSE`, `NOTICE`, `THIRD_PARTY.md`, and `licenses/` into every `.tar.gz` and `.zip` release archive, and the `finalize` job appends a `<sha256>  <asset>` table to the GitHub Release body.
- **Windows MSVC + nvcc CUDA build support**: `rust/engine/build.rs` now passes `/MD,/EHsc,/bigobj,/Zc:__cplusplus,/std:c++17` (or `/MT,...` for static CRT) via `-Xcompiler` to `nvcc` and prints full command/stdout/stderr diagnostics when `nvcc` fails. `.github/workflows/platforms.yml` (`windows-cuda`) clones `third_party/cutlass` and `third_party/flashinfer` so CUTLASS and Marlin kernels are exercised on Windows.
- **ARM64 CUDA Docker support**: `docker/Dockerfile.cuda`, `rust/engine/build.rs`, and `.github/workflows/docker-cuda.yml` (`docker-image-cuda`) now support `linux/arm64` builds targeting Jetson Orin (`sm_87`), Grace Hopper (`sm_90`), and DGX Spark (`sm_121` on CUDA >= 12.8).

## 3. Supply Chain Security, Signing, Coverage, CodeQL & Dependabot

- **Signing & provenance**: `.github/workflows/release.yml` and `.github/workflows/supply-chain.yml` sign release archives and the CycloneDX SBOM with keyless OIDC `cosign sign-blob` (`sigstore/cosign-installer`) and generate SLSA build provenance attestations (`actions/attest-build-provenance`). Optional Windows Authenticode (`signtool.exe`) and macOS `codesign` + `notarytool` steps run when certificate secrets are configured.
- **SBOM & vulnerability scanning**: `.github/workflows/supply-chain.yml` generates `fllm-sbom.cdx.json` via `cargo-cyclonedx` and scans the repository with `aquasecurity/trivy-action`.
- **Coverage, Nightly, CodeQL & Dependabot**: Added `.github/workflows/coverage.yml` (`cargo llvm-cov` + weekly nightly compiler check), `.github/workflows/codeql.yml` (Rust and GitHub Actions), and `.github/dependabot.yml` (`open-pull-requests-limit: 5` for `cargo` and `github-actions`).

## 4. Package Manager Manifests

Added starter manifests in `packaging/`:
- `packaging/homebrew/fllm.rb`
- `packaging/winget/Marco-Lu2508.fLLM.yaml`
- `packaging/scoop/fllm.json`
- `packaging/aur/PKGBUILD`
- `packaging/nix/package.nix`

## 5. Documentation & Source Comment Audit Fixes

- Applied `doc-audit-fixes.patch` and the remaining items from `docs/changes/hygiene-b.md` to `README.md`, `THIRD_PARTY.md`, `docs/architectures.md`, `docs/ci.md`, `docs/configuration.md`, `docs/cpu-kernels.md`, `docs/cuda-qgemm4k.md`, `docs/gpu-backends.md`, `docs/inference-features.md`, `docs/server-api.md`, `docs/user/commands.md`, `rust/gguf-probe/README.md`, `rust/engine/src/bin/serve.rs`, `install.sh`, `install.ps1`, `scripts/test-install-sh.sh`, and `scripts/test-install-ps1.sh`.
- Updated `STUDY_*.md` references in `rust/engine/src/backend/npu/{hexagon.rs,ane.rs,xdna.rs,hexagon/graph.rs}` to `docs/research/npu-study-*.md`.
- `python3 scripts/check_docs.py` passes with 0 errors across `links`, `paths`, `german`, `jargon`, and `yaml`.
