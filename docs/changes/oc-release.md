# Linux x86_64 CUDA release package

## 1. Why the existing Linux assets cannot use a GPU

The Linux release assets are static musl binaries (`static-pie linked`, no interpreter segment,
`ldd` says "statically linked"). A static-PIE binary has no dynamic loader, so `dlopen` cannot
work at all - a two-line C probe compiled the same way reports "Dynamic loading not supported"
for `libcuda.so.1`, `libvulkan.so.1` and even `libm.so.6`, while the same source linked
dynamically (glibc) loads `libm` and reports missing shared objects normally. On a machine with
an NVIDIA driver the musl asset therefore never reaches the CUDA driver: the driver library
cannot be opened at run time. The same applies to the Vulkan loader (`libvulkan.so.1`) of the
wgpu backend. GPU support needs a glibc (dynamic) binary; that is the reason for this package.

Two related findings:

* The CUDA runtime is linked statically (`cargo:rustc-link-lib=static=cudart_static` in
  `rust/engine/build.rs`), so a CUDA binary needs only `libcuda.so.1` from the installed driver -
  no CUDA toolkit on the user's machine. HIP links `libamdhip64` dynamically as well (the
  release has no HIP asset); Vulkan needs the loader `libvulkan.so.1` (part of any driver
  install).
* The CLI has no `--version` option (it exits with "unknown option"); the smoke tests use
  `--help` and `devices`.

## 2. New release asset: `fllm-linux-x86_64-cuda.tar.gz`

`.github/workflows/release.yml` builds it in an optional job (`linux-cuda`,
`continue-on-error: true` like the Windows CUDA job, so a failure never blocks a release):

* The build runs inside `nvidia/cuda:12.8.1-devel-ubuntu22.04` via
  `scripts/release/build-linux-cuda.sh` (same base image as `kernels.yml`). The Ubuntu 22.04
  base keeps the glibc floor at 2.35 - a binary built on 24.04 would not start on 22.04.
* CUDA architectures: `75;80;86;89;90;100;120` (Turing through Blackwell). `build.rs` appends
  the PTX of the newest architecture automatically, so future drivers can JIT for anything newer.
* The job itself runs on the runner (not inside a container) so that the smoke test can start a
  clean `ubuntu:22.04` container with docker: `scripts/release/verify-linux-cuda-package.sh`
  checks that the dynamic section contains no `libcudart`, that `--help` and `devices` work
  (cpu and cuda among the compiled backends) and that `selftest-gpu` reports the missing GPU
  with a clear message instead of crashing. With the real package this prints
  `Compiled backends: cpu, vulkan, metal, dx12, wgpu, cuda`, `Error: "no GPU device found; ..."`
  for `selftest-gpu` and `Error: "Device not found: no CUDA driver: ..."` for
  `selftest-gpu --device cuda` (both exit 1, no panic).
* The package contains `fllm-rs` and `LICENSE` and ships with `<asset>.sha256`, exactly like
  the other assets. The binary is 863.7 MB (824 MiB) and the compressed asset 244.1 MB; the
  CPU-only musl asset is 12.5 MB. Stripping saves only about 9 MB (the size is the embedded CUDA
  code for seven architectures), so the binary ships unstripped like the other release assets.

`scripts/release/build-linux-cuda.sh` also runs by hand on any machine with docker (it mounts
the checkout at `/src`, uses the shared cargo registry volume `fllm-cargo-cache` and writes
`dist-cuda/fllm-rs`); `CUDA_IMAGE`, `CUDA_ARCHS`, `RUST_FEATURES`, `TARGET_DIR`, `OUT`, `JOBS`
and `NAME` are environment overrides.

## 3. Installers pick the CUDA asset automatically

`install.sh` (Linux/macOS, POSIX sh) and `install.ps1` (Windows) gained the environment
variable `FLLM_VARIANT`:

| Value | install.sh on Linux x86_64 | install.ps1 |
| --- | --- | --- |
| `auto` (default) | cuda asset when `nvidia-smi` is on PATH, else musl; a missing cuda asset falls back to musl with a warning | `-Variant auto`: `nvcuda.dll` -> cuda, `amdhip64*.dll` -> hip, else default (unchanged) |
| `cpu` | always the musl asset | maps to `-Variant default` |
| `cuda` | always the cuda asset; missing asset is an error (no fallback) | maps to `-Variant cuda` |
| `hip` | rejected | maps to `-Variant hip` |

An explicit `-Variant` argument of `install.ps1` still wins over `FLLM_VARIANT`; on macOS,
aarch64 Linux and ARM64 Windows only `auto`/`cpu` are accepted (no CUDA asset exists there).
The checksum contract is unchanged: the `.sha256` file of the chosen asset is verified before
anything is installed.

## 4. Windows CUDA jobs: documented state

`windows-cuda` and `windows-package` are jobs in `platforms.yml` (manual only), not the
release workflow; the packaged Windows CUDA build is the `windows-cuda` job in `release.yml`.
The manual jobs were made consistent with the release path:

* `windows-package` now installs the Vulkan SDK (like `release.yml`) and no longer passes
  `-NoVulkan`, so it really produces what the release produces: `dist/<asset>.zip` plus
  `.sha256`, with the unit tests (the release job itself skips them).
* `windows-cuda` keeps `build.log` as an artifact (it exists on failure and success); it
  produces no package and is the compile-only check for the CUDA kernels (sm_80 + sm_120).
* Both jobs still need a Windows runner; only the review and the PowerShell parse check
  (`pwsh` in a container) happened here, no execution.

## 5. Test level and limits

Test level L0: the Linux CUDA package was built and smoke-tested without a GPU (compile,
start, clear message without a device). No GPU ran the kernels; nothing above L0 is claimed.
Verifying the kernels needs a machine with an NVIDIA driver (for example a Colab GPU).

Limits:

* Only `x86_64` Linux has a CUDA asset; aarch64 Linux and macOS stay CPU-only (musl assets
  cannot load drivers; macOS has no CUDA since 10.13).
* `continue-on-error` keeps the asset optional: an older release simply has no
  `fllm-linux-x86_64-cuda.tar.gz` and `install.sh` falls back to the musl asset with a warning.
* The asset is much bigger than the musl asset: it carries the CUDA kernels for seven
  architectures plus PTX (244.1 MB versus 12.5 MB compressed); stripping saves only about one
  percent, so the binary ships unstripped like the other release assets.

## Environment variables touched here

| Variable | Where | Meaning |
| --- | --- | --- |
| `FLLM_VARIANT` | `install.sh`, `install.ps1` | asset variant: `auto` (default) / `cpu` / `cuda` (install.sh), `auto`/`cpu`/`cuda`/`hip` (install.ps1) |
| `CUDA_IMAGE`, `CUDA_ARCHS`, `RUST_FEATURES`, `TARGET_DIR`, `OUT`, `JOBS`, `NAME` | `scripts/release/build-linux-cuda.sh` | overrides for the local/CI build (defaults: `nvidia/cuda:12.8.1-devel-ubuntu22.04`, `75;80;86;89;90;100;120`, `rust-gpu,cuda`, `target-linux-cuda`, `dist-cuda`, `3`, `fllm-linux-cuda-build`) |
| `SMOKE_IMAGE` | `scripts/release/verify-linux-cuda-package.sh` | image for the smoke test (default `ubuntu:22.04`) |
