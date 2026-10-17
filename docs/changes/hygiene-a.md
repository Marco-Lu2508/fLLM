# Change fragment: hygiene-a (licences, supply chain, version, installers)

Merge notes for the documentation step: `THIRD_PARTY.md` and `NOTICE` were rewritten completely in this change (the task asked for it), so there is nothing to merge
into them from this fragment except what other fragments add. README quick start text to take from here is in "README text" below.

## 1. Licences and third-party inventory

* New licence texts in `licenses/`: `ggml-MIT.txt` (Copyright (c) 2023-2026 The ggml authors), `MLX-MIT.txt` (Copyright (c) 2023-2024 Apple Inc., the wording of the three
  `.metal` headers), `ExLlamaV3-MIT.txt` (Copyright (c) 2025 Turboderp, the wording of `exl3.rs` and `docs/exl-formats.md`), `rocWMMA-MIT.txt` (Copyright (c) Advanced Micro Devices,
  Inc.), `FlashInfer-Apache-2.0.txt`, `OpenXLA-Apache-2.0.txt`, `OpenVINO-Apache-2.0.txt`. The existing vLLM and Marlin Apache texts got a copyright preamble (the Apache text itself
  names no holder). Holder lines come from the file headers and from the earlier THIRD_PARTY.md; where a header gave none (OpenVINO: "Intel Corporation") the wording is exactly that,
  no year was invented.
* `NOTICE` (Apache-2.0 convention): the modified Apache-licensed files with one line each (vLLM Marlin copies and HIP kernels, Marlin Q4_K kernel, FlashInfer glue, OpenXLA `sys.rs`,
  OpenVINO `ffi.rs`) and the list of MIT/BSD works carried.
* `THIRD_PARTY.md` rewritten: one table of upstreams, then one section per upstream with file lists, licence, nature of use (copied / adapted / layout facts / design reference /
  loaded at run time), upstream commits and the status of unchecked terms (Qualcomm QNN/Genie, AMD Ryzen AI, Intel oneAPI redistribution, NVIDIA toolkit): what is not redistributed and what
  the user installs. The stale CI paragraph ("none has run", duplicated path) is gone.

### Inventory (file -> upstream -> licence -> what was taken)

| File(s) | Upstream | Licence | Taken |
| --- | --- | --- | --- |
| `rust/engine/src/quant_iq_tables.rs`, `cuda/iq_tables.cuh` | ggml `ggml-common.h`, llama.cpp b1ff4ca2 | MIT | integer codebooks (generated copies; headers say "Portions Copyright (c) The ggml authors") |
| `rust/engine/src/quant_iq.rs`, `quant_extra.rs`, `cuda/iq_decode.cuh` | ggml `ggml-quants.c` | MIT | block layouts and dequantization, adapted into Rust / device code |
| `cuda/kv_quant.cu`, `metal/common*.metal`, `shaders/vulkan/_q_fmt_iq4_nl.glsl`, `_ops_wq.glsl`, `src/backend/vulkan/weights/mirror.rs`, `src/cpu_arm/formats.rs`, `src/kv_format.rs` | ggml | MIT | the 16-entry IQ4_NL grid (format fact) |
| `rust/gguf-probe/src/quant_types.rs` | ggml type table | MIT | type ids, block sizes (format facts) |
| `metal/attn_decode.metal`, `norm_rms.metal`, `rope_apply.metal` | MLX `sdpa_vector.h`, `rms_norm.metal`, `rope.metal` | MIT (Apple) | kernel structure; notice kept in the headers |
| `rust/checkpoint/src/exl3.rs`, `docs/exl-formats.md` | ExLlamaV3 | MIT (turboderp) | codebook constants, trellis layout |
| `cuda/marlin_vllm/{marlin_template.h,dequant.h,marlin_mma.h,marlin_dtypes.cuh,marlin.cuh,kernel.h,core/scalar_type.hpp,generate_kernels.py}`, generated `sm75_/sm80_kernel_*.cu`, `kernel_selector.h` | vLLM `csrc/libtorch_stable/quantization/marlin` @ 9367d8b9 (Marlin, Neural Magic) | Apache-2.0 | copied (scalar_type include replaced) |
| `cuda/marlin_vllm/marlin_vllm.cu`, `marlin_vllm_{awq,gptq}_repack.cu` | vLLM `marlin.cu`, `awq_marlin_repack.cu`, `gptq_marlin_repack.cu` | Apache-2.0 | adapted (PyTorch glue replaced) |
| `cuda/marlin_q4k.cu` | IST-DASLab Marlin `marlin_cuda_kernel.cu` | Apache-2.0 | adapted to Q4_K |
| `hip/hip_skinny.cu`, `hip/hip_q4z.cu` (kernel) | vLLM `csrc/rocm/skinny_gemms*.cu` @ 7fad7db3 | Apache-2.0 | adapted |
| `cuda/nvfp4_cutlass.cu`, `nvfp4_quant.cuh` | vLLM nvfp4 sm120 kernels and quantization recipe; CUTLASS ex. 79a; FlashInfer fp4 GEMM templates | Apache-2.0 / BSD-3-Clause / Apache-2.0 | configuration, collective builders, epilogue and tile list (adapted); CUTLASS and FlashInfer headers are build-time dependencies, not copied |
| `cuda/flashinfer_prefill.cu` | FlashInfer `csrc/batch_prefill.cu`, prefill headers | Apache-2.0 | host glue; the kernel is a header dependency |
| `cuda/fused.cu`, `src/mem/*` | vLLM, SGLang | Apache-2.0 | design reference only |
| `cuda/tc_gemm.cu` (HIP branch), `hip/hip_mma.cuh`, `hip_gemm.*` | rocWMMA | MIT | header library of the ROCm install / lane layouts checked; not copied |
| `src/backend/tpu/pjrt/sys.rs`, `pjrt/proto.rs` | OpenXLA `pjrt_c_api.h`, `compile_options.proto`, `xla_data.proto` | Apache-2.0 | layout facts / constants transcribed; header kept |
| `src/backend/npu/openvino*` | OpenVINO C API | Apache-2.0 | signatures redeclared; library loaded at run time |
| `src/backend/npu/hexagon/{qnn,graph,genie}.rs` | Qualcomm QAIRT (QNN 2.27, Genie 1.11) | proprietary | binary interface restated; SDK not shipped; terms unchecked |
| `src/backend/npu/xdna/*` | AMD Ryzen AI docs, onnx, onnxruntime | proprietary docs / Apache-2.0 / MIT | PCI ids, field numbers, `OrtApi` slot order; terms of Ryzen AI unchecked |
| `src/backend/npu/ane/mlpackage.rs` | apple/coremltools | BSD-3-Clause | format specification only |
| `src/cpu_kquant.rs`, `src/cpu_avx512/*` | ik_llama.cpp (MIT) | MIT | technique read, nothing copied |
| `rust/engine/sycl/*` | own (SPDX Apache-2.0) | Apache-2.0 | none; oneAPI is a build tool |

Headers without a copyright line that stay as they are (I may not touch `rust/engine/cuda`): the unchanged vLLM Marlin files `dequant.h`, `marlin_mma.h`, `marlin_dtypes.cuh`,
`marlin.cuh`, `kernel.h`, `core/scalar_type.hpp` have no licence header in the vLLM tree either; their terms are those of the directory (Apache-2.0, stated in `THIRD_PARTY.md` and `NOTICE`).
A follow-up that may edit `rust/engine/cuda` should add the vLLM SPDX lines to them and a `NOTICE` pointer to `iq_tables.cuh`.

## 2. Supply chain: `deny.toml`, `.cargo/audit.toml`, `supply-chain.yml`

`cargo deny --locked check` (advisories, bans, licenses, sources): **ok** after the changes below; `cargo audit --deny warnings`: **ok** (1329 advisories loaded 2026-10-15, 342 crates).

Found and fixed:

* RUSTSEC-2026-0346..0349 in `rustls 0.23.45` (transitive via reqwest): `cargo update -p rustls` to 0.23.46 (lockfile only, no manifest change; tests and clippy re-run).
* `fllm-checkpoint` / `fllm-engine` path dependencies without version counted as wildcards: the three crates got `publish = false`.
* `borrow-or-share` is `MIT-0` (via jsonschema/llguidance): MIT-0 added to the allow-list.

Found and left (reported, not fixed):

* RUSTSEC-2024-0436 `paste 1.0.15` unmaintained (proc-macro, compile time only), pulled by `tokenizers =0.23.2`, which is the latest stable tokenizers release. Ignored in `deny.toml`
  and `.cargo/audit.toml` with the reason; it goes away only with another tokenizers major/rc.
* The exact pins were not touched. Latest crates.io versions on 2026-10-15: `tokenizers` 0.23.2 (= pin), `wgpu` 30.0.1 (= pin), `llguidance` 1.9.1 (pin 1.8.0), `jsonschema` 0.58.6 (pin 0.29.1).
  Moving the last two is a version bump, which was out of scope. Relaxing `=` to a caret range changes nothing in locked builds and was not done either.
* Duplicate versions (warnings only): base64 x3, bit-set, bit-vec, getrandom, hashbrown, miniz_oxide, syn, windows-sys and a few more.

Dependency licence summary (all features, all targets, dev-dependencies included; a crate with `A OR B` counts for both; "relied on" is what the allow-list accepts):

| Licence | Crates | Remark |
| --- | --- | --- |
| MIT | 234 | |
| Apache-2.0 (incl. `WITH LLVM-exception` forms: wasi, wasip2, wit-bindgen) | 195 | |
| Unicode-3.0 | 19 | icu_* and friends |
| Zlib | 8 | bytemuck, miniz_oxide, zune-* |
| ISC | 5 | ring, rustls, rustls-webpki, hyper-rustls, untrusted |
| BSD-3-Clause | 4 | matchit, moxcms, pxfm, subtle |
| BSD-2-Clause | 2 | zerocopy |
| MIT-0 | 1 | borrow-or-share |
| CDLA-Permissive-2.0 | 1 | webpki-roots (CA list) |
| not relied on (`OR` alternative only) | 0BSD (adler2), BSL-1.0 (ryu), Unlicense (aho-corasick, memchr, byteorder-lite), LGPL-2.1-or-later (r-efi, triple-licensed MIT/Apache/LGPL: MIT is used) | no GPL/LGPL/AGPL obligation arises |

`.github/workflows/supply-chain.yml`: jobs `deny` and `audit` on pull request, push to main, a weekly schedule (Monday 04:17 UTC) and manual start; `permissions: contents: read`;
actions pinned by full SHA (`actions/checkout` v6.1.0, `dtolnay/rust-toolchain` as in the other workflows, `taiki-e/install-action` v2.87.28 `1168f681...`, a lightweight tag, so the tag SHA is the commit).
Both jobs install prebuilt binaries (`taiki-e/install-action`) instead of compiling them. **Not run on GitHub**; it was linted with actionlint (0 findings over all 10 workflows).

## 3. Version and `--version`

* Root `Cargo.toml` has `[workspace.package]` (`version = "0.1.2"`, `edition = "2021"`, `license`, `repository`, `rust-version = "1.89"`); all three crates inherit. `fllm-gguf-probe` moved from
  edition 2024 to 2021 and compiled unchanged. `Cargo.lock` workspace entries updated to 0.1.2.
* MSRV: **1.89**. Tried: 1.85 fails (locked deps need 1.86 to 1.89), 1.88 fails (`uuid 1.27` needs 1.89), 1.89 `cargo check --locked --workspace --all-targets` passes. The code was not tested lower. `msrv.yml`
  reads `rust-version` from `Cargo.toml` and builds with that toolchain (default features).
* `fllm-rs --version` / `-V` / `version`: `fllm-rs 0.1.2 (git bb3153e, 2026-10-15) features: avx512 backends: cpu` (new `rust/engine/src/version.rs`; `build.rs` adds `FLLM_GIT_SHORT`, `FLLM_BUILD_DATE`,
  `FLLM_FEATURES`). `git` is `unknown` without git, outside a checkout and when the manifest is not tracked (tarball inside another repository). `FLLM_GIT_COMMIT` and `SOURCE_DATE_EPOCH` override.
* The same string appears in the startup banner (printed before `fllm-rs listening on ...`), `GET /api/version` (`build`; `version` stays `0.4.1-rust` on purpose, Ollama clients compare it, plus `fllm_version`),
  `GET /info` (`build`, `sha`), `GET /props` (`build_info`), `fllm-rs devices --json` (`version_long`) and the `converter` field of the conversion sidecar. There is no GGUF `general.*` key carrying the version in this
  tree (the checkpoint crate writes none); the sidecar JSON is the conversion metadata that embedded the crate version.
* `release.yml`: new first job `check-version` (the `prepare` job needs it): resolves the tag like `prepare` does, checks out that tag and fails if `v<workspace version>` differs. Nothing else in the workflow changed.
  Tested the version extraction (awk) against the real `Cargo.toml`; the job itself has not run on GitHub.
* `docs/releasing.md`: release procedure, what each workflow does, what a release does and does not guarantee.

## 4. Installers

| Item | Default | Meaning |
| --- | --- | --- |
| `install.sh --version TAG` / bare `TAG` | newest release | release to install (`0.1.2` gets a `v`) |
| `install.sh --prefix DIR` | none | install to `DIR/bin` |
| `FLLM_INSTALL_DIR` | `$HOME/.local/bin` | exact install directory (was `./.local/bin` of the clone) |
| `install.sh --variant cpu\|cuda` | `cpu` | `cuda` = asset `fllm-linux-x86_64-cuda.tar.gz`; missing asset prints the `docker run --gpus all ghcr.io/marco-lu2508/fllm:cuda ...` commands and exits 1 |
| `install.sh --modify-path` | off | append a guarded block (`# >>> fllm >>>` ... `# <<< fllm <<<`) to `~/.bashrc` (macOS: `~/.bash_profile` when there is no `.bashrc`), `${ZDOTDIR:-~}/.zshrc`, `~/.config/fish/config.fish`, otherwise `~/.profile`; once only |
| `install.sh --uninstall` | | remove the binary and the PATH block from every rc file that has it |
| `install.sh --build`, `--no-verify`, `FLLM_REPO_URL`, `FLLM_FORGE`, `FLLM_API_URL`, `RUST_FEATURES` | as before | `--build` needs a clone |
| `install.ps1 -Version`, `-Variant auto\|default\|cpu\|cuda\|hip`, `-NoVerify`, `-Build` | as before | |
| `install.ps1 -InstallDir`, `FLLM_INSTALL_DIR` | `%LOCALAPPDATA%\fllm\bin` | install folder (was `.local\bin` of the clone) |
| `install.ps1 -AddToPath` / `-NoAddToPath` | prompt in an interactive host, otherwise no change | user PATH only, never the machine PATH |
| `install.ps1 -Uninstall` | | remove `fllm-rs.exe` and the PATH entry |

What changed in behaviour: `curl ... | sh` and `irm ... | iex` work without a clone (the old script used `$0` for the install location and the git remote); the `to_https` bug (`$1` never passed, `parameter not set` under
`set -u`) is fixed by reading the URL from stdin, and the remote is now used only when the script is a file inside a clone. `run.sh` finds the binary in `$FLLM_INSTALL_DIR`, `./.local/bin`, the PATH and `~/.local/bin`.
Checksums: the `.sha256` file of the release is checked as before; additionally a `<sha256>  <asset>` list in the release notes (API `releases/tags/<tag>`) is compared when present, and the two must agree.
The scripts print what this protects against (damaged/mismatched downloads) and what it does not (not a signature; whoever replaces the asset replaces the checksum). `release.yml` does not write the checksum list into the
notes yet (left alone on purpose, see "Not done"). macOS: the quarantine attribute is removed when present (`xattr -d com.apple.quarantine`), with an explanation. Windows: unsigned-binary note, `Unblock-File` on the installed exe,
TLS 1.2 forced for Windows PowerShell 5.1, work done inside a function so that `StrictMode` and `$ErrorActionPreference` do not leak into the session that ran `iex`, and errors do not `exit` an interactive session.

## 5. Tests (levels)

* Level labels: L0 = compiles/lints only, L1 = ran on a device (none here), L2 = text matches CPU, L3 = benchmarked. Nothing in this change is GPU related; the checks are host-level.
* `cargo test --release --locked -p fllm-engine -p fllm-checkpoint -p fllm-gguf-probe`: engine 353 passed / 3 ignored, checkpoint 41 passed, gguf-probe 3 passed (includes the new `version::tests`).
* `cargo clippy --release --workspace --all-targets --locked -- -D warnings`: clean for default features and `--features rust-gpu`. `--features cuda`: see REPORT.md for the result of the CUDA image run.
* `scripts/test-install-sh.sh` (38 checks): fake release server (`python3 -m http.server`), pipe mode, newest-release resolution, checksum from file and notes, mismatch aborts, notes/file disagreement aborts,
  notes-only, no-checksum refusal and `--no-verify`, missing tag, missing CUDA asset message, `--prefix`, `FLLM_INSTALL_DIR`, idempotent `--modify-path` (bash), fish, `--uninstall` keeps the other rc lines, `--help`, unknown option.
  Passed in `ubuntu:24.04` (dash), `debian:12` (dash), `alpine:3.20` (busybox ash), `debian:12` with bash, and curl-less `alpine:3.20` / `ubuntu:24.04` (busybox wget, GNU wget). shellcheck (`koalaman/shellcheck:stable`, `-s sh`, `-s dash`, `-s bash` for run.sh): clean.
* `scripts/test-install-ps1.sh` (35 checks, `mcr.microsoft.com/powershell`, PowerShell 7.4.2 on Linux with faked `LOCALAPPDATA`, `SystemRoot`, `PROCESSOR_ARCHITECTURE`, a file standing in for the user PATH): parse, install flow, checksum
  from file and notes, mismatch/missing tag/no checksum, `-AddToPath` twice, NVIDIA detection via `nvcuda.dll` selecting the CUDA asset and the fallback when the release has none, `-Uninstall`, `iex` flow, script-block flow with parameters,
  unreachable server with a surviving session. PSScriptAnalyzer `PSUseCompatibleSyntax` for 5.1 and 7.0: no findings. **Not run on real Windows PowerShell 5.1**, not run against GitHub.
* `actionlint` (docker `rhysd/actionlint:latest`, repository with `.git`): 0 errors in all 10 workflow files.

## 6. README text (for the central README)

````
## Install

```sh
curl -fsSL https://raw.githubusercontent.com/Marco-Lu2508/fLLM/main/install.sh | sh        # Linux, macOS -> ~/.local/bin/fllm-rs (add --modify-path for PATH)
irm https://raw.githubusercontent.com/Marco-Lu2508/fLLM/main/install.ps1 | iex             # Windows (PowerShell) -> %LOCALAPPDATA%\fllm\bin
fllm-rs --version
```

The installers verify the release's SHA256 file (and the checksum list in the release notes when there is one), install to a per-user folder and change PATH only when you ask them to.
The binaries are not code-signed. Docker and source builds: see docs/getting-started.md. Third-party code and licences: THIRD_PARTY.md, NOTICE, licenses/. Release procedure: docs/releasing.md.
````

Also: the README line `./install.sh   # newest release -> ./.local/bin/fllm-rs` (line 23) and `./install.sh --build   # build and install into ./.local/bin` (line 292) are outdated by this change.

## 7. Known limits

* `release.yml`, `msrv.yml` and `supply-chain.yml` have not run on GitHub; the install scripts have not run against the real GitHub release API or on real Windows/macOS.
* No Linux CUDA release asset exists; `--variant cuda` therefore ends with the Docker commands on current releases.
* MSRV 1.89 comes from locked dependencies; the code was not tested with older compilers.
* Release assets do not yet bundle `licenses/`, `NOTICE` and `THIRD_PARTY.md`, and the release notes do not yet list checksums (the installers use the list when someone adds it).
* The unchanged vLLM Marlin files have no licence header of their own and `iq_tables.cuh` only refers to THIRD_PARTY.md (rust/engine/cuda was out of scope).
