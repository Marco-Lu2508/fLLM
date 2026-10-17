# Releasing

How a version is cut, what each workflow does and what a release does and does not guarantee.

## The version

One number, in the root `Cargo.toml`:

```toml
[workspace.package]
version = "0.1.2"
```

The three crates (`fllm-engine`, `fllm-checkpoint`, `fllm-gguf-probe`) inherit it (`version.workspace = true`). The same string is printed by

```sh
fllm-rs --version        # also -V and `fllm-rs version`
# fllm-rs 0.1.2 (git a1b2c3d, 2026-10-15) features: avx512 backends: cpu
```

and appears in `GET /api/version` (`fllm_version`, `build`; `version` keeps the Ollama-compatible `0.4.1-rust` because Ollama clients compare it),
`GET /info` (`version`, `sha`, `build`), `GET /props` (`build_info`), `fllm-rs devices --json` (`version`, `version_long`), the startup banner of the server
and the `converter` field of the sidecar written next to converted models. The pieces:

* `git <sha>`: `build.rs` asks `git rev-parse HEAD` (short form). It is `unknown` when git is missing, when the directory is not a checkout
  (a source tarball) and when the package manifest is not tracked by the enclosing repository (a tarball unpacked inside another repository).
  `FLLM_GIT_COMMIT=<hash>` in the environment overrides it (packagers without `.git`).
* the date is the UTC build date; `SOURCE_DATE_EPOCH` fixes it for reproducible builds.
* `features` are the cargo features of `fllm-engine` that were enabled (`none` if there are none; `default` is not listed), `backends` the compiled backends
  (`fllm-rs devices` prints the same list with the detected devices).

## Minimum supported Rust version

`rust-version` in `[workspace.package]` is the lowest toolchain that builds the workspace with the committed `Cargo.lock`. It is **1.89**: 1.85 and 1.88 were tried
and stop at the locked dependencies (`uuid 1.27` needs 1.89, `image 0.25.10` 1.88, `llguidance 1.8` 1.87, `idna_adapter` 1.86); the engine code was not tested below 1.89.
All three crates use edition 2021 (`fllm-gguf-probe` was 2024 and compiled unchanged as 2021). `msrv.yml` checks with exactly that toolchain on every pull request. When a dependency update raises
the requirement, `cargo` reports it; bump `rust-version` in the same pull request.

## Cutting a release

1. Pick the number (`0.1.3`; a tag with `-` such as `v0.2.0-rc1` stays a pre-release).
2. Bump `version` in `[workspace.package]` of `Cargo.toml`, then `cargo update -w` (rewrites the three workspace entries of `Cargo.lock`; nothing else changes).
3. Update `CHANGELOG.md` (there is none in the tree yet: create it with the first release that needs one; list the user-visible changes per version).
4. Run the checks the CI runs: `cargo test --release --locked -p fllm-engine -p fllm-checkpoint -p fllm-gguf-probe`,
   `cargo clippy --release --workspace --all-targets --locked -- -D warnings`, `cargo deny --locked check`, `cargo audit --deny warnings`.
5. Commit ("Release 0.1.3"), push to `main` through a pull request, wait for CI.
6. Tag the merge commit and push the tag:

   ```sh
   git tag v0.1.3
   git push origin v0.1.3
   ```

   `release.yml` starts on the tag. Its first job, `check-version`, fails the whole run if the tag differs from the workspace version in `Cargo.toml`
   (tag `v0.1.3` needs `version = "0.1.3"`), so a forgotten bump never produces a release whose binary says another version.
7. When the run is green, check the release page: assets, `.sha256` files, notes. A `v0.x` release stays a pre-release (not "latest"); `install.sh`/`install.ps1`
   find it through the API.

To rebuild the assets of an existing tag: Actions, `release`, Run workflow, enter the tag (the version check applies to that tag's commit too).

## What each workflow does

| Workflow | Trigger | Does |
| --- | --- | --- |
| `ci.yml` | push to main, pull request | `clippy` and `test` jobs of the CPU build |
| `platforms.yml` | push to main, pull request | builds, tests and CPU self-tests on Windows (MSVC), Apple silicon and Linux (x86_64, aarch64); manual Windows CUDA and package jobs |
| `msrv.yml` | push to main, pull request | `cargo check` with the toolchain named by `rust-version` |
| `supply-chain.yml` | push to main, pull request, weekly (Monday) | `cargo deny check` (licences, bans, sources, advisories) and `cargo audit`; the weekly run finds advisories published after the last push |
| `kernels.yml` | manual | compile-only checks of the CUDA and HIP kernels in vendor containers |
| `conformance.yml`, `bench.yml` | manual, self-hosted runners | `devices` / `selftest-gpu` on real GPUs; streaming benchmark |
| `release.yml` | tag `v*`, manual | `check-version`, then creates the release as a pre-release, builds the assets (Linux musl x86_64/aarch64, macOS arm64/x86_64, Windows MSVC, optional Windows CUDA), uploads `.sha256` files, then sets the final state |
| `docker.yml`, `docker-cuda.yml` | tag `v*`, manual | CPU image (amd64, arm64) and CUDA image to `ghcr.io/<owner>/fllm` |

Third-party actions are pinned by full commit SHA with the version in a comment; update a pin by looking up the new tag
(`git ls-remote --tags https://github.com/<owner>/<action> <tag>`, take the `^{}` entry when the tag is annotated).

## What a release guarantees

* The assets are built from a clean checkout on GitHub-hosted runners (no build cache), see the comment at the top of `release.yml`.
* Each asset has a `.sha256` file. `install.sh` and `install.ps1` verify it, and also a `<sha256>  <asset>` list in the release notes when the notes contain one
  (put it there by hand with `sha256sum fllm-* > notes`; the workflow does not do it yet). This detects a damaged, truncated or mismatched download. It is not a
  signature: whoever can replace the asset can replace the checksum.
* The binaries are not code-signed (Windows) or notarized (macOS), and there is no provenance attestation yet. `install.sh` clears the macOS quarantine flag of what it
  installs; Windows SmartScreen or Defender may warn about the unsigned `.exe`.
* Licence texts for the third-party code in the binary are in `licenses/`, the modification notices in `NOTICE`, the inventory in `THIRD_PARTY.md`; the release assets
  do not contain them yet, the source tree does.
* The Linux CUDA binary (`fllm-linux-x86_64-cuda.tar.gz`, which `install.sh --variant cuda` asks for) is not built by `release.yml`; until it is, the CUDA route on Linux
  is the Docker image (`docker.md`) or a source build.
