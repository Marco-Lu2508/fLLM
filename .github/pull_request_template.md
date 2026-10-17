## What and why

<!-- What does this change, and why? Link the issue if there is one. -->

## How it was tested

<!-- Exact commands and results. Say which hardware ran them. -->

- [ ] `cargo test --release --locked -p fllm-engine -p fllm-checkpoint -p fllm-gguf-probe`
- [ ] `cargo clippy --release --workspace --all-targets --locked -- -D warnings` (and with the feature flags you touched)
- [ ] `fllm-rs selftest-cpu` / `selftest-gpu --device ...` where kernels changed
- [ ] Model-level check (`scripts/verify_architectures.py`, `make test-model`, ...) where behaviour changed

Test level reached on hardware (see docs/test-levels.md): L0 / L1 / L2 / L3, on: <!-- device, driver, OS -->

## Checklist

- [ ] New behaviour that can change numerics or performance has an environment switch with a documented default
- [ ] Unsupported cases return a clear error or fall back; nothing is stubbed silently
- [ ] Documentation updated (docs/, CHANGELOG.md under "Unreleased")
- [ ] Code adapted from another project keeps its licence header and has a THIRD_PARTY.md entry; nothing is copied from llama.cpp
- [ ] No models, large dumps, credentials, private host names or e-mail addresses are committed
- [ ] Line endings of existing files are unchanged
