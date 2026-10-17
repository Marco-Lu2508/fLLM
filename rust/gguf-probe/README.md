# Rust GGUF metadata/index probe

This crate exposes `MappedGguf`, a read-only mmap-backed GGUF reader. It parses metadata and tokenizer arrays, indexes tensor names/dimensions/types, calculates byte lengths from supported GGML block sizes, validates every tensor range against the file, and returns validated zero-copy raw bytes through `MappedGguf::tensor_bytes`.

Against the active Python `fllm.gguf_reader.GGUFModel`, three fresh-process Python/Rust/Rust/Python ABBA pairs matched the complete normalized index: 35 metadata fields, 151,936 tokenizer tokens, and 310 tensor descriptors including byte lengths. Parser medians were 1,579.04 ms for Python and 22.64 ms for Rust (69.8x for this parser-only measurement). The host, model SHA-256, all 12 timings, and per-sample index hashes were recorded in the ABBA metadata benchmark archive.

A separate full-payload check matched SHA-256 for all 310 tensor byte ranges (390,753,280 bytes) between Rust mmap slices and Python's mmap-backed tensors. Per-tensor hashes and model identity were recorded in the payload-parity benchmark archive.

These checks establish GGUF index and raw-byte parity only; this crate does not decode quantized weights or instantiate a model. The decoder in `rust/engine` consumes the mapped tensor slices directly.

```sh
cargo test --manifest-path rust/gguf-probe/Cargo.toml --release
cargo build --manifest-path rust/gguf-probe/Cargo.toml --release
python3 rust/gguf-probe/bench_abba.py
python3 rust/gguf-probe/verify_payloads.py
```