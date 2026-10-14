# Release License Report

Review of files containing third-party-derived code.

- THIRD_PARTY.md and license headers are present for several backends (vLLM-derived pieces, CUTLASS/Marlin, FlashInfer references in docs). 
- Current code contains code derived from llama.cpp/ggml concepts in quantization files; each carries provenance mentions where present. A file-by-file header audit is still recommended before publication.
- Potential gaps: examples and micro-benchmarks may lack license headers; ensure every file copied/adapted from an external project has both a file header and a THIRD_PARTY.md entry.
- No GPL/AGPL-licensed material should be merged unless explicitly approved.
