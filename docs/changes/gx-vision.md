# Vision & Audio Stack: SigLIP, Gemma 3, PaliGemma, AnyRes, Audio & Whisper (gx-vision)

User guide: `docs/vision-audio.md`.

## Features

* **Vision Plan & Checkpoint Converter**:
  - `rust/checkpoint/src/vision_plan.rs` extended with families: `paligemma` (`paligemma`/`paligemma2`), `gemma3` (`gemma3_vision`), `mllama`/`llama4`, `llava`/`llava_next`, and `whisper`.
  - Automatic projector tensor remapping: `multi_modal_projector.*` and `mm_projector.*` mapped to `v.hf.projector.*`.
  - Config extraction handles `vision_config`, `encoder_config`, and multimodal special token IDs.

* **SigLIP Vision Tower & Projectors (`rust/engine/src/vision/siglip.rs`)**:
  - `SiglipEncoder` implementing `MultimodalEncoder`.
  - ViT encoder: patch embedding (14x14 or configurable), learned position embeddings, pre-LayerNorm, Multi-Head Self-Attention, MLP (GELU), and post-LayerNorm.
  - Projectors:
    - Linear projector for PaliGemma (`v.hf.projector.linear.weight`/`bias`).
    - 2D Spatial pooling projector for Gemma 3 (2x2 spatial downsampling + linear projection).
  - Dynamic registration in `vision::load_from_model`.

* **Bidirectional Image Attention & Prefix-LM**:
  - `rust/engine/src/rust_decoder.rs`: added `row_visible_lengths` and `set_row_visible_lengths`.
  - `attention_reference`: rows with custom visible lengths attend across their full visible window (e.g. image end or prompt prefix) rather than strictly causal `lengths[r]`.
  - Automatic fallback from tiled attention to reference attention when custom visible lengths are active.
  - Gemma 3: image tokens attend bidirectionally across their image span; text continues causally.
  - PaliGemma: prefix-LM mask (all prompt prefix tokens attend bidirectionally).
  - Runtime and pipeline integration: `MediaInputs.visible_lengths` plumbed through `attach_media` and forward pass.

* **AnyRes & Spatial Tiling (`rust/engine/src/vision/preprocess.rs`)**:
  - `best_fit_grid(width, height, candidate_grids, max_tiles)`: optimal aspect-ratio matching grid selection with tie-breaking for minimum tile count.
  - `slice_into_tiles(image, grid)`: slices high-resolution images into uniform spatial tiles.
  - `pixel_unshuffle(x, h, w, c, scale)`: spatial downsampling unpacking blocks into channel dimension.

* **Audio Stack & Whisper (`rust/engine/src/vision/audio.rs`)**:
  - Pure Rust WAV/PCM reader (`decode_wav`): parses RIFF WAV (16-bit PCM, 32-bit float, mono and stereo downmixed to mono) and raw PCM; linear resampler to 16 kHz.
  - Log-mel filterbank frontend (`log_mel_spectrogram`): periodic Hann window, centered STFT, 80 or 128 triangular Slaney mel filterbank bins, log10 compression, dynamic range clamping (max - 8.0), and Whisper normalization.
  - Whisper Encoder architecture (`WhisperEncoder`): Conv1D downsampling (stride 2), sinusoidal positional encoding (1500 positions), transformer encoder layers (pre-LN, MHA, MLP), and final LayerNorm.
  - `WhisperMultimodalEncoder` implementing `MultimodalEncoder`.

* **HTTP Audio Endpoints & Discovery (`rust/engine/src/server.rs`)**:
  - New endpoints: `POST /v1/audio/transcriptions` and `POST /v1/audio/translations`.
  - Multipart/form-data audio file extraction (boundary parsing, `name="file"` / `filename=`), base64 JSON, and raw audio byte input.
  - `/v1/models` reports `"capabilities": {"vision": ..., "video": ..., "audio": true}` and `"modalities": [..., "audio"]` when audio capability is present.

## Test Levels & Verification

* **Quality Gate Passed**:
  - `cargo test --release --locked -p fllm-engine -p fllm-checkpoint -p fllm-gguf-probe`: 380 engine tests, 45 checkpoint tests, 3 gguf-probe tests passed (0 failed).
  - Clippy clean: 0 warnings on default features.
  - Clippy clean: 0 warnings on `--features rust-gpu`.
  - Clippy clean: 0 warnings on `--features cuda`.
  - CRLF line endings preserved on `runtime.rs`, `draft.rs`, `gguf.rs`, `gguf-probe/src/lib.rs`, `checkpoint/src/tests.rs`, `checkpoint/src/lib.rs`.
  - Privacy check: 0 private traces.
