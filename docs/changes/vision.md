# Vision: multimodal pipeline, Qwen-VL family (fragment)

User guide: `docs/vision-audio.md`.

## Features

* `fllm-rs convert` copies the vision tower of Qwen2-VL, Qwen2.5-VL and Qwen3-VL checkpoints into the
  same GGUF (tensors `v.hf.*`, metadata `fllm.vision.*`, `fllm.rope.mrope_*`); `fllm-rs serve` enables
  image and video input from that file without any extra flag.
  `fllm-rs convert Qwen2.5-VL-3B-Instruct/ model.gguf --outtype f16 && fllm-rs serve --model model.gguf`
* General interface `MultimodalEncoder` (`rust/engine/src/vision.rs`): preprocessing, tower, projector,
  placeholder expansion; `vision/pipeline.rs` inserts embeddings, DeepStack rows and position ids into the
  token stream; `server/media.rs` collects `image_url` / `input_image` / `video_url` / `video` parts.
* Real multimodal RoPE (three position axes, sectioned or interleaved) in the decoder: CPU rotation per
  lane, accelerator path as three ordinary rope calls with per-axis tables. Text-only requests are unchanged.
* DeepStack (Qwen3-VL): features added after decoder layers 0..2 (CPU and accelerator path).
* Animated GIF / APNG and frame lists as video; frame sampling and frame rate from the frame delays.
* `/v1/models` reports `capabilities` and `modalities`.
* Limits: see the table below. The vision admission limit is no longer 1.

Commands to check it (CPU, no network, needs torch + transformers):

```
cargo build --release -p fllm-engine --bin fllm-rs --example verify_multimodal
python scripts/verify_multimodal.py --generate --fllm target/release/fllm-rs \
    --example target/release/examples/verify_multimodal
```

## Flags and variables

| Name | Default | Meaning |
| --- | --- | --- |
| `--no-vision` | off | do not use the tower stored in the model file |
| `--max-image-pixels N` / `FLLM_MAX_IMAGE_PIXELS` | 1003520 | pixel budget per image or frame |
| `--limit-mm-per-prompt N` / `FLLM_MAX_MEDIA_PER_PROMPT` | 4 | images and clips per prompt |
| `--vision-concurrency N` / `FLLM_VISION_CONCURRENCY` | 2 | concurrent image preprocessing and encoding |
| `FLLM_CONVERT_VISION=0` | unset | converter leaves the vision tower out |
| `VERIFY_PLAIN_ROPE`, `VERIFY_NO_DEEPSTACK` | unset | `verify_multimodal` example only: negative controls |

## THIRD_PARTY entries

* Hugging Face `transformers` (Apache-2.0): the architectures of the Qwen2-VL, Qwen2.5-VL and Qwen3-VL
  vision towers, `get_rope_index`, `smart_resize` and the video processing rules were re-implemented in
  Rust from the published model code and used as the numerical reference in `scripts/verify_multimodal.py`;
  no source code was copied.
* Cargo features `gif` and `webp` of the `image` crate (MIT / Apache-2.0) add the crates `gif`, `weezl`,
  `color_quant`, `image-webp` and `byteorder-lite` to `Cargo.lock`; all permissive.

## Test levels

* L2 on the CPU for Qwen2-VL, Qwen2.5-VL and Qwen3-VL: tiny random checkpoints converted by `fllm-rs convert`,
  logits of all positions for an image prompt and for a four-frame clip compared with transformers (relative
  error about 5e-7, tolerance 2e-3, equal arg-max); negative controls for image, M-RoPE and DeepStack; served
  chat (`/v1/chat/completions` with a base64 image, greedy) equals the reference continuation.
* L0 (compiles, not run): accelerator M-RoPE (three rope calls with per-axis tables) and DeepStack add
  (`residual_add`) in `forward_backend_inner`; `--features rust-gpu` and `--features cuda` builds.
* Unit tests: preprocessing (`smart_resize`, merge-window patch order, data URLs, GIF sampling), M-RoPE
  positions, prompt construction, vision plan of the converter, request part collection.
* SmolVLM / Idefics3 (`--mmproj`) moved onto the new interface; not run end to end here.

## Known limits

* Not implemented: Gemma 3 / PaliGemma image attention masks and towers, Llama 4, Pixtral, LLaVA, InternVL,
  MiniCPM-V, Phi vision, GLM-4V, Kimi-VL, Molmo, LFM2-VL, Cohere2-Vision, Nemotron-VL, all audio.
* No real checkpoint was run; the Qwen3-VL video placeholder string follows the reference processor code
  (time stamp text, then a vision start / end pair around each frame) but was not compared with a real
  prompt template.
* Sequences with media bypass the prefix cache and CUDA graphs; one sequence with media per request.
* Remote image URLs wait for the server's fetcher (`vision::preprocess::set_image_source`).
* Real video containers are not decoded.
