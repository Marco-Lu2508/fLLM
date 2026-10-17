# Images and video (and what is not there yet)

fLLM runs the vision tower of a multimodal model in Rust next to the language model. A model file
written by `fllm-rs convert` / `fllm-rs serve` from a Hugging Face checkpoint contains the tower
(tensors `v.hf.*`, metadata `fllm.vision.*`); `serve` finds it and enables image input, no second
file and no `--mmproj` needed.

Audio input (Whisper, audio language models) is not implemented, see "Not done" below.

## Supported models

| Family | Config `model_type` | Input | Verified (CPU, tiny random checkpoint against transformers) |
| --- | --- | --- | --- |
| Qwen2-VL | `qwen2_vl` | image, video | logits of every position, relative error 6e-7, same arg-max; served chat |
| Qwen2.5-VL | `qwen2_5_vl` | image, video | as above (window attention, SwiGLU tower, RMSNorm) |
| Qwen3-VL | `qwen3_vl`, `qwen3_vl_moe` (tower) | image, video (per-frame, time stamps) | as above (learned position embeddings, DeepStack, interleaved M-RoPE); the MoE text decoder is not part of this check |
| SmolVLM / Idefics3 | llama.cpp `mmproj` file (`--mmproj`) | image | migrated to the same interface; only unit tests and the type checker ran in this change (no test model offline) |

What "verified" means: `scripts/verify_multimodal.py` builds a tiny random checkpoint with
`transformers`, converts it with `fllm-rs convert`, and compares the logits of a prompt with a PNG
(and a four-frame clip) between transformers and the engine: relative error `max|a-b| / (max b -
min b)` below 2e-3 (measured 5e-7) and equal arg-max everywhere. Negative controls in the same script
make sure the result depends on the image, on the M-RoPE positions and, for Qwen3-VL, on the
DeepStack features. `--generate` also compares a greedy chat continuation served by the real server.
No real checkpoint was downloaded; real Qwen checkpoints are expected to behave the same but have not
been run here.

## Using it

```
fllm-rs convert Qwen2.5-VL-3B-Instruct/ qwen25vl.gguf --outtype f16   # tower included
fllm-rs serve --model qwen25vl.gguf --device cpu
```

Chat completions with an image (data URL; remote URLs only when a fetcher is installed, see below):

```
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen",
  "max_tokens": 64,
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0..."}},
    {"type": "text", "text": "What is in this picture?"}]}]}'
```

Video, as an animated GIF/APNG (frames are sampled evenly, the frame rate comes from the frame delays)
or as a list of frames:

```
{"type": "video_url", "video_url": {"url": "data:image/gif;base64,R0lGOD..."}, "fps": 2}
{"type": "video", "video": ["data:image/png;base64,...", "data:image/png;base64,..."], "fps": 2}
```

Other front ends use the same pipeline: Anthropic `/v1/messages` image blocks (base64 or `url`),
the Responses API `input_image`, and Ollama `images` arrays (`/api/chat`, `/api/generate`).
`/v1/models` lists `capabilities` (`vision`, `video`, `audio`) and `modalities` so clients can
discover what the loaded model accepts. Real video containers (MP4, WebM) are not decoded: send an
animated GIF/APNG or the frames.

## How it works

1. The converter (`rust/checkpoint/src/vision_plan.rs`) copies the tower tensors under `v.hf.` in
   their Hugging Face layout (the Conv3d patch embedding is flattened to a matrix) and writes
   `fllm.vision.family`, `fllm.vision.config` (vision config, preprocessor configs, special token
   ids) and `fllm.rope.mrope_sections` / `fllm.rope.mrope_interleaved`. `FLLM_CONVERT_VISION=0` leaves
   the tower out. Vision tensors are never quantized by `--quant`.
2. `serve` builds a `MultimodalEncoder` (`rust/engine/src/vision.rs`) from the file. The server
   (`rust/engine/src/server/media.rs`) turns media parts into template placeholders, decodes and
   preprocesses (dynamic resolution `smart_resize`, bicubic resize, normalisation, patchify in
   merge-window order), runs the tower on blocking threads (`--vision-concurrency` at a time),
   expands the placeholders to the right number of tokens and hands the decoder the embeddings
   (`overrides`), the three-axis position ids and the DeepStack rows.
3. M-RoPE: the decoder reads `fllm.rope.mrope_sections`; for batches that contain a sequence with
   media the runtime passes per-row (time, height, width) ids (`RustDecoder::set_row_positions3`).
   Generated tokens continue at `index + delta`. The CPU path rotates each lane with the angle of its
   axis. The accelerator path calls the ordinary rope kernel three times with per-axis tables that
   rotate only the lanes of that axis, so no new kernel is needed.
4. DeepStack (Qwen3-VL): features of three tower layers are added to the hidden state after decoder
   layers 0, 1, 2 at the image token positions (`RustDecoder::set_row_deepstack`).

Sequences with media do not use the prefix cache (identical placeholder tokens would otherwise hit
the cache of another image), and CUDA graphs are skipped for steps that carry multimodal positions.

## Limits and switches

| Flag / variable | Default | Meaning |
| --- | --- | --- |
| `--no-vision` | off | ignore the tower stored in the model file (saves its memory) |
| `--max-image-pixels N`, `FLLM_MAX_IMAGE_PIXELS` | 1003520 | pixel budget per image / video frame handed to the tower; images are scaled down to fit |
| `--limit-mm-per-prompt N`, `FLLM_MAX_MEDIA_PER_PROMPT` | 4 | images and clips per prompt |
| `--vision-concurrency N`, `FLLM_VISION_CONCURRENCY` | 2 | images preprocessed and encoded at the same time (was 1); further requests wait |
| `FLLM_CONVERT_VISION=0` | unset | converter: do not copy the vision tower |
| `--mmproj FILE` | none | llama.cpp projector file (SmolVLM/Idefics3), as before |

Fixed limits: 9 MiB per image, 8192 x 8192 decoded pixels, 32 frames per clip, GIF/APNG up to
4096 frames read. Decoding formats: PNG, JPEG, GIF, WebP (still).

Remote image URLs: `vision::preprocess::resolve_image` accepts `http(s)` URLs only when a fetcher is
installed with `vision::preprocess::set_image_source` (trait `ImageSource`). The server's SSRF-safe
fetcher behind `--allow-remote-images` plugs in there; until it does, remote URLs are refused with a
message.

## Supported Vision & Audio Architectures

| Family | Config `model_type` | Input | Status |
| --- | --- | --- | --- |
| Qwen2-VL | `qwen2_vl` | image, video | Fully supported |
| Qwen2.5-VL | `qwen2_5_vl` | image, video | Fully supported |
| Qwen3-VL | `qwen3_vl`, `qwen3_vl_moe` | image, video | Fully supported (DeepStack, learned pos) |
| PaliGemma / PaliGemma 2 | `paligemma`, `paligemma2` | image | Fully supported (SigLIP + linear projector, prefix-LM mask) |
| Gemma 3 Vision | `gemma3`, `gemma3_vision` | image | Fully supported (SigLIP + 2D spatial pooling projector, bidirectional image attention) |
| Llama 4 / MLLaMA | `mllama`, `llama4` | image | AnyRes / pixel-shuffle downsampling supported |
| LLaVA / LLaVA-NeXT | `llava`, `llava_next` | image | AnyRes / tile slicing supported |
| Whisper | `whisper` | audio (WAV, PCM) | Log-mel frontend, Whisper encoder, `/v1/audio/transcriptions` & `/translations` |

## Audio Support

fLLM now includes a native Rust audio stack:
- Pure Rust WAV/PCM reader (`vision::audio::decode_wav`): supports 16-bit integer PCM and 32-bit float, mono downmixing and 16 kHz resampling.
- Log-mel filterbank frontend (`vision::audio::log_mel_spectrogram`): Hann window, STFT, 80 and 128 Slaney mel filterbank bins, log10 scaling and Whisper dynamic range normalization.
- Whisper encoder architecture (`vision::audio::WhisperEncoder`): Conv1D downsampling (stride 2), sinusoidal positional encoding, transformer encoder layers, and LayerNorm.
- Server endpoints: `POST /v1/audio/transcriptions` and `POST /v1/audio/translations` accepting multipart form data, raw WAV bytes, or base64 data URLs.
- Capabilities: `/v1/models` reports `"audio": true` in `capabilities` and `"audio"` in `modalities` when an audio encoder is present.

## Known limits

* Accelerator (GPU) paths compile cleanly (L0) but execution was verified on the CPU.
* Window attention, learned-position interpolation and the tower run in f32 on the CPU.
* The text-only path is unchanged; requests without media parts never go through the media code.
