# Troubleshooting

Symptoms, causes and fixes. Exact messages come from
`rust/engine/src/bin/serve.rs`, `rust/engine/src/devices.rs`,
`rust/engine/src/backend/wgpu_backend.rs`, `rust/engine/src/gpu.rs`,
`rust/engine/src/encoder.rs`, `rust/engine/src/server.rs`, and from
[quickstart-windows-linux.md](../quickstart-windows-linux.md) (Docker GPU),
[docker.md](../docker.md), [configuration.md](../configuration.md).

## Device selection errors

Which string you see depends on the code path (see
[commands](commands.md#device-selection)):

| Message | Meaning / fix |
| --- | --- |
| `unsupported GPU backend X` | Decoder with the `rust-gpu` feature rejected `X`. Valid: `vulkan`, `metal`, `dx12`, `gpu`/`wgpu` (= all compiled backends), `auto`, `cuda` (CUDA builds only), `cpu`, `kind:index`. |
| `unsupported Rust GPU backend X` | BERT encoder / vision path (or the legacy GPU context) rejected `X` — including `cuda`, which the encoder never accepts, and `kind:N`, which that path does not split. Same valid set otherwise, bare kinds only. |
| `rebuild with --features rust-gpu for native Rust GPU projections` | Binary built without `rust-gpu` (decoder asked for GPU). Rebuild with the feature or use `--device cpu`. |
| `build with --features rust-gpu to enable the independent Rust GPU backend` | Same situation on the encoder path. |
| `invalid device index in '…'` | The `:N` in `kind:N` is not a number (decoder path). |
| `no native GPU adapter: …` | No adapter for the requested backend (missing driver, headless Vulkan, container without GPU). |
| `<name> is not a hardware GPU (set FLLM_ALLOW_SOFTWARE_GPU=1 to use it anyway)` | Decoder wgpu path refused a software adapter (lavapipe, WARP). Set `FLLM_ALLOW_SOFTWARE_GPU=1` to use it anyway (slower than CPU; meant for `devices`/`selftest-gpu` checks). |
| `<name> is not a hardware GPU; refusing silent CPU fallback` | Encoder/vision path (`gpu.rs`) refused a software adapter — this path honors no opt-out variable. Use `--device cpu`. |
| `the vulkan backend only supports the f16 and q8_0 KV cache (requested …)` | wgpu backends keep only `f16`/`q8_0`. Use `--device cuda` or `--device cpu` for other KV formats. |

`--device cuda` fails on purpose when no GPU is visible (e.g. `--gpus all`
missing) instead of silently running on the CPU. `--device auto` uses CUDA
when available; note its startup line reads `--device auto selects
CUDA/Metal/Vulkan when available, otherwise CPU`.

## My graphics card is old or has no Vulkan driver

`fllm-rs devices` lists the cards that a Vulkan, DirectX 12 or Metal driver serves. For a card that has none of them (an old NVIDIA or AMD
card, an old Intel graphics chip, Linux with only Mesa's OpenGL) there is `--device opengl`: it runs the portable kernels through OpenGL 4.3 /
OpenGL ES 3.1 compute shaders, so the card needs at least that version (`glxinfo -B` shows it). `--device auto` tries it after DirectX 12
and before the CPU, but never takes a software renderer (llvmpipe, softpipe). Expect it to be slower than Vulkan, and on a card with little
memory or an old integrated chip the CPU can be faster: compare with `--device cpu`. It has been tested on a software OpenGL (Mesa
llvmpipe) only, not on an old card. Quantized weights without a portable kernel are expanded when they are uploaded (the log says so).

## The model is in the GPU memory but runs on the CPU

**Symptom:** the graphics card's memory fills up when a model loads, but answers are slow and the processor is busy.

**How to see what runs where** (three places, all say the same):

* The Studio shows an amber notice when part of a model runs on the CPU although a GPU was chosen: a line in the footer (click it for the
  hardware page), a box at the top of Settings, Hardware and on the Models page. Typical wording: *Weights in GPU memory, compute mostly on the
  CPU*, *Part of this model runs on the CPU* or *The whole model runs on the CPU*, followed by the count and the reason.
* The log (`server.log` in the Studio folder, or the terminal of `fllm-rs serve`) has one line `GPU <name> (<backend>): <N> of <M> operations run on
  the CPU: <reasons>` at load, and `fllm-rs serve` prints where every layer lives (`placement: layers 0-9: on the GPU (cuda)`, `placement: layers
  10-31: attention on the GPU, feed-forward on the CPU`) and the expected speed class. An *operation* here is the attention (or other mixer) half
  or the feed-forward half of one layer; a model has two per layer. Single device operations that fall back (an expert product, an attention
  with a head wider than 512) are counted per call and logged when the first one happens.
* `GET /api/studio/status` has `device_fallbacks` (`mode`: `device`, `partial`, `mostly_cpu`, `cpu`; the counts; `reasons`; `operation_fallbacks`
  with call counts), `/metrics` has `fllm_device_fallback_operations`, `fllm_device_fallback_whole_model` and `fllm_device_fallback_calls_total`.

**One line per graphics card: `fllm-rs doctor --gpu`** (or the button *Test this GPU* on every device card in Settings, Hardware). It multiplies a
quantized matrix with one token and with 64 tokens on the GPU and on the CPU and prints `OK (12 times faster than the CPU)`, `FALLS BACK: <reason>`
(the device cannot be created, a kernel fails or an operation goes to the CPU), `SLOWER THAN THE CPU` or `WRONG RESULT`. Send this line, the output
of `fllm-rs devices` and the log when you ask for help.

**Causes, most likely first:**

| Cause | What the notice says | What to do |
| --- | --- | --- |
| The model does not fit into the free GPU memory, so the memory plan keeps the feed-forward weights of the last layers in host memory and the CPU computes them every token | `not enough device memory: the feed-forward weights of these layers stay in host memory` and `N of M layers fit on the device completely` in the log | close programs that use GPU memory (browsers, games, a second model), use a smaller quantization (Q4_K_M instead of Q6_K/Q8_0) or a smaller model, a smaller context (`--ctx 4096`), fewer sequences (`--max-seqs 2`); Windows keeps about 0.5 to 1 GB of the card for the display |
| A model with a recurrent part (Qwen3.5 / Qwen3-Next DeltaNet layers, Mamba hybrids) reserved device memory for 16 conversations | the same notice; the log says `16 sequence slots would reserve ... MiB of recurrent states` | fixed: the plan lowers the slots itself before it gives up layers. Measured on an RTX 2000 Ada laptop GPU (8 GB) with Qwen3.5-9B Q4_K_M: 10 of 32 layers on the GPU and about 23 tokens/s before (`--max-seqs 16` keeps that plan), all 32 layers and about 41 tokens/s now; `--max-seqs N` sets the number by hand |
| Layers of the model have no kernels for this backend (a recurrent mixer, an attention variant, a gate) | the reason names the feature (`a recurrent (DeltaNet / Mamba) mixer without device kernels ...`) | try another backend (`--device cuda` / `vulkan`), see [architectures.md](../architectures.md); `--strict-device` makes the load fail instead of running them on the CPU |
| The graphics driver is too old or missing, so the CUDA/ROCm backend cannot start and the model runs on another backend or the CPU | `the whole model runs on the CPU: ...` or a `native vulkan backend could not be used` notice | update the NVIDIA/AMD driver; `fllm-rs devices` shows what the program can see |
| It only looks like the CPU | the notice is absent and `doctor --gpu` says OK | the Windows Task Manager shows the *3D* engine by default; compute work (CUDA, Vulkan) is under *Compute_0* / *Cuda* (change one of the graph titles), and the GPU memory (*Dedicated GPU memory*) is the reliable sign |

`FLLM_STRICT_DEVICE=1` (or `fllm-rs serve --strict-device`) turns every one of these hand-overs into an error: the load stops with the reason
instead of running part of the model on the CPU. Use it in scripts that must never run slowly by accident. Sampling a token from the logits on
the host (a few microseconds) is by design and is listed but never an error.

## A big model on an integrated GPU: no token, nothing in the log

**Symptom:** `--device vulkan` on an integrated GPU (Intel Iris Xe / Arc graphics, an AMD APU) with a model that is as large as the RAM or larger:
the first token does not come for many minutes, the free memory of the machine drops to zero, and the log was silent.

**Why:** an integrated GPU has no memory of its own. Everything it holds is taken from the RAM that the model file is memory-mapped in, so a weight that
is copied to the GPU exists twice (the mapped file pages and the device copy) until the mapped pages are dropped, and a model larger than the RAM
cannot be resident at all. Earlier versions also dequantized the lattice formats (IQ1 / IQ2 / IQ3) and Q2_0 on the host while uploading them and
planned with the expanded size (a 75.8 GB file was planned as 104.9 GB); these formats stay packed on the device now and the plan counts the real size.

**What fLLM does now** (all devices that report a unified memory, nothing machine specific):

* The memory plan counts the weights against the free system memory less a reserve (1/32 of the RAM, at least 512 MiB), not against the figure the driver
  reports. When even the attention weights and one sequence do not fit, the load stops with an error that names the device and the memory instead of copying
  until the machine swaps. When the weights are larger than the RAM, the log says by how much (`the weights are N MiB larger than ...`): the layers that
  stay in the file are then read from the disk again and again.
* After a weight has been copied to the device its mapped file pages are dropped (Linux `MADV_PAGEOUT`; Windows: they leave the working set; macOS: no
  effect), so the file and the copy are not both resident. `FLLM_KEEP_HOST_PAGES=1` turns that off.
* The load prints a line every five seconds: `weights on the device: layer 12/48, 21.40 GiB copied (310 MiB/s), 20.90 GiB of mapped pages released, 1840 MiB
  of host memory free, 69 s`, or `copying blk.7.ffn_up_exps.weight` while one large weight is on its way. A load that stands still shows which weight it stands on.
* A GPU submission that does not finish is no longer silent: after 20 seconds the log says `<device>: still waiting for GPU submission N after 20 s (... MiB
  of host memory free)` every 20 seconds, and after 180 seconds the request fails with `Vulkan device '<name>' did not finish GPU submission N within 180 s`.
  `FLLM_VK_WAIT_TIMEOUT_S=N` sets the limit (`0`: wait forever).
* `FLLM_VK_TRACE=1` logs every uploaded weight and every compiled shader with its time in milliseconds; without it, anything slower than three seconds
  (`FLLM_VK_SLOW_MS`) is logged. A first token that is slow because the driver compiles many shaders (one per format and shape) shows up there.

**What to do:** a smaller quantization or a shorter `--ctx`, `--gpu-layers N` to put fewer layers on the GPU, or `--device cpu`. If the log stops at
`copying <weight>` or at `still waiting for GPU submission`, send the log with `FLLM_VK_TRACE=1` and the output of `fllm-rs devices`.

## Docker / GPU host problems

| Symptom | Cause / fix |
| --- | --- |
| `could not select device driver "" with capabilities: [[gpu]]` | NVIDIA Container Toolkit missing (Linux) or Docker Desktop without WSL 2 backend / driver (Windows). |
| `docker run --gpus all ubuntu nvidia-smi` fails on Windows | Update the Windows NVIDIA driver, run `wsl --update`, restart Docker Desktop. |
| `fllm-rs devices` shows no CUDA device | Add `--gpus all`; check `nvidia-smi` on the host. |
| Runs but slowly | GPU not used or model does not fit VRAM (log shows layers offloaded); compare `--device cpu` vs `--device cuda`. |
| Container OOM | Smaller quantization, smaller `--ctx`, `--kv-cache-type q8_0`, `--gpu-memory-utilization 0.8`. |

Check what the container sees: `docker run --rm --gpus all <registry>/fllm:cuda devices`.

## Requests and HTTP errors

- `prompt + max_tokens` must fit `--ctx`; the prompt must encode to at least
  one token and every token id must be inside the vocabulary — otherwise `400`.
- `401 {"error":{…}}`: missing/wrong `Authorization: Bearer <key>` (Anthropic
  clients: `x-api-key`). Routes `/`, `/health`, `/metrics`, `/api/tags`,
  `/api/version`, `/get_model_info`, `/get_server_info` stay open.
- `404` on embeddings: the `model` field does not match the served model name
  (`--model-name` or the file stem).
- `429 inference request limit reached`: admission limit (128 concurrent
  requests) exhausted — retry later, raise client backoff, or lower fan-out.
- `503`: model worker unavailable (still starting, or a failed load).
- Bodies are capped at 16 MiB (2 MiB on the encoder server).
- `n: 1..16`; `n > 1` currently needs explicit CPU compute on a stateless
  model without draft/MTP and unconstrained text — GPU/stateful setups answer
  a validation error.
- Generation endpoints do not exist on the BERT encoder server (only
  `/health`, `/metrics`, `/v1/models`, `/v1/embeddings`, `/api/embed`,
  `/api/embeddings`, `/tokenize`, `/detokenize`, rerank/score/classify/pooling).
- Ollama model management (`/api/pull`, `/api/delete`, `/api/copy`) always
  answers `501` — fLLM serves one model per process. `/infill` answers `501`
  without FIM tokens in the GGUF.

## Models, downloads and files

- `GGUF selection is ambiguous`: a repository holds many files — add the
  quant suffix (`owner/repo:Q4_K_M`) or link the exact file.
- `directory '…' does not contain any .gguf models`: point `--model` at a GGUF
  file, or `convert` the checkpoint first.
- `unknown model_type` at convert: the architecture is not recognized —
  conversion refuses to guess (see [architectures.md](../architectures.md)).
- Download failures: connectivity, wrong repository/quant name, or a gated
  repository needing `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`. `--revision`,
  `--local-files-only`/`--offline` restrict Hub resolution to cache/revision.
- `… already exists` from `quantize`/`convert`/`merge-lora`/`gguf-split`:
  existing outputs are refused — pass `--overwrite` (where supported) or
  remove the file.
- Slow first token on big prompts with a quantized KV cache is expected when
  the generic attention kernel runs; the FP16 scratch path (default) is much
  faster (see [gpu-backends.md](../gpu-backends.md#kv-cache-formats)).

## Wrong output shape / sampling surprises

- The server default is greedy (`temperature 0.0`) unless `--temp` or the
  request says otherwise — silent-looking determinism is usually this.
- Tool calls not parsed: check `--tool-parser` (default `auto` picks from chat
  template/architecture); force one (e.g. `--tool-parser hermes`) to test.
  Reasoning leaking into text: same for `--reasoning-parser`.
- Quantized KV caches below 8 bits (and `fp8`) lose accuracy on models with
  large key outliers; keep keys at `q8_0` (`--cache-type-k q8_0`). `nvfp4`
  cache is experimental.
- `FLLM_CPU_ISA` caps the CPU kernel tier (`scalar`, `avx2`, `avx-vnni`, `avx512bw`,
  `avx512-vnni`, `avx512-bf16`, `amx`; the last two exist only in builds with the `avx512`
  feature) to isolate a misbehaving SIMD path; `fllm-rs selftest-cpu` checks the tiers up to the
  cap against scalar references, `fllm-rs selftest-cpu --tiers all` every supported tier.
- Two CPU tiers (or two thread counts, or `FLLM_NO_REPACK=1`) can give different logprobs in the
  second decimal and a different greedy text where the model is undecided between two answers.
  That is the 8-bit activation arithmetic amplifying float rounding differences, not a defect;
  see "Accuracy" in [cpu-kernels.md](../cpu-kernels.md).

- CUDA decode uses CUDA graphs when the model allows it (the startup log says
  `CUDA graphs for decode steps: on`). If you suspect them for odd output or a
  failing capture, start with `FLLM_CUDA_GRAPH=0`: same text, slower; please
  report the difference.

## Getting help from the binary

```sh
fllm-rs --help            # usage summary (note: --max-batched-tokens 1024 there is stale; code default is 512)
fllm-rs devices           # compiled backends, detected devices, CPU ISA
fllm-rs selftest-cpu      # CPU kernels vs scalar references
fllm-rs selftest-gpu --device cuda    # every GPU op vs CPU reference (CUDA builds)
fllm-rs doctor --gpu      # per graphics card: does it do the work, or does the CPU? (one line each)
fllm-rs <command> --help  # per-command usage
```
