# Docker

fLLM ships two image definitions ([`docker/Dockerfile`](../docker/Dockerfile): CPU + Vulkan, and
[`docker/Dockerfile.cuda`](../docker/Dockerfile.cuda): CUDA, see below) and two compose files
([`compose.yml`](../compose.yml): CPU with a model directory, [`compose.cuda.yml`](../compose.cuda.yml): NVIDIA GPU
with a Hugging Face link). All run the same `fllm-rs` binary; the runtime images contain no Python.

## Image

```sh
docker build -f docker/Dockerfile -t fllm .
# or, with limited build parallelism on a shared host:
make build-image JOBS=2
```

`make build-image` tags `ghcr.io/marco-lu2508/fllm:<short-sha>` and `:latest`
(`REGISTRY`/`IMAGE`/`TAG` are Makefile variables); `make push-image` pushes both tags.

The Dockerfile is a two-stage build: a `rust:1-slim-bookworm` builder runs

```sh
cargo build --release --locked --features "rust-gpu" -p fllm-engine --bin fllm-rs
```

and the `debian:bookworm-slim` runtime stage installs `libvulkan1`, `curl` and
`ca-certificates`, copies `/usr/local/bin/fllm-rs`, creates an unprivileged `fllm` user and
ends with:

```
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD curl -fsS http://127.0.0.1:${FLLM_PORT:-8000}/health || exit 1
ENTRYPOINT ["fllm-rs"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
```

Because the entrypoint is `fllm-rs`, any argument after the image name is a CLI subcommand or
`serve` flag.

## CUDA image (NVIDIA GPUs)

[`docker/Dockerfile.cuda`](../docker/Dockerfile.cuda) builds the same binary with `--features "rust-gpu,cuda"`
(the wgpu paths and the CUDA backend are combinable). Short copy-and-paste instructions for Windows and Linux
are in [quickstart-windows-linux.md](quickstart-windows-linux.md); this section describes the image.

```sh
docker build -f docker/Dockerfile.cuda -t fllm:cuda .
# only one GPU generation, much faster build (61 Pascal, 70 Volta, 75 Turing, 80/86 Ampere, 89 Ada, 90 Hopper, 120 Blackwell):
docker build -f docker/Dockerfile.cuda --build-arg CUDA_ARCHS=120 -t fllm:cuda .
make build-image-cuda JOBS=3        # tags ghcr.io/marco-lu2508/fllm:cuda-<short-sha> and :cuda
```

- **Builder**: `nvidia/cuda:12.8.1-devel-ubuntu24.04` plus rustup. CUDA 12.8 compiles for Pascal (sm_61) up to
  Blackwell (sm_120); CUDA 13 dropped Pascal and Volta. The newest architecture is also embedded as PTX.
  CUTLASS (v4.8.0) and FlashInfer (v0.6.5) headers are cloned from GitHub into `third_party/` during the build
  (see `THIRD_PARTY.md`), so the image build needs internet access. Without those directories `build.rs` would
  silently build slower prefill kernels; the Dockerfile therefore fails if the clones do not deliver the headers.
- **Layer cache**: crates.io dependencies are compiled against stub sources first; that layer is reused until
  `Cargo.toml` / `Cargo.lock` change. A change in `rust/` recompiles all CUDA kernels (the longest step).
- **Runtime**: `nvidia/cuda:12.8.1-base-ubuntu24.04` + `curl` + `ca-certificates` + the stripped binary.
  `cudart` is linked statically; `libcuda.so.1` is loaded from the host driver, which the NVIDIA Container
  Toolkit (or Docker Desktop with WSL 2) injects with `--gpus all`. `NVIDIA_VISIBLE_DEVICES=all` and
  `NVIDIA_DRIVER_CAPABILITIES=compute,utility` are preset. No Vulkan loader is installed: inside the container the
  GPU path is CUDA.
- **User and cache**: the process runs as user `fllm` (uid 1000). The Hugging Face cache is
  `HF_HOME=/home/fllm/.cache/huggingface`, so mount a named volume at `/home/fllm/.cache`
  (`-v fllm:/home/fllm/.cache`). `fllm-rs run <link>` and `serve --model owner/repo[:QUANT]` download there on the
  first start and reuse the cache afterwards. `FLLM_CACHE` / `HF_HUB_CACHE` override the location.
- **Device**: the examples pass `--device cuda`, which fails with an error when no GPU is visible (for example when
  `--gpus all` is missing) instead of silently running on the CPU. `--device auto` (the default) also ends up on the
  CUDA device in this image (measured), although its startup line still says "selects CPU".

```sh
docker run --rm --gpus all fllm:cuda devices
docker run -it --rm --gpus all -v fllm:/home/fllm/.cache fllm:cuda run unsloth/Qwen3.5-9B-GGUF:Q4_K_M --device cuda
docker run -d --gpus all -v fllm:/home/fllm/.cache -p 8000:8000 fllm:cuda \
  serve --model unsloth/Qwen3.5-9B-GGUF:Q4_K_M --device cuda --host 0.0.0.0 --port 8000 --ctx 8192
```

`run` starts its own server on `127.0.0.1` inside the container and chats with it, so it needs `-it` and no
`-p`. Only `serve` needs `-p 8000:8000`.

### Compose for the GPU

[`compose.cuda.yml`](../compose.cuda.yml) reserves all NVIDIA GPUs
(`deploy.resources.reservations.devices`, driver `nvidia`, `count: all`, `capabilities: [gpu]`), keeps the
model cache in the named volume `fllm-cache` and passes the model as a Hugging Face link:

```sh
FLLM_MODEL=unsloth/Qwen3.5-9B-GGUF:Q4_K_M docker compose -f compose.cuda.yml up -d
```

| Variable | Default | Effect |
| --- | --- | --- |
| `FLLM_MODEL` | required | Hugging Face link or `owner/repo[:QUANT]` |
| `FLLM_PORT` | `8000` | host port |
| `FLLM_BIND_HOST` | `127.0.0.1` | host address of the port mapping |
| `FLLM_CTX` | `8192` | `--ctx` |
| `FLLM_KV_CACHE` | `f16` | `--kv-cache-type` (any type of [KV cache formats](gpu-backends.md#kv-cache-formats)) |
| `FLLM_API_KEY` | empty | key the server requires |
| `HF_TOKEN` | empty | token for gated or private repositories |
| `FLLM_EXTRA_ARGS` | empty | appended to the `serve` command |
| `FLLM_IMAGE` | `ghcr.io/marco-lu2508/fllm:cuda` | image to run |

### Host requirements

NVIDIA driver on the host (Windows: the normal Windows driver, nothing inside WSL), NVIDIA Container Toolkit on
Linux, Docker Desktop with the WSL 2 backend on Windows. Blackwell GPUs need a driver that supports CUDA 12.8
(570 or newer). Tested here only with driver 595 on an RTX PRO 3000 Blackwell Laptop; older drivers
are untested.

### CI

The `docker-cuda.yml` workflow builds this image and pushes
`ghcr.io/marco-lu2508/fllm:cuda` plus a tag per build; see [ci.md](ci.md) for the triggers and tags. The build
needs no GPU on the runner.

## CPU-only compose run with a model directory

The shipped `compose.yml` is CPU-only: the command line it generates does not pass `--device`,
and `--device auto` selects the CPU. Models are mounted read-only at `/models`.

```sh
FLLM_MODELS_DIR=/path/to/models FLLM_MODEL=model.gguf docker compose up -d
curl -s http://127.0.0.1:8000/health
```

Equivalent `docker run`:

```sh
docker run --rm -p 127.0.0.1:8000:8000 \
  -v /path/to/models:/models:ro \
  ghcr.io/marco-lu2508/fllm:latest \
  serve --host 0.0.0.0 --port 8000 --model /models/model.gguf
```

(`make run-image MODEL=model.gguf` does the same with `$(HOME)/models`.)

## Compose variables

| Variable | Default | Effect |
| --- | --- | --- |
| `FLLM_IMAGE` | `ghcr.io/marco-lu2508/fllm:latest` | Image to run (the `build:` section still points at `docker/Dockerfile`) |
| `FLLM_MODELS_DIR` | `./models` | Host directory mounted at `/models:ro` |
| `FLLM_MODEL` | `model.gguf` | File name inside `/models` passed to `--model` |
| `FLLM_MODEL_NAME` | value of `FLLM_MODEL` | `--model-name`, i.e. the id clients see |
| `FLLM_BIND_HOST` | `127.0.0.1` | Host side of the port mapping |
| `FLLM_PORT` | `8000` | Host side port (the container always listens on 8000, so the health check inside the container keeps using 8000) |
| `FLLM_API_KEY` | empty | Passed as `FLLM_API_KEY` into the container; set it together with `FLLM_BIND_HOST=0.0.0.0` when the port is reachable from other hosts |
| `FLLM_MAX_MODEL_LEN` | `4096` | `--ctx` |
| `FLLM_MAX_SEQS` | `16` | `--max-seqs` |
| `FLLM_MAX_BATCHED_TOKENS` | `512` | `--max-batched-tokens` |
| `FLLM_SPECULATIVE_LOOKUP` | `0` | `--speculative-lookup` |
| `FLLM_EXTRA_ARGS` | empty | Appended verbatim to the `serve` command line |

The generated command is:

```sh
fllm-rs serve --host 0.0.0.0 --port 8000 --model /models/$FLLM_MODEL \
  --model-name $FLLM_MODEL_NAME --ctx $FLLM_MAX_MODEL_LEN \
  --max-seqs $FLLM_MAX_SEQS --max-batched-tokens $FLLM_MAX_BATCHED_TOKENS \
  --speculative-lookup $FLLM_SPECULATIVE_LOOKUP $FLLM_EXTRA_ARGS
```

## Tuning examples

```sh
# threads, CPU affinity mask and a quantized KV cache (see docs/configuration.md)
FLLM_MODELS_DIR=/models FLLM_MODEL=model.gguf \
FLLM_EXTRA_ARGS="--threads 10 -C 0xFF5 --kv-cache-type q8_0" \
docker compose up -d

# reachable on the LAN, protected by a key
FLLM_MODELS_DIR=/models FLLM_MODEL=model.gguf \
FLLM_BIND_HOST=0.0.0.0 FLLM_API_KEY=secret \
docker compose up -d

# larger context and a different served model id
FLLM_MODELS_DIR=/models FLLM_MODEL=model-q4_k_m.gguf FLLM_MAX_MODEL_LEN=16384 \
FLLM_MODEL_NAME=my-model docker compose up -d
```

## Notes

- The default port mapping is `127.0.0.1:8000`; other hosts reach the server only after
  `FLLM_BIND_HOST=0.0.0.0` (and should then set `FLLM_API_KEY`).
- The mount is read-only. `FLLM_MODEL` is a file name relative to `/models`; passing a
  directory works too — the server picks a `*q4_k_m*.gguf` file from it, otherwise the first
  `*.gguf`.
- GPU use is not wired up in `compose.yml` (use `compose.cuda.yml` for NVIDIA): there is no `--device` flag in the generated
  command and no device mapping for `/dev/dri`. The image does contain `libvulkan1`, so a
  Vulkan run needs both `FLLM_EXTRA_ARGS="--device vulkan"` and extra container device
  access that you add yourself.
- The image build is also what the `docker.yml` workflow runs (see [ci.md](ci.md)); release binaries are built by the
  `release.yml` workflow.
- The images contain no web browser. The Code agent's browser (Settings > Browser in the Studio, see
  [the browser](user/studio.md#the-browser)) needs a Chromium-family browser inside the container, so add one to your own image,
  for example `RUN apt-get update && apt-get install -y --no-install-recommends chromium && rm -rf /var/lib/apt/lists/*`, or set
  `FLLM_BROWSER` to its path. fLLM adds `--no-sandbox` to the browser when it runs as root, as it does in a container. Nothing
  else changes without it: the browser tools then say that no browser was found.
