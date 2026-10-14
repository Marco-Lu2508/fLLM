# Quickstart: fLLM in Docker on an NVIDIA GPU (Windows and Linux)

One image, one command, terminal only. You paste a Hugging Face link, the model is downloaded on the
first start and cached in a Docker volume, and you chat in the terminal.

The GPU image (`docker/Dockerfile.cuda`) contains the `fllm-rs` binary with the CUDA backend and nothing
else: no CUDA toolkit, no Python. CUDA kernels are compiled for Pascal (GTX 10xx) up to Blackwell (RTX 50xx).

## 1. Prerequisites

You need a recent NVIDIA driver and Docker with GPU access. Test it first:

```sh
docker run --rm --gpus all ubuntu nvidia-smi
```

If this prints your GPU, skip to section 2.

### Windows 10/11

1. Install the current NVIDIA driver for Windows (version 570 or newer recommended; 535 is the lowest
   version Docker's GPU support is documented for). Do **not** install a Linux driver inside WSL.
2. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and keep
   **Use the WSL 2 based engine** enabled (Settings -> General). WSL 2 itself: `wsl --install` in an
   administrator PowerShell, then reboot.
3. Open PowerShell and run the test command above.

All commands below work unchanged in PowerShell, cmd and Windows Terminal (they are written on one line).

### Linux

1. Install the NVIDIA driver (`nvidia-smi` must work on the host) and Docker Engine.
2. Install the NVIDIA Container Toolkit and register it with Docker (Debian/Ubuntu shown; other
   distributions: see NVIDIA's installation guide for the container toolkit):

```sh
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

3. Run the test command above.

## 2. Get the image

```sh
docker pull ghcr.io/marco-lu2508/fllm:cuda
```

(The image is public on the GitHub Container Registry.) Or build it yourself from a checkout; the
first build takes a while because the CUDA kernels are compiled for all GPU generations:

```sh
git clone https://github.com/Marco-Lu2508/fLLM && cd fLLM
docker build -f docker/Dockerfile.cuda -t ghcr.io/marco-lu2508/fllm:cuda .
```

Only your own GPU: add `--build-arg CUDA_ARCHS=86` (Ampere consumer), `89` (Ada), `120` (RTX 50xx),
`75` (Turing), `61` (Pascal); this shortens the build considerably. The commands below use the short
name `fllm`; either tag the image (`docker tag ghcr.io/marco-lu2508/fllm:cuda fllm`) or write the full name.

## 3. Chat with a model

```sh
docker run -it --rm --gpus all -v fllm:/home/fllm/.cache fllm run unsloth/Qwen3.5-9B-GGUF:Q4_K_M --device cuda
```

- `run` accepts what you copy from the browser: `https://huggingface.co/owner/repo`, a link to one
  `.gguf` file, `hf.co/owner/repo:Q4_K_M`, or `owner/repo:Q4_K_M`.
- The first start downloads the model (several GB) into the Docker volume `fllm`; later starts load it
  from there without downloading.
- `--device cuda` makes the GPU mandatory: without a visible GPU (missing `--gpus all`) you get an error
  instead of a silent, slow CPU run. (`--device auto`, the default, also uses the GPU: it takes the best GPU the
  container sees and says so in its startup line; without one it runs on the CPU.)
- Quit the chat with Ctrl-D.
- `run` opens its own server inside the container; no `-p` is needed for chatting.

## 4. Use it as an API server

```sh
docker run -d --name fllm --gpus all -v fllm:/home/fllm/.cache -p 8000:8000 fllm serve --model unsloth/Qwen3.5-9B-GGUF:Q4_K_M --device cuda --host 0.0.0.0 --port 8000 --ctx 8192
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" -d "{\"prompt\":\"The capital of France is\",\"max_tokens\":8,\"temperature\":0}"
docker logs -f fllm
docker stop fllm && docker rm fllm
```

The server speaks the OpenAI API (`/v1/chat/completions`, `/v1/completions`, `/v1/models`); see
[server-api.md](server-api.md). Chat against it from another terminal with
`docker run -it --rm --network host fllm chat --url http://127.0.0.1:8000` (Linux), or use any OpenAI client.
`-p 8000:8000` publishes the port on all host interfaces; use `-p 127.0.0.1:8000:8000` to keep it local,
and add `--api-key SECRET` when other machines can reach it.

With Docker Compose ([`compose.cuda.yml`](../compose.cuda.yml) in the repository root):

```sh
FLLM_MODEL=unsloth/Qwen3.5-9B-GGUF:Q4_K_M docker compose -f compose.cuda.yml up -d
```

PowerShell: `$env:FLLM_MODEL="unsloth/Qwen3.5-9B-GGUF:Q4_K_M"; docker compose -f compose.cuda.yml up -d`.

## 5. Quantization and context size

- **Quantization**: the part after the colon selects the file in a GGUF repository, e.g.
  `owner/repo:Q4_K_M`, `:Q5_K_M`, `:Q8_0`. Smaller numbers are smaller and a bit less accurate. `Q4_K_M` is a
  good default. Repositories with many files need the suffix: without it fLLM stops with "GGUF selection is
  ambiguous" and lists the candidates. A link to one file (`.../blob/main/model-Q4_K_M.gguf`) pins that exact file.
- **Context size**: `--ctx 8192` sets the maximum tokens per request (prompt + answer). Model weights and
  KV cache must fit into GPU memory together. If the cache does not fit, the server prints a warning and
  reduces the context. `--kv-cache-type q8_0` or `fp8_e4m3` roughly doubles the context that fits (see docs/gpu-backends.md for all KV cache formats and their accuracy).
- Rule of thumb: model file size + KV cache + about 1 GB overhead <= VRAM. A 12 GB card runs a 9B model
  at Q4_K_M with several thousand tokens of context; use `--gpu-memory-utilization 0.85` to leave room
  for other programs (for example the Windows desktop).
- Further options (threads, batch sizes, speculative decoding): [configuration.md](configuration.md).

## 6. The model cache

Models live in the Docker volume `fllm`, mounted at `/home/fllm/.cache` in the container (Hugging Face
layout `huggingface/hub/models--owner--repo`).

```sh
docker run --rm -v fllm:/home/fllm/.cache fllm pull --list     # what is cached
docker volume inspect fllm                                      # where Docker stores it
docker volume rm fllm                                           # delete all cached models
```

On Linux the volume is under `/var/lib/docker/volumes/fllm/_data`. On Windows it is inside Docker
Desktop's WSL 2 disk (`docker-desktop` distribution), not in a normal folder. If you want the models in a
normal folder, mount one instead of the volume:
Linux `-v /srv/fllm-cache:/home/fllm/.cache` (the folder must be writable for uid 1000),
Windows `-v C:\fllm-cache:/home/fllm/.cache` (slower than a volume).
Private or gated repositories: add `-e HF_TOKEN=hf_...`.

## 7. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `could not select device driver "" with capabilities: [[gpu]]` | NVIDIA Container Toolkit missing (Linux) or Docker Desktop without WSL 2 backend / driver (Windows) |
| `docker run --gpus all ubuntu nvidia-smi` fails on Windows | update the Windows NVIDIA driver, run `wsl --update`, restart Docker Desktop |
| `fllm-rs devices` shows no CUDA device | add `--gpus all`; check `nvidia-smi` on the host |
| runs but slowly | the GPU is not used or the model does not fit into VRAM (the log shows `layers offloaded`); try `--device cpu` vs. `--device cuda` to compare |
| out of memory | smaller quantization, smaller `--ctx`, `--kv-cache-type q8_0`, `--gpu-memory-utilization 0.8` |
| download fails | connectivity, wrong repository / quantization name, gated repository needs `HF_TOKEN` |

Check which devices the container sees: `docker run --rm --gpus all fllm devices`.
