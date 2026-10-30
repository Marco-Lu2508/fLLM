# The NPU of an AMD Ryzen AI processor

Ryzen AI 300 / 400 and AI Max processors have an NPU (XDNA 2). There are two ways to use it with fLLM:

* **FastFlowLM** (this page, below): a separate program with AMD's own kernels that runs whole models; fLLM drives it.
* **fLLM's own kernels** (experimental, Linux, one machine tested; [../npu-xdna.md](../npu-xdna.md)): fLLM's matrix program for the NPU, written and built from open
  tools and run through XRT. It computes the long-prompt part of any GGUF model that fLLM runs on the processor (`FLLM_NPU_HOST=1`), keeps fLLM's
  features and model files, needs no FastFlowLM, and is faster than the processor (Llama-3.2-3B, 1180-token prompt: 169-180 against 131-136 tok/s) but
  slower than FastFlowLM (306). It needs XRT's development files once to build a small helper library; the Studio card says whether it is ready.

The rest of this page is about FastFlowLM. fLLM has no kernels for it in that route; [FastFlowLM](https://github.com/FastFlowLM/FastFlowLM)
does: it runs Llama, Qwen, Gemma, gpt-oss and other models on the NPU and serves them over an OpenAI-compatible interface. fLLM finds it, downloads
models, starts it for one model and lists that model as the provider **AMD NPU (FastFlowLM)**, so the NPU is one more place a chat can run. The NPU
kernels of FastFlowLM are AMD's and may be used and redistributed unmodified; nothing of it is bundled with fLLM, you install it yourself.

What you get: on a Ryzen AI 5 PRO 435 (Radeon 840M, Linux), Llama-3.2-3B ran at 31 tok/s decode and 306 tok/s prompt on the NPU, against 28 / 213 on the
GPU with fLLM's native Vulkan backend and 17 / 146 on the processor ([benchmarks](../benchmarks.md)). The NPU also leaves the processor and the GPU free.
What you do not get: fLLM's own features for a model on the NPU (grammars, speculative decoding, its cache and scheduler); the model is FastFlowLM's, with
its own files and its own quantization, and one model at a time.

## Set up

| | Windows | Linux |
|---|---|---|
| NPU driver | the current AMD Ryzen AI driver (NPU driver 32.0.203.x or newer) | the `amdxdna` driver of the kernel (Linux 6.14 or newer; Ubuntu 25.10 and 26.04 have it) |
| FastFlowLM | the installer `flm-setup.msi` from the [release page](https://github.com/FastFlowLM/FastFlowLM/releases) | the `.deb` for your distribution from the same page |
| Extra | nothing | the memlock limit of your user must be `unlimited`: a line `<user> - memlock unlimited` in `/etc/security/limits.d/`, then log in again |

`flm validate` checks all of it; fLLM shows the result.

## In the Studio

Settings > Hardware shows a card **AMD NPU (Ryzen AI): native kernels and FastFlowLM** when an AMD NPU or `flm` is found. Its first line says whether
fLLM's own kernels are usable ("Native fLLM kernels: ready", or "not available (the reason)": no helper library, no XRT, no device). The rest of the
card is FastFlowLM's. It shows what `flm validate` reported
(a missing memlock limit, an old driver ...), the models of FastFlowLM (the downloaded ones first), and the buttons **Download**, **Run on the NPU** and
**Stop**. The first start of a model can take minutes. While a model runs it is listed under Settings > Model providers and in the model list of the
chat as "AMD NPU (FastFlowLM) · <model>"; stopping it removes the entry again. These routes work from the computer the server runs on only.

## On the command line

```sh
fllm-rs amd-npu status              # the NPU, FastFlowLM, its check of the NPU stack, the models, as JSON
fllm-rs amd-npu list                # [x] = downloaded
fllm-rs amd-npu pull llama3.2:3b    # download a model
fllm-rs amd-npu serve llama3.2:3b   # run it until Ctrl-C and print the address of its OpenAI-compatible interface
fllm-rs amd-npu serve llama3.2:3b --register   # the same, and list it in the Studio as a provider while it runs
```

`FLLM_FLM_PATH` names the `flm` program when it is not on the `PATH` (fLLM also looks where the installers put it).

For fLLM's own kernels (details in [../npu-xdna.md](../npu-xdna.md)):

```sh
fllm-rs amd-npu build-shim                     # once: the helper library over XRT (needs a C++ compiler and XRT's development files)
fllm-rs devices                                 # lists "AMD XDNA NPU" as npu:0 when the helper, XRT and the device are there
FLLM_NPU_HOST=1 fllm-rs serve --model m.gguf    # long prompts of any model run on the NPU
fllm-rs amd-npu native-bench 8192 3072 512 q4_k # one projection on the NPU against the processor
```

## Limits

* fLLM's own kernels: tested on the same one machine only (Linux); the NPU computes prompt projections of 64 or more rows in bf16, decode stays on the
  processor, and a model's weights are held in bf16 for it (memory!). Not tested on Windows; [../npu-xdna.md](../npu-xdna.md) writes down what is needed.

* Tested on one machine (a Ryzen AI 5 PRO 435 laptop with Ubuntu 26.04 and FastFlowLM 1.0.11): detection, `flm validate`, download, start, stop, the provider
  entry and a chat through the Studio. Not tested on Windows with an AMD NPU (the code paths are the same; the Windows build runs and reports "no NPU" on a
  PC without one).
* An Intel NPU is a different path (OpenVINO, experimental), see [intel-npu.md](intel-npu.md).
