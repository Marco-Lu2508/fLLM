# The NPU of an Intel Core Ultra processor

Core Ultra processors have an NPU ("Intel AI Boost"). fLLM can use it through the OpenVINO runtime, **to read long prompts faster**. It does not make the
model write its answer faster, and on the machine it was measured on it is not faster than the integrated graphics. This page says what to expect.

## What it does

The model runs on the processor as usual. The large matrix multiplications of a prompt of 64 tokens or more run on the NPU instead, with the weights converted to
8 bit (one scale per output row). Answers are written one token at a time; that stays on the processor. The NPU keeps its own converted copy of the weights, so a
model needs about its own size again in memory (fLLM limits this to 40 % of the free memory).

## Using it

1. Settings, Hardware, NPU card: *Install the runtime* (or `fllm-rs npu-setup`). The NPU driver comes with Windows Update or your distribution.
2. Load a model with *Run on: NPU* (Advanced), or start with `--device npu`, or set `FLLM_NPU_HOST=1` and run on the processor.
3. The first prompts: fLLM compiles about one NPU program per weight matrix (about a second each). It does this in the background and keeps reading prompts on the
   processor meanwhile, so nothing waits; the card shows "still compiling". The compiled programs are cached on disk (`%LOCALAPPDATA%\fllm\npu`, `~/.cache/fllm/npu`;
   `FLLM_NPU_CACHE_DIR` moves it, an empty value turns it off), so the next start has them within seconds (about 40 ms per program).

The NPU is never chosen by `--device auto`. If it cannot take a model, the model runs on the processor and the card says why.

## What to expect (Core Ultra 5 235U, Llama-3.2-1B Q4_K_M, Windows)

* 512-token prompt: about twice as fast as the processor alone, about as fast as the integrated graphics.
* 130-token prompt: 1.0 to 1.4 times as fast as the processor alone.
* Qwen2.5-0.5B: little gain (most of its matrices are small), and the integrated graphics are 2 to 3 times faster.
* Answer speed: unchanged.

The PC's own speed drifts by 40 to 100 % between runs, so these are ratios from alternated runs, not absolute guarantees; the table is in
`rust/engine/src/backend/npu/OPENVINO.md` section 0. Not measured: energy use, Lunar Lake and newer NPUs, Linux, models of 3B and more, prompts above 512 tokens.
Use the NPU when the integrated graphics are busy or missing, or to keep the processor free; otherwise `--device auto` (the graphics) is faster.

## Settings

| Variable | Meaning |
|---|---|
| `FLLM_NPU_HOST=1` | run on the processor and send prompt projections to the NPU (what `--device npu` does) |
| `FLLM_NPU=0` | no offload |
| `FLLM_NPU_WEIGHTS` | `int8` (default), `int4` (half the memory, compiles slowly) or `exact` (Q4_0/Q8_0 bit for bit, compiles very slowly) |
| `FLLM_NPU_MIN_BATCH`, `FLLM_NPU_CHUNK` | fewest prompt rows worth a call (64), rows per NPU call (256) |
| `FLLM_NPU_BUDGET_MB` | memory the NPU programs may hold (default: 40 % of the free memory at start) |
| `FLLM_NPU_STATS=N` | print where the time goes every N offloaded projections |
| `FLLM_NPU_HYBRID=1` / `--npu-mode all` | the old hybrid backend (slow; for tests and for running every product on the NPU) |

`fllm-rs npu-bench [--n 2048 --k 2048 --batches 128,256 --variants q4_0:i8,f16 --cache DIR]` times one matrix on the NPU against the processor.
