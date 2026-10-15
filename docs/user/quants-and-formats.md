# Quants and formats

How fLLM writes quantized weights, what it can read, and how weight and
KV-cache formats affect serving. Source docs:
[quantization.md](../quantization.md),
[awq-gptq-marlin.md](../awq-gptq-marlin.md),
[rust-formats-and-import.md](../rust-formats-and-import.md),
[gpu-backends.md](../gpu-backends.md) (KV cache formats),
[configuration.md](../configuration.md) (cache flags).

## Writers: `quantize` and `convert --outtype`

```sh
fllm-rs quantize INPUT.gguf OUTPUT.gguf TYPE [--overwrite]
fllm-rs convert CHECKPOINT OUTPUT.gguf [--outtype preserve|f16|f32|TYPE] [--overwrite]
```

`TYPE` is one of (names match case-insensitively):

| Type | GGML id | Block | Notes |
| --- | --- | --- | --- |
| `q4_0` | 2 | 32 elems / 18 bytes | |
| `q4_1` | 3 | 32 / 20 | |
| `q4_k_s` | 12 (Q4_K) | 256 / 144 | smaller mix |
| `q4_k_m` | 12 (Q4_K) | 256 / 144 | alias: `q4_k` |
| `q5_0` | 6 | 32 / 22 | |
| `q5_1` | 7 | 32 / 24 | |
| `q5_k_s` | 13 (Q5_K) | 256 / 176 | smaller mix |
| `q5_k_m` | 13 (Q5_K) | 256 / 176 | alias: `q5_k` |
| `q6_k` | 14 (Q6_K) | 256 / 210 | |
| `q8_0` | 8 | 32 / 34 | |

An unknown name fails with
`unknown type <name>; use one of q4_0, q4_1, q5_0, q5_1, q8_0, q4_k_s, q4_k_m, q5_k_s, q5_k_m, q6_k`.
An existing output file is refused unless `--overwrite` is given. On success the
command reports `Quantized <n> of <m> tensors: <in> -> <out> bytes (<ratio>x smaller)`.

What gets quantized:

- Only **float tensors** are rewritten (F32/F16/BF16 rows are decoded and
  re-encoded). Already quantized tensors of any kind are copied byte for byte.
- A tensor must be a 2-D `.weight` matrix. Names containing `norm`,
  `ffn_gate_inp`, `rope_`, `ssm_` or `pos_embd`, names starting with `v.` and
  anything that is not a matrix stay unchanged — norms, routers and embeddings
  keep their source encoding.
- `output.weight` is written as `q6_k` for every preset except `q8_0`.
- The K-quants are mixes: for `q4_k_m`/`q5_k_m` some `attn_v` and `ffn_down`
  tensors (first and last eighth of the layers plus every third one) are
  upgraded to `q6_k`; for `q4_k_s` early `attn_v` layers (below 4) and the first
  eighth of `ffn_down` layers become `q5_k`.
- Rows whose length is not a multiple of the block size fall back
  `Q4_K → q5_0`, `Q5_K → q5_1`, `Q6_K → q8_0`; a tensor that still does not fit
  is left unchanged.
- Metadata is copied unchanged. The reader requires GGUF version 3.

`q4_0`, `q4_1`, `q5_0`, `q5_1` and `q8_0` come out byte-identical to
`llama-quantize`; the K-quants differ only in rare float-summation ties.
`scripts/compare_gguf.py` compares two files tensor by tensor.

`convert --outtype` values: `preserve` (default, keep checkpoint dtypes),
`f16`, `f32`, or one of the presets above (converts to F16 in a staging file
next to the output, quantizes it to the final path, then removes staging).
With a quant preset the output must not exist unless `--overwrite` is passed.

## Requantization rule

`quantize` only touches float tensors:

```sh
# works: float source -> quantized output
fllm-rs quantize model-f16.gguf model-q4_k_m.gguf q4_k_m

# no effect on already quantized tensors: copied unchanged
fllm-rs quantize model-q4_k_m.gguf model-q8_0.gguf q8_0
```

To change the quantization of a quantized file, start again from the F16/F32
source (or re-run `convert` with the desired `--outtype`). There is **no
writer** for IQ/TQ/FP4: the presets above are the only types fLLM can produce.

## Reading: `formats` and `inspect`

```sh
fllm-rs formats          # every registered GGML storage type + decoder class
fllm-rs inspect model.gguf   # tensor types with GGML ids and counts
```

The decoder reads F32/F16/BF16, the classic Q formats, all K-quants, the IQ
family (`iq1_s`, `iq1_m`, `iq2_*`, `iq3_*`, `iq4_nl`, `iq4_xs`), TQ and FP4 —
but registration is not a promise of inference support. Formats without a
packed kernel decode rows on demand (see [memory](memory.md) for what that
costs on each backend).

## AWQ / GPTQ checkpoints (Q4Z, Marlin)

`convert` keeps 4-bit AWQ (AutoAWQ GEMM) and GPTQ (AutoGPTQ, no act-order)
checkpoints 4-bit. Each weight matrix becomes a `Q4Z` tensor (GGUF ids
`0x8000_0010`/`0x8000_0011`/`0x8000_0012` for groups of 32/64/128); nothing is
re-quantized. GPTQ checkpoints with act-order (`desc_act = true`) become
`Q4ZP` tensors (ids `0x8000_0014`/`0x8000_0015`/`0x8000_0016`) carrying the
column permutation. Group sizes other than 32/64/128, fp32 scales and stacked
experts fall back to F16 expansion automatically.

Execution: on NVIDIA GPUs (Turing/sm_75 and newer, `cuda` backend) Q4Z/Q4ZP
matrices are repacked once at upload and run with vLLM's Marlin GEMM for
decode and prompts. Shapes must be multiples of 128 — anything else, older
GPUs and Vulkan dequantize instead. On x86-64 CPUs (AVX2 and up, runtime-selected) Q4Z/Q4ZP run on
own integer-dot kernels (AVX2, AVX-VNNI, AVX-512 BW/VNNI; about the speed of Q4_K_M of the
same size); other CPUs use the slow generic path. Details and measurements:
[awq-gptq-marlin.md](../awq-gptq-marlin.md).

## NVFP4 / MXFP4 notes

- ModelOpt NVFP4 safetensors checkpoints are converted to GGUF type 40 by `run`, `serve`, `pull` and `convert`.
  The converted files run on Blackwell (sm_12x) FP4
  tensor cores; elsewhere the weights expand to F16 at load (the CPU decodes
  them per row, slowly). On pre-Blackwell NVIDIA GPUs they can run through the
  Marlin FP4 path instead (see [awq-gptq-marlin.md](../awq-gptq-marlin.md)).
- gpt-oss MXFP4 checkpoints keep their native 4-bit storage on CUDA; other
  backends decode on demand.

## KV cache formats (memory vs accuracy)

`--kv-cache-type TYPE` (alias `--kv-cache-dtype`) sets both halves;
`--cache-type-k`/`--cache-type-v` (llama.cpp `-ctk`/`-ctv`) set keys and values
separately. `auto` is `f16`. Memory relative to FP16 and support:

| Format | Bits/value | KV memory vs FP16 | CPU | CUDA (sm_75+) | wgpu |
| --- | --- | --- | --- | --- | --- |
| `f16` | 16 | 100 % | yes | yes | yes |
| `bf16` | 16 | 100 % | yes | yes | no |
| `fp8_e4m3` (`fp8`) | 8.125 | 51 % | yes | yes | no |
| `fp8_e5m2` | 8 | 50 % | yes | yes | no |
| `int8` | 8.125 | 51 % | yes | yes | no |
| `q8_0` | 8.5 | 53 % | yes | yes | yes |
| `q5_0` | 5.5 | 34 % | yes | yes | no |
| `q5_1` | 6 | 38 % | yes | yes | no |
| `q4_0` | 4.5 | 28 % | yes | yes | no |
| `q4_1` | 5 | 31 % | yes | yes | no |
| `iq4_nl` | 4.5 | 28 % | yes | yes | no |
| `nvfp4` (experimental) | 4.5 | 28 % | yes | yes | no |
| `f32` | 32 | 200 % | yes | yes | no |

The wgpu backends (Vulkan/Metal/DX12) keep `f16` and `q8_0` and refuse the
others at start-up:

```text
the vulkan backend only supports the f16 and q8_0 KV cache (requested …); use --device cuda or --device cpu for the other formats
```

That startup guard matches the bare names `vulkan`/`dx12`/`metal` only: the
aliases `gpu`/`wgpu` and the `kind:N` forms bypass that exact check, so prefer
a bare name (or `f16`/`q8_0`) with those spellings.

Accuracy guidance (measured, see
[gpu-backends.md](../gpu-backends.md#kv-cache-formats)): keys are the
sensitive half — `--cache-type-k q8_0 --cache-type-v q4_0` keeps most of the
saving on outlier-heavy models. The server warns when keys use a format below
8 bits or `fp8`; `nvfp4` is always marked experimental. A split `K q8_0 / V
q4_0` cache uses ~66 % of FP16 memory.

## GGUF tooling

```sh
fllm-rs gguf-set KEY=VALUE ... IN.gguf OUT.gguf
fllm-rs gguf-split IN.gguf --max-size 4G OUT-PREFIX
fllm-rs gguf-merge FIRST-SHARD.gguf OUT.gguf
fllm-rs merge-lora BASE.gguf ADAPTER OUT.gguf [--scale S] [--outtype f16|f32|TYPE]
```

- `gguf-set`: values parse as JSON first, else string (quote `KEY=VALUE` when
  the value has spaces). Tensor payloads are copied unchanged. Merge shards
  first.
- `gguf-split`: cuts at tensor boundaries; `--max-size` takes `K`/`M`/`G`/`T`
  suffixes (powers of 1024, metadata and padding count). A single tensor over
  the cap is refused; existing outputs are refused; partial shards are removed
  on failure.
- `gguf-merge`: takes the first shard (`…-00001-of-0000N.gguf`), validates
  order, writes one file, drops `split.*` metadata.
- `merge-lora`: `ADAPTER` is a PEFT directory (with `adapter_config.json`) or
  a `.safetensors` file beside it. Defaults `--scale 1`, `--outtype f16`.
  Existing outputs are refused.
