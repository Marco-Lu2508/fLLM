# Quantization and GGUF file tooling

Everything here comes from `rust/checkpoint/src/requant.rs`, `quantize.rs` and
`gguf_tools.rs`. Two commands produce quantized weights — `fllm-rs quantize` for an existing
GGUF and `fllm-rs convert --outtype <type>` for a checkpoint in one step — and four GGUF tools
handle metadata and file layout: `gguf-set`, `gguf-split`, `gguf-merge` and `merge-lora`.

```sh
fllm-rs quantize INPUT.gguf OUTPUT.gguf TYPE [--overwrite]
```

`TYPE` is one of:

| Type | GGML id | Block | Notes |
| --- | --- | --- | --- |
| `q4_0` | 2 | 32 elems / 18 bytes | |
| `q4_1` | 3 | 32 / 20 | |
| `q5_0` | 6 | 32 / 22 | |
| `q5_1` | 7 | 32 / 24 | |
| `q8_0` | 8 | 32 / 34 | |
| `q4_k_s` | 12 (Q4_K) | 256 / 144 | smaller mix |
| `q4_k_m` | 12 (Q4_K) | 256 / 144 | alias: `q4_k` |
| `q5_k_s` | 13 (Q5_K) | 256 / 176 | smaller mix |
| `q5_k_m` | 13 (Q5_K) | 256 / 176 | alias: `q5_k` |
| `q6_k` | 14 (Q6_K) | 256 / 210 | |

Type names are matched case-insensitively; an unknown name fails with
`unknown type <name>; use one of q4_0, q4_1, q5_0, q5_1, q8_0, q4_k_s, q4_k_m, q5_k_s, q5_k_m, q6_k`.
An existing output file is refused unless `--overwrite` is given. The command reports
`Quantized <n> of <m> tensors: <in> -> <out> bytes (<ratio>x smaller)`.

## What gets quantized

- Only **float tensors** are rewritten: F32, F16 and BF16 rows are decoded and re-encoded.
  Already quantized tensors (any Q/IQ/TQ/FP4 type) are copied byte for byte — re-quantizing
  from one quant type to another therefore needs the float source file (see
  [Requantization](#requantization)).
- A tensor must be a 2-D `.weight` matrix. Names containing `norm`, `ffn_gate_inp`, `rope_`,
  `ssm_` or `pos_embd`, names starting with `v.` and everything else that is not a matrix are
  kept unchanged — norms, routers and embeddings stay as they are.
- `output.weight` is written as `q6_k` for every preset except `q8_0`.
- The K-quants are mixes, not single types. For `q4_k_m`/`q5_k_m`, some `attn_v` and
  `ffn_down` tensors (the first and last eighth of the layers plus every third one) are
  upgraded to `q6_k`. For `q4_k_s`, early `attn_v` layers (below 4) and the first eighth of
  `ffn_down` layers become `q5_k`.
- If a row length is not a multiple of the block size the type falls back
  `Q4_K → q5_0`, `Q5_K → q5_1`, `Q6_K → q8_0`; if it still does not fit, the tensor is left
  unchanged.
- GGUF metadata is copied unchanged; only tensor infos and payloads are rewritten. The reader
  requires GGUF version 3.

The repository's verification notes state that `q4_0`, `q4_1`, `q5_0`, `q5_1` and `q8_0` come
out byte-identical to `llama-quantize` while the K-quants differ only in rare float-summation
ties ([architectures.md](architectures.md#quantization));
`scripts/compare_gguf.py` compares two files tensor by tensor.

## `convert --outtype`

```sh
fllm-rs convert CHECKPOINT OUTPUT.gguf [--outtype preserve|f16|f32|TYPE] [--overwrite]
```

| Value | Effect |
| --- | --- |
| `preserve` (default) | Keep the checkpoint dtypes (F16/BF16/F32 as stored) |
| `f16` | Convert all weights to F16 |
| `f32` | Convert all weights to F32 |
| one of the quant types above | Convert to F16 into a staging file next to the output (`model.gguf` → `model.f16.tmp`), quantize it to the final path, then remove the staging file |

With a quant preset the output file must not exist unless `--overwrite` is passed. Conversion
and quantization can also be run separately: `convert --outtype f16` followed by
`quantize …q4_k_m`.

## Requantization

`fllm-rs quantize` only touches float tensors, so:

```sh
# works: float source -> quantized output
fllm-rs quantize model-f16.gguf model-q4_k_m.gguf q4_k_m

# no effect on the already quantized tensors: they are copied unchanged
fllm-rs quantize model-q4_k_m.gguf model-q8_0.gguf q8_0
```

To change the quantization of a quantized file, re-quantize from the F16/F32 source (or
re-run `convert` with the desired `--outtype`).

## Reading quantized files

`fllm-rs formats` prints every registered GGML storage type with its block size, byte width
and decoder class (35 valid ids; removed ids are rejected explicitly). The decoder reads
F32/F16/BF16, the classic Q formats, all K-quants, the IQ family (`iq1_s`, `iq1_m`, `iq2_*`,
`iq3_*`, `iq4_nl`, `iq4_xs`), TQ and FP4 — but there is **no writer** for IQ/TQ/FP4: the
`quantize` presets above are the only types fLLM can produce.

```sh
fllm-rs formats
fllm-rs inspect model.gguf     # tensor types with GGML ids and counts
```

## GGUF metadata: `gguf-set`

```sh
fllm-rs gguf-set KEY=VALUE ... IN.gguf OUT.gguf
```

Rewrites metadata key/value pairs into a new file. Values are strings, numbers or booleans —
the text after `=` is parsed as JSON first and otherwise treated as a string, so quote the
whole `KEY=VALUE` argument when the value contains spaces. Tensor payloads are copied
unchanged. Split files must be merged first; empty keys, keys starting with `GGUF.` or
`split.` and duplicate keys are rejected.

```sh
fllm-rs gguf-set 'general.name=my model' model.gguf model-renamed.gguf
```

## Splitting and merging shards

```sh
fllm-rs gguf-split IN.gguf --max-size 4G OUT-PREFIX
fllm-rs gguf-merge FIRST-SHARD.gguf OUT.gguf
```

- `gguf-split` cuts at tensor boundaries. `--max-size` takes a positive integer with an
  optional `K`, `M`, `G` or `T` suffix (powers of 1024); the cap counts metadata and padding
  as well, and a single tensor that exceeds it on its own is refused. Outputs are named
  `OUT-PREFIX-00001-of-0000N.gguf`, existing outputs are refused, and on failure the shards
  already written are removed again.
- `gguf-merge` takes the first shard of a consecutive set (`…-00001-of-0000N.gguf`), validates
  the order, writes one file and drops the `split.*` metadata. A file that is not a shard set
  is rejected.

```sh
fllm-rs gguf-split model-q8_0.gguf --max-size 2G /models/model-q8_0
fllm-rs gguf-merge /models/model-q8_0-00001-of-00003.gguf /models/model-q8_0.gguf
```

## LoRA merge output type

```sh
fllm-rs merge-lora BASE.gguf ADAPTER OUT.gguf [--scale S] [--outtype f16|f32|TYPE]
```

`ADAPTER` is a PEFT directory (with `adapter_config.json`) or a `.safetensors` file beside it.
Defaults: `--scale 1`, `--outtype f16`; `--outtype` also accepts the quantization presets from
the table above. Existing outputs are refused. Details of load-time adapters are in
[inference-features.md](inference-features.md).

## See also

- [rust-formats-and-import.md](rust-formats-and-import.md) — which storage types the decoder
  supports and how checkpoints are imported
- [architectures.md](architectures.md) — per-family notes and the verification commands
- [getting-started.md](getting-started.md) — convert → quantize → serve
# EXL2 / EXL3 checkpoint import

`fllm-rs convert MODEL_DIR MODEL.gguf --outtype f16` imports EXL2 mixed-width
linear weights and EXL3 integer-width trellis linear weights from mmap-backed
SafeTensors. `--outtype q4_k_m` (or another existing quantization preset) first
writes a temporary F16 GGUF and then uses the existing GGUF quantizer. This
requires disk space for both outputs. Config/tokenizer files remain required;
architecture recognition uses the normal conversion plan. The complete
quantization_config JSON is retained as `fllm.source.quantization_config`, including
bits, head_bits and calibration when supplied. Actual widths come from each
matrix's group or trellis tensors, not from a checkpoint's average bit count.

F16 conversion expands the weights and loses EXL's storage and bandwidth
advantages. Requantization introduces another lossy step and does not preserve
the original EXL codes. Import is not native EXL inference. EXL2 supports upstream
widths 2/3/4/5/6/8 and validated input permutations. EXL3 supports integer widths
1..8, default/mcg/mul1 codebooks and 128-block Hadamard transforms. Fractional
EXL3, deprecated separate scale tensors and non-linear n-gram rings are rejected.
Fused and transposed architecture-specific projections need a separate streaming
mapping and are outside this import path. See [exl-formats.md](exl-formats.md) for
the source revisions, arithmetic details and unresolved numerical equivalence.
