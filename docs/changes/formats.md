# Formats: chat templates, tokenizers, quantized checkpoints, parsers

Fragment of the `formats` work. Everything here is CPU-only and was verified on CPU; nothing was run on a GPU.

## 1. Chat templates

* `chat_template.jinja`, `chat_template.json` (`{"chat_template": ...}`), the `chat_template` entry of `tokenizer_config.json`
  (a string or a list of `{"name", "template"}`) and the files of `additional_chat_templates/` are all read by the converter,
  in the precedence `transformers` uses (jinja file, json file, `tokenizer_config.json`). The default template is stored as
  `tokenizer.chat_template`, every named template as `tokenizer.chat_template.<name>`.
* The server renders with the `tool_use` template (`tokenizer.chat_template.tool_use`) when the request carries tools and the
  model has one (Hermes-style repositories, Command-R), otherwise with the default template. The tool-parser autodetection
  looks at both templates.
* `rust/engine/src/chat_template.rs` replaces the ad-hoc Jinja environment of `server.rs`. It reproduces the environment of
  `transformers.apply_chat_template`: `trim_blocks` and `lstrip_blocks`, `{% break %}`/`{% continue %}` (minijinja feature
  `loop_controls`), Python spelling of `True`/`False`/`None` and of lists and dicts when they are printed, `x|trim` and `x|string`
  of containers, `len(undefined) == 0`, `None is iterable == false`, `tojson` with `indent`, `ensure_ascii`, `sort_keys`,
  `separators`, `raise_exception`, `strftime_now`, `{% generation %}` tags, and the Python string methods `swapcase`, `casefold`,
  `zfill`, `ljust`, `rjust`, `center`, `partition`, `rpartition`, `rsplit`, `removeprefix`, `removesuffix`, `index`, `rindex`,
  `istitle`, `expandtabs`, `isspace` on top of minijinja-contrib's `pycompat`.
* Verification: `scripts/chat_template_golden.py` renders templates with the reference environment (`transformers` 5.19 /
  jinja2) and `rust/engine/src/chat_template.rs` renders the same inputs; the comparison is exact.
  * 213 feature probes (`testdata/chat_templates/probes.json`): all but 24 identical to jinja2. The 24 are listed with the reason in
    `KNOWN_DIFFERENCES` (no tuple type, printf-style `%`, `|attr`, `round(precision, method)`, a few rarely used filters/globals
    such as `truncate`, `wordcount`, `urlencode`, `filesizeformat`, `xmlattr`, `striptags`, `forceescape`, `cycler`, `joiner`,
    `dict(list_of_pairs)`). Mutating methods (`list.append`, `dict.update`, ...) fail in `transformers`' immutable sandbox too.
  * 21 conversations (system, multi-turn, tools, parallel tool calls and results, image and text content parts, reasoning
    history, `enable_thinking`, `reasoning_effort`, documents, role-order errors) rendered with 17 real templates
    committed under `testdata/chat_templates/templates/`: Qwen2.5, Qwen3, Qwen3-Coder, Qwen3.5, Phi-3, Phi-4, DeepSeek-R1,
    DeepSeek-V3, gpt-oss (Harmony), GLM-4.5, Kimi-K2, SmolLM3, Granite 3.3, Granite 4.0, InternLM2.5, Mixtral, Mistral-7B-v0.3,
    plus four small re-creations in the style of Llama 3, Gemma, Command-R and a Hermes `tool_use` template (the real ones
    are under licences that do not allow committing them here; the re-creations are not copies).
  * The same test, run with `FLLM_CHAT_TEMPLATE_CORPUS=<dir>` (see below), also rendered the real Llama 3.1/3.2, Gemma 2/3,
    Hermes-3 (default and `tool_use`), GLM-4-9B and MiniCPM3 templates: 525 renderings, all identical to `transformers`.
    Templates not tested: Command-R/Command-R7B (gated repositories), Phi-3 vision.
* Template licences: the committed templates come from the Apache-2.0 (Qwen, SmolLM3, Granite, InternLM, Mixtral, Mistral,
  gpt-oss), MIT (Phi, DeepSeek, GLM-4.5) and modified-MIT (Kimi) repositories named above.

## 2. Tokenizers without `tokenizer.json`

`convert` (and `fllm-rs run hf://...`, which downloads the extra files) now builds the Hugging Face tokenizer JSON in memory
from:

| Source | File(s) | Result |
| --- | --- | --- |
| SentencePiece | `tokenizer.model`, `spiece.model`, `sentencepiece.bpe.model`, `sentencepiece.model` | Unigram or BPE (byte fallback, `unk`, control and user-defined symbols as added tokens), `Precompiled` charsmap normalizer (nmt_nfkc and friends), strip/collapse whitespace, dummy prefix, decoder with byte fallback |
| tiktoken | `tiktoken.model` (Kimi K2), `qwen.tiktoken` (Qwen1), `tokenizer.model` that holds `base64 rank` lines (GLM-4), any `*.tiktoken` | byte-level BPE, `ignore_merges`, pattern from a table of known families (kimi, qwen1, glm, cl100k, o200k) or from `pat_str` in `tokenizer_config.json`, a `pat_str.txt`, or the `pat_str` literal in the repository's tokenizer code |
| Mistral tekken | `tekken.json` (v3 and later) | byte-level BPE, special tokens first (`<unk>`, `<s>`, `[INST]` ... then `<SPECIAL_n>`), regular tokens shifted by `default_num_special_tokens` |
| Mistral native layout | `params.json` + `consolidated*.safetensors` (no `config.json`) | `params.json` becomes a `mistral`/`ministral3`/`mixtral` config, tensor names are mapped to the Hugging Face names, FP8 `qscale_weight` becomes `weight_scale`, Q/K rows keep the original interleaved rotary order, vision tensors are dropped |

Verification with the reference implementations (`scripts/verify_tokenizers.py`, 265 strings: unicode, emoji, ZWJ sequences, whitespace runs,
digits, code, special-token literals, seeded fuzz):

| Tokenizer | Reference | Result |
| --- | --- | --- |
| GLM-4-9B-chat (`tokenizer.model`, tiktoken) | `tiktoken` | 265/265 identical (encode and decode) |
| Kimi-K2-Instruct (`tiktoken.model`) | `tiktoken` | 265/265 |
| Qwen-7B-Chat (`qwen.tiktoken`) | `tiktoken` | 265/265 |
| Mistral-Nemo-2407, Mistral-Small-3.1 (`tekken.json`) | `mistral_common` | 303/303 (corpus plus the ids of 40 special tokens) |
| Llama (hf-internal-testing/llama-tokenizer) | `sentencepiece` | 264/264 |
| Mistral-7B-Instruct-v0.3 | `sentencepiece` | 263/263 |
| Gemma-2 (`tokenizer.model`) | `sentencepiece` | 263/264; the one difference is a literal U+2581 at the start of the input |
| T5-small (`spiece.model`) | `sentencepiece` | all ids identical; `decode` prints `<unk>` where SentencePiece prints ` ⁇ ` |
| XLM-R base (`sentencepiece.bpe.model`) | `sentencepiece` | 261/264: a 200-character run of one letter and a letter followed by four combining marks differ in the first piece; the `<unk>` decode difference as for T5 |

Committed fixtures (`rust/engine/testdata/tokenizers/`): five tiny SentencePiece models (unigram with nmt_nfkc charsmap, unigram with byte
fallback, BPE Llama style with user-defined symbols, BPE plain), a tiktoken rank file and a tekken file, each with the ids the
reference library produces for 184 strings; `rust/engine/src/tokenizer_convert_tests.rs` compares them.

Known deviations: a literal `U+2581` at the start of the text (SentencePiece treats it as a space), and SentencePiece's treatment of
text after a user-defined symbol or `<s>` written into the text: the dummy prefix is added for every segment between added tokens,
exactly as `transformers`' legacy Llama tokenizers and llama.cpp do. InternLM2's `tokenizer.model` cannot be loaded by the
`sentencepiece` package (NUL in a piece), so it was converted but not compared.

GGUF side (`tokenizer.rs`): a Unigram `T5` model with a `tokenizer.ggml.precompiled_charsmap` now applies the normalization table
(it was refused before); for the `llama` model name the table is ignored as llama.cpp does; an unknown `tokenizer.ggml.pre` prints a
loud warning. Three llama.cpp architecture names are mapped to the layout fLLM implements (`exaone` to `llama`, `ernie4_5-moe` to
`ernie4_5moe`, `phi2` to `phi`), see `ARCHITECTURE_ALIASES` in `model.rs`; only layouts whose tensor names equal the converter's own
were mapped. Not verified against a llama.cpp-written file (none available here).

## 3. Quantized checkpoint formats

Dequantizing paths expand to F16 (F32 with `--outtype f32`) and print a warning with the size; `convert --outtype q4_k_m` (or
`q8_0`, `q6_k`) then gives a compact file. Formats with a compact fLLM type keep it.

| `quant_method` / format | Status | What happens |
| --- | --- | --- |
| none | accepted | dense |
| `gptq`, 4-bit, group size > 0, `checkpoint_format: gptq` | accepted (before) | native `Q4Z`/`Q4ZP` rows |
| `gptq`, 2/3/8-bit, `gptq_v2`, `group_size: -1`, `desc_act` | accepted (new) | expanded to F16; checked against GPTQModel packing (`scripts/make_quant_fixtures.py`) |
| `gptq` with another bit width | rejected | `GPTQ with {bits} bits is not supported (2, 3, 4 and 8 are)` |
| `gptq` with `checkpoint_format` marlin/bitblas/ipex/... | rejected | `GPTQ checkpoint_format "marlin" is not supported (gptq and gptq_v2 are); Marlin/BitBLAS/IPEX repacked checkpoints need the original GPTQ files` |
| `awq` 4-bit GEMM with zero points | accepted (before) | native `Q4Z` |
| `awq` GEMV, GEMV_fast, `zero_point: false`, Marlin | rejected | `only AWQ 4-bit GEMM packing is supported` / `only explicitly declared AutoAWQ GEMM packing is supported` / `only AutoAWQ checkpoints with explicit zero points are supported` |
| `fp8`, `fbgemm_fp8` | accepted | per-channel `FP8M` or block-wise F16; `input_scale_ub`/`activation_scale` ignored |
| `mxfp4` (gpt-oss, `*_blocks`/`*_scales`) | accepted (was rejected by `quantization()`) | experts as GGUF `MXFP4` |
| `compressed-tensors` `mxfp4-pack-quantized` | accepted (new) | GGUF `MXFP4` (E8M0 block scale) |
| `compressed-tensors` `nvfp4-pack-quantized` | accepted (new) | GGUF `NVFP4` (type 40), per-tensor factors `1/weight_global_scale` and `1/input_global_scale` |
| `compressed-tensors` int4/int8 symmetric | accepted (before) | `Q4Z` / `Q8_0` |
| `compressed-tensors` asymmetric / actorder / sparse / mixed | rejected | messages from `compressed_tensors()`, unchanged |
| `modelopt` NVFP4, FP8 | accepted (before) | `NVFP4`, `FP8M` |
| `modelopt` other `quant_algo` | rejected | `unsupported ModelOpt quant_algo "X": only NVFP4 and FP8 checkpoints can be imported` |
| `bitsandbytes` 4-bit NF4/FP4, double quantization; 8-bit LLM.int8 | accepted (new) | expanded to F16, checked against bitsandbytes 0.50 |
| MLX affine (`quantization: {bits, group_size}`), 2 to 8 bits, per-module overrides | accepted (new) | expanded to F16, checked against `mlx` 0.32 (`mx.quantize`/`mx.dequantize`) |
| MLX `mode` other than affine | rejected | `MLX quantization mode "mxfp4" is not supported (only affine); convert the original model instead` |
| `exl2`, `exl3` | accepted (before) | native rows |
| `hqq`, `torchao`, `quark`, `eetq`, `bitnet`, `aqlm`, `quip#`, `spqr`, `vptq`, `higgs`, `fp_quant`, `auto-round` | rejected | `unsupported quantization_config quant_method "X" (supported: ...); <what to do instead>` |

All fixtures are produced by the real libraries (`bitsandbytes`, `compressed-tensors`, `gptqmodel`, `mlx`, `transformers`' gpt-oss
dequantizer) by `scripts/make_quant_fixtures.py` and `scripts/make_gptoss_fixture.py`; the Rust tests (`dequant.rs`, `dequant_bits.rs`)
compare the converter's values with the library's. The gpt-oss test converts a tiny checkpoint in the published layout
end to end and decodes the GGUF experts back to the `transformers` values (this failed with `unsupported quantization_config
quant_method "mxfp4"` before).

## 4. Parsers

Tool-call parsers added (ids and aliases of `--tool-parser`): `pythonic` (`toolace`), `llama4_pythonic` (`llama4`, with the
`<|python_start|>` tags), `olmo3` (`<function_calls>` with one call per line), `tool_code` (`gemma3`, a ```` ```tool_code ```` block),
`xlam` (bare JSON array, fenced, or after `[TOOL_CALLS]`), `internlm` (`<|action_start|><|plugin|>`), `granite20b`
(`<function_call>`), `seed_oss` (`<seed:tool_call>`), `step3` (`<｜tool_calls_begin｜>` with `steptml:invoke`). Aliases of existing
parsers: `hunyuan`, `hunyuan_a13b`, `jamba` (`<tool_calls>`), `ernie`, `ernie45`, `qwen3_omni`, `hermes-2-pro` (`<tool_call>`).
Template autodetection knows the markers of the new ones. Each has streaming (every prefix) and truncated/malformed tests with
outputs in the style of vLLM's parser tests.

Reasoning parsers added: `granite` (`Here is my thought process:` ... `Here is my response:`, autodetected from the template) and
`hunyuan` (`<think>` then `<answer>` tags, flag only).

## 5. Environment variables and flags

| Name | Default | Meaning |
| --- | --- | --- |
| `FLLM_TIKTOKEN_PATTERN` | unset | `kimi`, `qwen1`, `glm`, `cl100k` or `o200k`: split pattern for a tiktoken rank file whose family is not detected |
| `FLLM_CHAT_TEMPLATE_CORPUS` | unset | test only: directory with extra templates (`<name>.jinja`, `<name>@<template>.jinja`, `<name>.meta.json`, `<name>.golden.json`) that `chat_template::tests::external_corpus_matches_transformers` compares with `transformers` |
| `--tool-parser`, `--reasoning-parser` | `auto` | new values as listed in section 4 |

## 6. Third party

* Merge derivation for SentencePiece BPE and for tiktoken ranks, and the ModelOpt/compressed-tensors scale conventions follow the
  published algorithms of Hugging Face `transformers` (`convert_slow_tokenizer.py`, Apache-2.0) and `compressed-tensors`
  (Apache-2.0); no code was copied.
* Pattern strings of Kimi K2 (`tokenization_kimi.py`, modified MIT), GLM-4 (`tokenization_chatglm.py`, GLM-4 licence), Qwen
  (`tokenization_qwen.py`, Tongyi Qianwen licence) and `cl100k_base`/`o200k_base` (tiktoken, MIT) are reproduced as data.
* The special-token list of the tekken format follows `mistral_common` (Apache-2.0).
* The tool-call formats follow vLLM's parsers (Apache-2.0) as documentation; no code was copied.
* Test templates: see section 1.

## 7. Test levels

L0 compiles only, L2 text matches a CPU reference. Everything in this fragment is L2 against the named reference library on CPU; no
GPU was used, no model larger than a few hundred KB was converted end to end. Real tokenizer files (GLM-4, Kimi, Qwen, Mistral,
Gemma, Llama, T5, XLM-R) were compared in `scripts/verify_tokenizers.py` runs on the build host and are not committed.

## 8. Known limits and what was not done

See REPORT.md ("NOT done").
