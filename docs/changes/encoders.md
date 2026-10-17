# Encoder models: new families, sparse and multi-vector outputs, rerankers

This fragment covers `rust/engine/src/encoder*.rs` (CPU reference implementation of the encoder
server) and the encoder converters in `rust/checkpoint/src/`. All of it is CPU only (test level L2
against `transformers`, see below); nothing here has been run on a GPU.

## Model table

Every row was verified with a tiny random checkpoint built by `transformers` (or by the model's own
remote code) and compared with the Python reference: cosine >= 0.9999 and max error / max |ref| <= 1e-3
(measured: <= 1e-5 unless noted). "tiny" means the architecture and tensor names of the real model, with
small dimensions. No real checkpoint of the families marked "layout only" was downloaded.

| Family | HF ids | Status | Verification script |
|---|---|---|---|
| BERT / RoBERTa / ModernBERT (regression) | bert-*, BAAI/bge-*, answerdotai/ModernBERT-* | unchanged, re-verified (ModernBERT 1.5e-4..3.5e-4) | `verify_encoders_basic.py` |
| sentence-transformers pipelines: pooling `cls` `mean` `max` `weightedmean` `mean_sqrt_len_tokens` `lasttoken`, Dense, Normalize | any `modules.json` checkpoint | verified (tiny BERT) | `verify_st_pipeline.py` |
| GTE new (`NewModel`, rotary, NTK scaling, fused GLU) | Alibaba-NLP/gte-base-en-v1.5, gte-large-en-v1.5, gte-multilingual-base, Snowflake/snowflake-arctic-embed-m-v2.0 (accepted by `model_type: gte`, not verified) | verified (tiny, with and without NTK) | `verify_gte.py` |
| Jina embeddings v3: XLM-R + rotary + per-task LoRA (`retrieval.query`, `retrieval.passage`, `separation`, `classification`, `text-matching`) | jinaai/jina-embeddings-v3 | verified (tiny, 3 adapters incl. word-embedding LoRA) | `verify_jina.py` |
| Jina reranker v2 (XLM-R flash layout, absolute positions, cross-encoder head) | jinaai/jina-reranker-v2-base-multilingual | converter path only: tensor names and shapes checked against the published safetensors header, no numeric check | none |
| BGE-M3 dense + sparse lexical weights + ColBERT vectors | BAAI/bge-m3 (heads from `sparse_linear.pt`, `colbert_linear.pt`) | verified (tiny BERT body with the real head formulas) | `verify_heads.py` |
| SPLADE (MLM head, `log(1+relu)`, max pooling) | naver/splade-*, opensearch-neural-sparse (MLM variants) | verified (tiny `BertForMaskedLM`) | `verify_heads.py` |
| ColBERT v2 (`HF_ColBERT`, markers, query augmentation, punctuation skip list) | colbert-ir/colbertv2.0 | verified against a reference written from the ColBERT recipe, not against the ColBERT package | `verify_heads.py` |
| Token classification (NER) | any `BertForTokenClassification` / RoBERTa | verified | `verify_heads.py` |
| EmbeddingGemma (`Gemma3TextModel`, bidirectional sliding window, mean pooling, two Dense, Normalize) | google/embeddinggemma-300m | verified (tiny, window smaller than the input) | `verify_decoder_enc.py` |
| Llama bidirectional with llama3 rotary scaling | nvidia/llama-embed-nemotron-8b | verified (tiny, eager attention without mask as reference) | `verify_decoder_enc.py` |
| Qwen3 embedder (`Qwen3Model`, last token, EOS suffix from the tokenizer post-processor) | Qwen/Qwen3-Embedding-* when stored as `Qwen3Model` | verified (tiny) | `verify_decoder_enc.py` |
| Qwen3-Reranker (yes/no logits) | Qwen/Qwen3-Reranker-* | verified (tiny) | `verify_decoder_enc.py` |
| Sequence classification on decoders (reward models) | any `Qwen2/Qwen3/LlamaForSequenceClassification` | verified (tiny, 1 and 3 labels) | `verify_decoder_enc.py` |
| mxbai-embed-large-v1, snowflake-arctic-embed (BERT / XLM-R v1) | mixedbread-ai/mxbai-embed-large-v1, Snowflake/snowflake-arctic-embed-* | plain BERT / XLM-R path, covered by the regression script | `verify_encoders_basic.py` |

Not done (see "Known limits"): Jina v5, Jina reranker v1/v3, jina-colbert-v2, nomic v2 MoE, ColPali / ColQwen,
all GPU work.

## Running the verification

```
python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install transformers sentence-transformers safetensors numpy tokenizers
# remote-code references (GTE, Jina) need transformers 4.x:
python3 -m venv .venv4 && .venv4/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv4/bin/pip install "transformers==4.51.3" safetensors numpy einops
cargo build --release --bin fllm-rs
export FLLM_BIN=$PWD/target/release/fllm-rs
.venv/bin/python scripts/encoder_verify/verify_encoders_basic.py
.venv/bin/python scripts/encoder_verify/verify_st_pipeline.py
.venv/bin/python scripts/encoder_verify/verify_heads.py
.venv/bin/python scripts/encoder_verify/verify_decoder_enc.py
GTE_REMOTE_CODE_DIR=/path/to/new-impl .venv4/bin/python scripts/encoder_verify/verify_gte.py
JINA_REMOTE_CODE_DIR=/path/to/xlm-roberta-flash-implementation .venv4/bin/python scripts/encoder_verify/verify_jina.py
```

`new-impl` is the Hugging Face repository Alibaba-NLP/new-impl (`modeling.py`, `configuration.py`), the Jina
directory is jinaai/xlm-roberta-flash-implementation. The scripts download nothing; the reference code is
not stored in this repository.

## Requests

Convert and serve any of the models above the usual way:

```
fllm-rs convert ./gte-base-en-v1.5 gte.gguf --outtype f16
fllm-rs serve --model gte.gguf --embeddings --port 8000
```

`/v1/models` lists `embedding_dimensions`, `max_model_len` and `capabilities`
(`dense`, `sparse`, `multi_vector`, `token_classification`, `sequence_classification`).

Dense (extension fields are all optional):

```
curl localhost:8000/v1/embeddings -H 'content-type: application/json' -d '{
  "model":"m","input":["a text","another text"],
  "dimensions":256,"encoding_format":"float","pooling":"mean","normalize":true,
  "truncate_prompt_tokens":512}'
```

Jina v3 task adapters (`task` or `adapter`; the task instruction is prepended to the text like the
reference `encode` does; without `task` the base weights run):

```
curl localhost:8000/v1/embeddings -d '{"input":"what is a vector?","task":"retrieval.query"}'
```

Sparse (SPLADE, BGE-M3) and multi-vector (BGE-M3, ColBERT). A model without a dense output (SPLADE,
ColBERT) returns `"embedding": []` and its sparse / multi-vector output by default; BGE-M3 returns the dense
vector and adds the others on request:

```
curl localhost:8000/v1/embeddings -d '{"input":"a text","sparse":true,"multi_vector":true}'
# data[i].sparse        {"token_id": weight, ...}
# data[i].multi_vector  [[...], ...]  (or a base64 string plus multi_vector_shape with encoding_format base64)
curl localhost:8000/v1/embeddings -d '{"input":"a query","multi_vector":true,"input_type":"query"}'
```

vLLM style `/pooling` (`task`: `embed`, `sparse`, `token_embed`, `token_classify`; `adapter` for Jina tasks):

```
curl localhost:8000/pooling -d '{"input":["a text"],"task":"token_embed"}'
```

Scoring. `/v1/score` and `/v1/rerank` use a cross-encoder when the model has a single-label classification
head (BERT / XLM-R cross-encoders, Jina reranker, Qwen3-Reranker), otherwise the dense cosine, the sparse
dot product or MaxSim. `score_type` selects `dense`, `sparse`, `multi_vector`, `combined` (default weights
0.4 / 0.2 / 0.4, `weights` overrides) or `cross_encoder`; BGE-M3 divides MaxSim by the number of query vectors:

```
curl localhost:8000/v1/rerank -d '{"query":"capital of france","documents":["paris is ...","berlin is ..."],"top_n":2,
  "instruction":"Given a web search query, retrieve relevant passages that answer the query"}'
curl localhost:8000/v1/score  -d '{"text_1":"a query","text_2":["doc one","doc two"],"score_type":"multi_vector"}'
```

Classification: `/classify` returns `label`, `probs` and raw `logits` for sequence classifiers (including
decoder reward models) and, for token classification models, `tokens`, `labels` and `probs` per token.

Sentence-transformers pipelines: the pooling mode of `1_Pooling/config.json` and the `Dense` /
`Normalize` modules of `modules.json` run in order. Without `modules.json` the output is normalised as before;
with `modules.json` it is normalised only if the pipeline has a `Normalize` module. `"normalize": true|false`
overrides either way.

## Decoder architectures as encoders

Decoder-style checkpoints (EmbeddingGemma, llama-embed-nemotron, `Qwen3Model`, `*ForSequenceClassification`,
Qwen3-Reranker) are converted into the BERT GGUF container (`bert.model_type = decoder_style`) and served by
the encoder server, not by the decoder server. The decoder server is untouched: its `/v1/rerank`, `/v1/score`,
`/classify` and `/pooling` routes still answer 501. A Qwen3 reranker is recognised by a checkpoint path that
contains "reranker" or by `FLLM_CONVERT_AS=reranker` (the repository's `config.json` alone does not tell it
from the language model).

## Environment variables and flags

| Name | Default | Meaning |
|---|---|---|
| `FLLM_ENCODER_ALWAYS_NORMALIZE` | unset | Normalise the embedding even when `modules.json` has no `Normalize` module (the behaviour before sentence-transformers pipelines were read). |
| `FLLM_ENCODER_MAX_INPUTS` | 32 | Maximum number of inputs per embeddings / pooling / classify request. |
| `FLLM_ENCODER_MAX_REQUEST_TOKENS` | 32768 | Maximum total tokens per request. |
| `FLLM_ENCODER_QUEUE` | 256 | Sequences that may wait in the batch queue; more get HTTP 429. |
| `FLLM_ENCODER_MAX_BATCHED_TOKENS` | 8192 | Token budget of one merged batch. `--max-batched-tokens` overrides it when given on the command line. |
| `FLLM_ENCODER_BATCH_WINDOW_MS` | 0 | Extra time the batch worker waits for more requests; 0 merges only what queued while the previous batch ran. |
| `FLLM_CONVERT_AS` | unset | `reranker` converts a `Qwen3ForCausalLM` checkpoint as a yes/no reranker. |

Request fields: `normalize`, `truncate_prompt_tokens` (`-1` = model context), `truncate` (bool),
`input_type` (`query` | `document`), `sparse`, `multi_vector`, `task` / `adapter`, `score_type`, `weights`,
`instruction`.

## THIRD_PARTY entries

- Alibaba-NLP/new-impl (`NewModel`), Apache-2.0: the model's rotary NTK scaling and gated MLP were
  re-implemented from reading the algorithm; no code copied. Used as a test reference only.
- jinaai/xlm-roberta-flash-implementation (LoRA parametrization derived from minLoRA, MIT; rest Apache-2.0 /
  CC-BY-NC for weights): the update `W + (alpha/rank) B A` was re-implemented; no code copied. Used as a test
  reference only.
- ColBERT (Stanford FutureData, MIT) and FlagEmbedding / BGE-M3 (MIT): scoring and head formulas were
  re-implemented from the published description. No code copied.

## Test levels

- L0 compiles only: `rust-gpu` feature build (clippy), nothing for `cuda`.
- L2 text matches the Python reference on the CPU: every row of the model table marked "verified"; commands above.
- Unit tests (`cargo test`): sentence-transformers module parsing, pooling-mode parsing, head detection, MaxSim,
  sparse dot, LoRA arithmetic, rotary and RMS norm, llama3 scaling, request option parsing, output defaults,
  score-mode selection.
- No L1 (device) and no L3 (benchmark) results exist for any of this.

## Known limits

- CPU only. The BERT six-linear wgpu offload is unchanged; the new paths (GTE, Jina, decoder-style, heads) run on
  the CPU, and `--device` other than `cpu` errors for ModernBERT only as before. No CUDA encoder kernel, no
  encoder `selftest-gpu` cases, no fp16/bf16 activations; the CPU path computes in f32.
- Not implemented: Jina v5, jina-reranker v1 (ALiBi) and v3, jina-colbert-v2, nomic v2 MoE, ColPali / ColQwen
  (no vision tower, the converter reports an unsupported architecture), PyLate token-level Dense (`Dense` before
  `Pooling` is rejected with a clear error), `LlamaBidirectionalModel` checkpoints that need other remote-code
  features, rotary scalings other than `llama3` and NTK.
- Jina reranker v2 conversion was only checked against the published tensor header, not numerically.
- Real BGE-M3 sparse/ColBERT, SPLADE and ColBERT checkpoints were not run; the head formulas follow the
  FlagEmbedding, naver/splade and ColBERT recipes and are checked against references written from them.
- The decoder server's `/v1/rerank`, `/v1/score`, `/classify`, `/pooling` still return 501; decoder-style
  models are served by the encoder server instead (see above).
- Dense decoder-style models hold weights in the GGUF dtype and compute in f32 on the CPU; an 8B model is slow.
- Truncation keeps the leading tokens and the final separator; left truncation is not offered.
