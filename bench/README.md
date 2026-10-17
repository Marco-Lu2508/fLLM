# Streaming benchmarks

`fllm-rs bench` is an HTTP client for an already running OpenAI-compatible
server. It does not load a model. `bench-report` compares saved results.
No measured baseline is included; create one on the intended runner after
reviewing a real measurement. These instructions are examples to run later,
not evidence that any server, model or workflow has been tested.

## Prepare a comparable model

Use one immutable revision of a small dense model supported by every backend,
for example Qwen2.5-0.5B-Instruct. Prepare a local **FP16 checkpoint** with its
original tokenizer, config and chat template. Record the upstream revision,
file hashes and any dtype conversion. The examples assume that checkpoint is
`/models/bench-f16` and its identical FP16 GGUF export is
`/models/bench-f16.gguf`. If the checkpoint is BF16, do not call it FP16 merely
because a server is instructed to use FP16; first prepare the same rounded
FP16 weight set for all engines. Model export is outside the benchmark client.

For the GGUF export, from the fLLM root:

```sh
cargo build --release -p fllm-engine --bin fllm-rs --features rust-gpu
target/release/fllm-rs convert /models/bench-f16 /models/bench-f16.gguf --outtype f16
```

Do not equate GGUF Q4_K_M, GPTQ/AWQ 4-bit and EXL2/EXL3 4 bpw: they have
different weights and kernels. An all-engine comparison should use supported
FP16 weights; a same-GGUF quantized comparison can instead cover fLLM,
llama.cpp and Ollama. BF16, FP16 and mixed tensor dtypes must be labeled
separately. Pin the server versions as well as the model revision.

## Start each server separately

The commands below use one GPU, a 20480-token per-request context, up to 32
requests, FP16 KV storage where selectable, and eight host threads. The GPU
must have enough memory for each server's cache policy. Stop the previous
server before starting another. A CUDA engine and a Vulkan engine may use the
same physical GPU, but record the API and offload placement. Run a separate
CPU comparison if GPU support or model support is unavailable in one engine.

Set these variables in the shell used for each example:

```sh
export HF_MODEL=/models/bench-f16
export GGUF=/models/bench-f16.gguf
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
```

fLLM (the binary must have been built with `rust-gpu`):

```sh
target/release/fllm-rs serve --model "$GGUF" --model-name bench-f16 \
  --device vulkan --host 127.0.0.1 --port 8000 --ctx 20480 \
  --max-seqs 32 --max-batched-tokens 512 --threads 8 --kv-cache-type f16
```

For the CPU track use `--device cpu`, and use CPU placement in the other
servers. fLLM prefix caching follows its current scheduler policy; the client
varies prompts across calibration, warmup, repetitions and streams to avoid
replaying a fully cached prompt.

llama.cpp, using a pinned build of `llama-server`:

```sh
llama-server --model "$GGUF" --alias bench-f16 --host 127.0.0.1 --port 8000 \
  --ctx-size 655360 --parallel 32 --threads 8 --threads-batch 8 \
  --n-gpu-layers 99 --cache-type-k f16 --cache-type-v f16 --cache-ram 0
```

The total context allocation is `32 * 20480` for this example; confirm the
logged per-slot limit for the pinned llama.cpp version. `--cache-ram 0`
disables the host prompt cache, but does not guarantee empty slot KV state.
For CPU placement use `--n-gpu-layers 0`.
See the [official server flags](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

vLLM, in its installed environment:

```sh
vllm serve "$HF_MODEL" --served-model-name bench-f16 --host 127.0.0.1 \
  --port 8000 --dtype float16 --kv-cache-dtype auto --max-model-len 20480 \
  --max-num-seqs 32 --tensor-parallel-size 1 --generation-config vllm \
  --no-enable-prefix-caching
```

Pin vLLM and record its batching defaults and memory fraction. The
[official CLI](https://docs.vllm.ai/en/latest/cli/serve/) documents version-specific flags.

SGLang, in its installed environment:

```sh
python -m sglang.launch_server --model-path "$HF_MODEL" \
  --served-model-name bench-f16 --host 127.0.0.1 --port 8000 \
  --dtype float16 --context-length 20480 --max-running-requests 32 \
  --tp-size 1 --disable-radix-cache
```

Do not configure a draft model. Record chunked-prefill and memory settings;
they are not equivalent to another engine's defaults. See SGLang's
[benchmark guide](https://docs.sglang.ai/developer_guide/bench_serving).

TabbyAPI / ExLlama, from a pinned TabbyAPI checkout with its dependencies
installed: write the following `config.yml` in **that checkout**, then start
`OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python main.py`.

```yaml
network:
  host: 127.0.0.1
  port: 8000
  disable_auth: false
model:
  model_dir: /models
  model_name: bench-f16
  max_seq_len: 20480
  cache_size: 655360
  cache_mode: FP16
  max_batch_size: 32
draft_model:
  draft_model_name: null
sampling:
  override_preset: null
```

Use the generated API key through `FLLM_API_KEY`; unset it for other local
servers that do not require authentication. TabbyAPI's current backend and
supported weight formats differ across releases; use a pinned version that
supports the prepared FP16 model. See the
[official configuration](https://github.com/theroyallab/tabbyAPI/blob/main/config_sample.yml)
and [start instructions](https://github.com/theroyallab/tabbyAPI/wiki/01.-Getting-Started).

Ollama: stop any existing service using port 8000. Start this process in one
terminal, and keep it running:

```sh
OLLAMA_HOST=127.0.0.1:8000 OLLAMA_NUM_PARALLEL=32 OLLAMA_MAX_LOADED_MODELS=1 \
  OLLAMA_CONTEXT_LENGTH=20480 OLLAMA_KV_CACHE_TYPE=f16 OLLAMA_KEEP_ALIVE=-1 \
  OMP_NUM_THREADS=8 ollama serve
```

In another terminal create a local model from the same GGUF. Completions mode
below avoids comparing differing chat templates:

```sh
cat > /tmp/fllm-bench.Modelfile <<'EOF'
FROM /models/bench-f16.gguf
PARAMETER num_ctx 20480
PARAMETER num_thread 8
PARAMETER temperature 0
PARAMETER repeat_penalty 1
PARAMETER seed 42
PARAMETER draft_num_predict 0
EOF
OLLAMA_HOST=127.0.0.1:8000 ollama create bench-f16 -f /tmp/fllm-bench.Modelfile
```

Confirm device placement in its log rather than assuming full GPU offload.
For a chat-mode comparison, explicitly match the original template and stop
tokens in the Ollama Modelfile to the other engines. References:
[Modelfile](https://docs.ollama.com/modelfile),
[OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility),
[server environment configuration](https://github.com/ollama/ollama/blob/main/envconfig/config.go).

## Run the same workload

Wait for the chosen server to finish loading. Execute from the fLLM root:

```sh
target/release/fllm-rs bench --url http://127.0.0.1:8000 --endpoint completions \
  --prompt-lens 128,512,2048,8192 --context-lens 1024,4096,16384 \
  --concurrency 1,2,4,8,16,32 --gen-len 256 --input-len 64 \
  --warmup 1 --repeats 3 --seed 42 --timeout 600 --length-control none \
  --label 'fLLM COMMIT; GPU FP16' --notes 'model revision; GPU; driver; launch flags' \
  --json fllm.json
```

Repeat this exact workload against each separately started server. Change only
label, notes, JSON filename and, when necessary, `--model`/URL/API key. The
model defaults to the first reported `/v1/models` entry; choose `--model` for
servers with several models. `--endpoint chat` uses one user message instead
of a raw completion prompt. Never compare the two endpoints as the same workload.

`--length-control auto` (the default) probes `min_tokens` and `ignore_eos`
together during calibration, falling back to `max_tokens` after HTTP 400/422.
Acceptance does not prove the fields were honored. Choose `min-tokens`,
`ignore-eos` or `both` only where the pinned server supports them. For an
all-engine run the example uses `none`: generation may end early; actual
usage and finish reasons are retained. Server sampling defaults can still
affect generation. Do not compare very different actual output lengths.

`--scenarios prefill,decode-single` selects a subset. `--requests` optionally
specifies the number of concurrency requests per repetition (at least the
largest concurrency). Otherwise a concurrency case starts exactly N streams.
`--shared-prefix` is a separate cache-enabled workload: 90% of prompt words
are shared, with at least eight unique suffix words when the length permits.
The shared prefix is also reused between repetitions. Both cases must be
labeled and compared separately. `--max-new` remains an alias for `--gen-len`;
old `--input-len` now means a token target rather than a repeated-word count.

```sh
target/release/fllm-rs bench-report fllm.json llama.json vllm.json sglang.json tabby.json ollama.json > comparison.md
target/release/fllm-rs bench-report --baseline bench/baseline.json --current fllm.json --max-regression 7
```

Rate changes use the first file as baseline: `[green: better]`, `[red: worse]` and
`[equal]` are plain text. Regression mode exits 1 for a primary median rate
loss strictly greater than the threshold, missing metrics/cases, measured
request errors, incomplete repetitions or incompatible parameters. Actual
prompt and completion medians must also agree within 10%; otherwise a faster
but shorter output is not accepted as an improvement. No ANSI is emitted.

## Measurement definitions and schema 1

The JSON format is stable within `schema: 1`; additive fields are allowed.
Consumers should ignore unknown fields and reject other schema versions.
Unavailable numbers are `null`, not zero. No estimated token count substitutes
for a missing server usage value. Durations use monotonic `Instant`; only the
metadata timestamp uses UTC wall time.

| Field | Meaning |
| --- | --- |
| `schema`, `bench_version`, `timestamp` | Schema integer, implementation version, RFC3339 start timestamp |
| `metadata` | Server URL, requested model, complete `/v1/models` JSON, optional server commit, compile-time client commit/dirty flag, client host inventory |
| `parameters` | All resolved CLI parameters, seed, word-generator version and fixed request options; only API-key presence is stored |
| `length_control_effective`, `warnings` | Accepted/fallback request mode and interpretation limitations |
| `results[]` | One case per requested prompt length or concurrency, with a stable workload `id` |
| `results[].target_prompt_tokens`, `requested_completion_tokens` | Requested limits, not measured token counts |
| `prompt_words`, `case_seed`, `shared_prefix_words` | Calibrated deterministic generator settings |
| `calibration[]` | One-token probes: initial word count, optional capability retry, and exactly one length adjustment/recheck; excluded from results |
| `warmups[]`, `repetitions[]` | Full raw repetitions; only `repetitions` contribute to summaries |
| `repetitions[].requests[]` | Request index, seed, word count, HTTP status, returned model, actual usage, finish reason, `[DONE]`, error and timings |
| `content_events_seconds` | Arrival times from POST start of nonempty content/reasoning/text or explicit token-ID/logprob events, excluding role-only/usage events |
| `first_event_tokens` | Explicit token-ID/logprob count in first event, or null; never inferred from text |
| `ttft_seconds`, `elapsed_seconds` | First observable generated event / entire response through `[DONE]`; client HTTP/queue latency included |
| `prefill_tokens_per_second` | `usage.prompt_tokens / TTFT`; an effective client rate including first-token compute, queueing, templates, networking and any cache reuse |
| `decode_tokens_per_second` | `(usage.completion_tokens - first_event_tokens) / (last_content_event - first_content_event)`; subtract one when first-event count is unavailable; null for insufficient events/tokens |
| `content_event_interval_ms` | Inter-arrival statistics of content events: observed SSE cadence, **not guaranteed per-token cadence** |
| `repetitions[].metrics` | Median per-stream prefill/decode rate; aggregate completed-token throughput; TTFT p50/p95; pooled event interval statistics |
| `observed_throughput_tokens_per_second` | Successfully completed token total / batch wall time, retained even if another request failed |
| `results[].metrics` | For each repetition metric: `count`, `median`, `min`, `max`, `mean`, `p05`, `p95`; all repetitions have equal weight |
| `errors`, `successful_requests` | Measured request outcomes; failed repetitions have null regular rate metrics and remain visible in raw output |
| `actual_prompt_tokens`, `actual_completion_tokens` | Usage distributions across measured requests, including any reported usage in failed streams |

Quantiles use linear interpolation at `(count - 1) * q`; the median of an
even-length sample is the mean of its middle pair. A per-repeat concurrency
throughput covers the interval from simultaneous worker release through the
last completed/failed response, including client setup and delivery overhead.
Per-stream metrics and raw requests remain available separately. With fewer
than two observable events, decode rate and event intervals are unavailable.
If one event holds several tokens or includes buffering, decode rate is only
a client-observed approximation even though its numerator is actual usage.
Special/EOS or invisible reasoning tokens counted by a server can also differ
from visible content. Do not infer token boundaries from UTF-8 bytes or words.

Calibration generates words from a fixed 64-word list with SplitMix64. Each
case uses its own deterministic seed; all phases use distinct prompt seeds.
The initial response supplies `prompt_tokens`; the word count is scaled by
`target / actual` and rechecked once. The final measured usage, rather than the
target, drives every rate. Chat-template overhead and tokenization variability
mean the requested length is approximate. Calibration/warmup failures are
retained; a measured request failure makes `bench` exit 1 **after** completing
the remaining cases and writing JSON. Discovery failure requires `--model`
to proceed. `--timeout` covers each complete HTTP request, not the entire run.

Host information describes the **client**, not a remote inference machine.
Linux reports CPU flags/features, CPU model, logical processors and RAM;
macOS uses sysctl; Windows uses the CPU registry entry and CIM for RAM.
Optional GPU inventory commands have three-second deadlines. Missing tools
or unavailable information produce null/empty fields. `build.rs` embeds the
client Git revision, with `FLLM_GIT_COMMIT` as a build-time override; source
archives without Git and without that variable report `unknown`.

## Fair-comparison checklist

- Pin identical model revision, weights, quantization, tokenizer and adapters;
  use the same chat template or completions mode.
- Match context limits, actual usage lengths, KV-cache dtype, batch/concurrency
  limits, device placement and offload. Reject silent context truncation.
- Record all server flags, versions, GPU drivers, thread/affinity counts and
  CPU/GPU power profiles. Host helper threads are not equivalent to GPU kernels.
- Keep warmup/repeat counts, seeds, generation limits and temperature identical;
  record EOS/stop policy, repetition penalties and other sampling overrides.
- Separate unique-prefix and shared-prefix runs. Prefix-cache hits can make
  apparent prefill throughput much higher without changing raw compute speed.
- Disable speculative decoding, drafting/MTP and lookups for a normal run;
  benchmark these separately if desired. Their bursty output changes cadence.
- Run one server and no other heavy workloads. Avoid thermal throttling, power
  saving, paging and concurrent CI builds. Hold the runner exclusive for baselines.
- Prefer localhost. Record network/proxy buffering, client CPU limitations and
  SSE batching; TTFT and token/event arrival intervals are client measurements.
- Inspect every error, missing usage field, actual length and min/max spread.
  Three repetitions reveal large changes, not statistical certainty.

## Manual CI

The GitHub Actions workflow `streaming-benchmark` (`.github/workflows/bench.yml`) runs manually (Actions -> streaming-benchmark ->
Run workflow) on a self-hosted runner whose labels you give as `runner_labels`. Set the input `model`, or the repository variable
`BENCH_MODEL`, to an absolute path of a preinstalled small GGUF on that runner. It builds a fresh CPU binary, starts its own server,
waits for readiness, runs a small workload, uploads JSON plus logs, and checks `bench/baseline.json` with a 10% threshold **only if
that file exists**. The model is never downloaded by CI. No baseline is synthesized.

The runner needs Rust, Git, curl and a POSIX shell; port 18107 on loopback is used by the benchmark job.
An unconfigured model path fails clearly. Benchmark errors still upload any produced JSON. A single concurrency group prevents
overlapping benchmark jobs; it cannot exclude unrelated jobs or other processes on the same host. See [docs/ci.md](../docs/ci.md)
and [docs/benchmarking.md](../docs/benchmarking.md).
