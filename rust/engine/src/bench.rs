//! Client-side streaming benchmarks. All token counts come from server usage.
use reqwest::blocking::Client;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufRead, BufReader, Read, Write},
    path::PathBuf,
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Barrier,
    },
    thread,
    time::{Duration, Instant},
};

const VERSION: &str = "1.0.0";
const WORDS: &[&str] = &[
    "river", "stone", "forest", "cloud", "window", "garden", "paper", "bridge", "orange", "silver",
    "quiet", "morning", "planet", "harbor", "meadow", "copper", "basket", "shadow", "winter",
    "summer", "pebble", "valley", "lantern", "pencil", "ocean", "branch", "island", "cotton",
    "mirror", "castle", "purple", "yellow", "circle", "square", "velvet", "rabbit", "turtle",
    "mountain", "breeze", "travel", "button", "ribbon", "compass", "village", "picture", "candle",
    "hammer", "needle", "book", "table", "chair", "door", "field", "road", "train", "wheel",
    "water", "bread", "apple", "flower", "leaf", "grass", "music", "story",
];

pub const HELP: &str = "usage: fllm-rs bench [OPTIONS]
  --url URL                 Server root or /v1 (default http://127.0.0.1:8000)
  --model ID                Default: first model reported by GET /v1/models
  --api-key KEY             Or FLLM_API_KEY; never written to results
  --endpoint chat|completions (default chat)
  --scenarios LIST          prefill,decode-single,decode-context,concurrency
  --prompt-lens LIST        128,512,2048,8192
  --context-lens LIST       1024,4096,16384
  --concurrency LIST        1,2,4,8,16,32
  --gen-len N               256 (alias --max-new)
  --input-len N             Short/concurrency prompt target: 64
  --requests N              Concurrency requests/repeat; default N simultaneous streams
  --shared-prefix           Share 90% of concurrency prompt words
  --warmup N                1 (zero allowed)
  --repeats N               3
  --seed N                  42
  --timeout SECONDS         Entire HTTP request deadline: 600
  --length-control MODE     auto|none|min-tokens|ignore-eos|both (default auto)
  --label TEXT --notes TEXT --json PATH
See bench/README.md for schema, definitions and comparison limitations.";

pub const REPORT_HELP: &str = "usage: fllm-rs bench-report A.json B.json [C.json ...]
       fllm-rs bench-report --baseline BASE.json --current NEW.json --max-regression 7
Prints Markdown. Regression mode fails for missing/invalid cases, request errors,
incompatible workloads or a primary rate falling more than the percentage limit.";

#[derive(Clone)]
struct Options {
    root: String,
    model: Option<String>,
    api_key: Option<String>,
    endpoint: String,
    scenarios: Vec<String>,
    prompt_lens: Vec<usize>,
    context_lens: Vec<usize>,
    concurrency: Vec<usize>,
    gen_len: usize,
    input_len: usize,
    requests: Option<usize>,
    shared_prefix: bool,
    warmup: usize,
    repeats: usize,
    seed: u64,
    timeout: u64,
    length_control: String,
    label: Option<String>,
    notes: Option<String>,
    output: Option<PathBuf>,
}

fn positive(value: &str, name: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|n| *n > 0)
        .ok_or_else(|| format!("{name} requires a positive integer"))
}
fn lengths(value: &str, name: &str, limit: usize) -> Result<Vec<usize>, String> {
    let result = value
        .split(',')
        .map(|s| positive(s.trim(), name))
        .collect::<Result<Vec<_>, _>>()?;
    if result.len() > 128 || result.iter().any(|n| *n > limit) {
        return Err(format!(
            "{name} accepts at most 128 entries, each <= {limit}"
        ));
    }
    if result.iter().collect::<BTreeSet<_>>().len() != result.len() {
        return Err(format!("{name} contains duplicate entries"));
    }
    Ok(result)
}
fn parse(args: &[String]) -> Result<Options, String> {
    let mut o = Options {
        root: "http://127.0.0.1:8000".into(),
        model: None,
        api_key: std::env::var("FLLM_API_KEY").ok(),
        endpoint: "chat".into(),
        scenarios: ["prefill", "decode-single", "decode-context", "concurrency"]
            .map(str::to_owned)
            .to_vec(),
        prompt_lens: vec![128, 512, 2048, 8192],
        context_lens: vec![1024, 4096, 16384],
        concurrency: vec![1, 2, 4, 8, 16, 32],
        gen_len: 256,
        input_len: 64,
        requests: None,
        shared_prefix: false,
        warmup: 1,
        repeats: 3,
        seed: 42,
        timeout: 600,
        length_control: "auto".into(),
        label: None,
        notes: None,
        output: None,
    };
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        if flag == "--shared-prefix" {
            o.shared_prefix = true;
            i += 1;
            continue;
        }
        i += 1;
        let value = args
            .get(i)
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag {
            "--url" => o.root = value.clone(),
            "--model" => o.model = Some(value.clone()),
            "--api-key" => o.api_key = Some(value.clone()),
            "--endpoint" => o.endpoint = value.clone(),
            "--scenarios" => o.scenarios = value.split(',').map(|s| s.trim().to_owned()).collect(),
            "--prompt-lens" => o.prompt_lens = lengths(value, flag, 1_000_000)?,
            "--context-lens" => o.context_lens = lengths(value, flag, 1_000_000)?,
            "--concurrency" => o.concurrency = lengths(value, flag, 1024)?,
            "--gen-len" | "--max-new" => o.gen_len = positive(value, flag)?,
            "--input-len" => o.input_len = positive(value, flag)?,
            "--requests" => o.requests = Some(positive(value, flag)?),
            "--warmup" => {
                o.warmup = value
                    .parse()
                    .map_err(|_| "--warmup requires a non-negative integer")?
            }
            "--repeats" => o.repeats = positive(value, flag)?,
            "--seed" => {
                o.seed = value
                    .parse()
                    .map_err(|_| "--seed requires an unsigned integer")?
            }
            "--timeout" => o.timeout = positive(value, flag)? as u64,
            "--length-control" => o.length_control = value.clone(),
            "--label" => o.label = Some(value.clone()),
            "--notes" => o.notes = Some(value.clone()),
            "--json" => o.output = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown bench option: {flag}")),
        }
        i += 1;
    }
    if !["chat", "completions"].contains(&o.endpoint.as_str()) {
        return Err("--endpoint must be chat or completions".into());
    }
    if !["auto", "none", "min-tokens", "ignore-eos", "both"].contains(&o.length_control.as_str()) {
        return Err("invalid --length-control".into());
    }
    if o.scenarios.is_empty()
        || o.scenarios.iter().any(|s| {
            !["prefill", "decode-single", "decode-context", "concurrency"].contains(&s.as_str())
        })
        || o.scenarios.iter().collect::<BTreeSet<_>>().len() != o.scenarios.len()
    {
        return Err("invalid or duplicate --scenarios entry".into());
    }
    if o.input_len > 1_000_000
        || o.gen_len > 1_000_000
        || o.repeats > 10_000
        || o.warmup > 10_000
        || o.requests.is_some_and(|n| n > 1_000_000)
        || o.timeout > 86400
    {
        return Err("benchmark parameters exceed resource limits (lengths/requests <= 1000000, repeats/warmup <= 10000, timeout <= 86400)".into());
    }
    if o.scenarios.iter().any(|s| s == "concurrency")
        && o.requests
            .is_some_and(|n| o.concurrency.iter().any(|c| *c > n))
    {
        return Err("--requests must be at least the largest --concurrency value".into());
    }
    let mut url = reqwest::Url::parse(&o.root).map_err(|e| format!("invalid --url: {e}"))?;
    if !["http", "https"].contains(&url.scheme())
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(
            "--url must be an HTTP(S) server URL without credentials, query or fragment".into(),
        );
    }
    let path = url.path().trim_end_matches('/').to_owned();
    let root = path
        .strip_suffix("/v1/chat/completions")
        .or_else(|| path.strip_suffix("/v1/completions"))
        .or_else(|| path.strip_suffix("/v1"))
        .unwrap_or(&path)
        .to_owned();
    url.set_path(&root);
    o.root = url.to_string().trim_end_matches('/').to_owned();
    Ok(o)
}

fn parameters(o: &Options) -> Value {
    json!({"url":o.root,"model":o.model,"api_key_present":o.api_key.is_some(),"endpoint":o.endpoint,
        "scenarios":o.scenarios,"prompt_lens":o.prompt_lens,"context_lens":o.context_lens,"concurrency":o.concurrency,
        "gen_len":o.gen_len,"input_len":o.input_len,"requests":o.requests,"shared_prefix":o.shared_prefix,
        "warmup":o.warmup,"repeats":o.repeats,"seed":o.seed,"timeout_seconds":o.timeout,
        "length_control_requested":o.length_control,"label":o.label,"notes":o.notes,"json":o.output,
        "temperature":0,"top_p":1,"n":1,"stream":true,"include_usage":true,"prompt_generator":"word-list-splitmix64-v1"})
}
fn mix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
fn prompt(words: usize, seed: u64, shared: Option<u64>) -> String {
    let prefix = if shared.is_some() {
        words.saturating_sub(8).min(words * 9 / 10)
    } else {
        0
    };
    let mut prefix_state = shared.unwrap_or(seed);
    let mut state = seed;
    let mut text = String::with_capacity(words * 8);
    for i in 0..words {
        if i > 0 {
            text.push(' ');
        }
        let rng = if i < prefix {
            &mut prefix_state
        } else {
            &mut state
        };
        text.push_str(WORDS[mix(rng) as usize % WORDS.len()]);
    }
    text
}

#[derive(Clone)]
struct Case {
    id: String,
    kind: String,
    target: usize,
    gen: usize,
    concurrency: usize,
    requests: usize,
    seed: u64,
}
fn cases(o: &Options) -> Vec<Case> {
    let mut result = Vec::new();
    let mut state = o.seed;
    for kind in &o.scenarios {
        let specs = match kind.as_str() {
            "prefill" => o
                .prompt_lens
                .iter()
                .map(|n| (*n, 1, 1, 1))
                .collect::<Vec<_>>(),
            "decode-single" => vec![(o.input_len, o.gen_len, 1, 1)],
            "decode-context" => o
                .context_lens
                .iter()
                .map(|n| (*n, o.gen_len, 1, 1))
                .collect(),
            _ => o
                .concurrency
                .iter()
                .map(|n| (o.input_len, o.gen_len, *n, o.requests.unwrap_or(*n)))
                .collect(),
        };
        for (target, gen, concurrency, requests) in specs {
            result.push(Case {
                id: format!("{kind}/p{target}/g{gen}/c{concurrency}/r{requests}"),
                kind: kind.clone(),
                target,
                gen,
                concurrency,
                requests,
                seed: mix(&mut state),
            });
        }
    }
    result
}

#[derive(Default)]
struct Sample {
    index: usize,
    seed: u64,
    words: usize,
    status: Option<u16>,
    model: Option<String>,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    usage: Option<Value>,
    events: Vec<f64>,
    first_event_tokens: Option<u64>,
    finish_reason: Option<String>,
    elapsed: f64,
    done: bool,
    error: Option<String>,
}
impl Sample {
    fn ttft(&self) -> Option<f64> {
        self.events.first().copied()
    }
    fn prefill_rate(&self) -> Option<f64> {
        rate(self.prompt_tokens? as f64, self.ttft()?)
    }
    fn decode_rate(&self) -> Option<f64> {
        let elapsed = self.events.last()? - self.events.first()?;
        let first = self.first_event_tokens.unwrap_or(1);
        let remaining = self.completion_tokens?.checked_sub(first)?;
        if remaining == 0 {
            None
        } else {
            rate(remaining as f64, elapsed)
        }
    }
    fn intervals(&self) -> Vec<f64> {
        self.events
            .windows(2)
            .map(|pair| (pair[1] - pair[0]) * 1000.0)
            .collect()
    }
    fn value(&self) -> Value {
        json!({"index":self.index,"prompt_seed":self.seed,"prompt_words":self.words,"http_status":self.status,
            "response_model":self.model,"prompt_tokens":self.prompt_tokens,"completion_tokens":self.completion_tokens,"usage":self.usage,
            "content_events_seconds":self.events,"first_event_tokens":self.first_event_tokens,
            "ttft_seconds":self.ttft(),"elapsed_seconds":self.elapsed,"prefill_tokens_per_second":self.prefill_rate(),
            "decode_tokens_per_second":self.decode_rate(),"content_event_interval_ms":summary(self.intervals()),
            "finish_reason":self.finish_reason,"done":self.done,"error":self.error})
    }
}
fn rate(tokens: f64, seconds: f64) -> Option<f64> {
    (seconds > 0.0)
        .then_some(tokens / seconds)
        .filter(|n| n.is_finite())
}
fn quantile(sorted: &[f64], q: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let at = (sorted.len() - 1) as f64 * q;
    let low = at.floor() as usize;
    let high = at.ceil() as usize;
    Some(sorted[low] + (sorted[high] - sorted[low]) * (at - low as f64))
}
fn summary(mut values: Vec<f64>) -> Value {
    values.retain(|n| n.is_finite());
    values.sort_by(f64::total_cmp);
    json!({"count":values.len(),"median":quantile(&values,0.5),"min":values.first(),"max":values.last(),
        "mean":if values.is_empty() { None } else { Some(values.iter().sum::<f64>()/values.len() as f64) },
        "p05":quantile(&values,0.05),"p95":quantile(&values,0.95)})
}

// SSE event boundaries are observations, not assumed tokenizer boundaries.
fn consume(data: &str, at: f64, sample: &mut Sample) -> Result<(), String> {
    if data.trim() == "[DONE]" {
        sample.done = true;
        return Ok(());
    }
    let v: Value = serde_json::from_str(data).map_err(|e| format!("invalid SSE JSON: {e}"))?;
    if let Some(e) = v.get("error") {
        return Err(format!("server stream error: {e}"));
    }
    if let Some(model) = v["model"].as_str() {
        sample.model = Some(model.into());
    }
    if v["usage"].is_object() {
        let u = &v["usage"];
        if let (Some(p), Some(c)) = (u["prompt_tokens"].as_u64(), u["completion_tokens"].as_u64()) {
            sample.prompt_tokens = Some(p);
            sample.completion_tokens = Some(c);
            sample.usage = Some(u.clone());
        }
    }
    if let Some(choices) = v["choices"].as_array() {
        for choice in choices {
            if choice["index"].as_u64().is_some_and(|n| n != 0) {
                continue;
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                sample.finish_reason = Some(reason.into());
            }
            let delta = &choice["delta"];
            let ids = delta["token_ids"]
                .as_array()
                .or_else(|| choice["token_ids"].as_array());
            let logprobs = choice
                .pointer("/logprobs/content")
                .and_then(Value::as_array);
            let tokens = ids
                .or(logprobs)
                .filter(|a| !a.is_empty())
                .map(|a| a.len() as u64);
            let content = [
                delta.get("content"),
                delta.get("reasoning_content"),
                delta.get("reasoning"),
                choice.get("text"),
            ]
            .iter()
            .any(|v| v.and_then(Value::as_str).is_some_and(|s| !s.is_empty()));
            if content || tokens.is_some() {
                if sample.events.is_empty() {
                    sample.first_event_tokens = tokens;
                }
                sample.events.push(at);
            }
        }
    }
    Ok(())
}
fn read_line(reader: &mut impl BufRead, line: &mut Vec<u8>) -> Result<bool, String> {
    line.clear();
    loop {
        let buf = reader
            .fill_buf()
            .map_err(|e| format!("stream interrupted or timed out: {e}"))?;
        if buf.is_empty() {
            return Ok(!line.is_empty());
        }
        let n = buf
            .iter()
            .position(|b| *b == b'\n')
            .map_or(buf.len(), |n| n + 1);
        if line.len() + n > 16 * 1024 * 1024 {
            return Err("SSE line exceeds 16 MiB".into());
        }
        let ended = buf[n - 1] == b'\n';
        line.extend_from_slice(&buf[..n]);
        reader.consume(n);
        if ended {
            return Ok(true);
        }
    }
}
struct Work {
    words: usize,
    seed: u64,
    index: usize,
}
fn request(
    client: &Client,
    o: &Options,
    model: &str,
    case: &Case,
    control: &str,
    work: Work,
) -> Sample {
    let Work { words, seed, index } = work;
    let shared = (o.shared_prefix && case.kind == "concurrency").then_some(case.seed);
    let text = prompt(words, seed, shared);
    let mut body = json!({"model":model,"max_tokens":case.gen,"temperature":0,"top_p":1,"seed":o.seed,
        "n":1,"stream":true,"stream_options":{"include_usage":true}});
    if o.endpoint == "chat" {
        body["messages"] = json!([{"role":"user","content":text}]);
    } else {
        body["prompt"] = json!(text);
    }
    if ["min-tokens", "both"].contains(&control) {
        body["min_tokens"] = json!(case.gen);
    }
    if ["ignore-eos", "both"].contains(&control) {
        body["ignore_eos"] = json!(true);
    }
    let mut sample = Sample {
        index,
        seed,
        words,
        ..Sample::default()
    };
    let endpoint = if o.endpoint == "chat" {
        "chat/completions"
    } else {
        "completions"
    };
    let mut req = client.post(format!("{}/v1/{endpoint}", o.root)).json(&body);
    if let Some(key) = &o.api_key {
        req = req.bearer_auth(key);
    }
    let started = Instant::now();
    let result = (|| -> Result<(), String> {
        let mut response = req
            .send()
            .map_err(|e| format!("connection failed or timed out: {}", e.without_url()))?;
        sample.status = Some(response.status().as_u16());
        if !response.status().is_success() {
            let mut text = String::new();
            let _ = Read::by_ref(&mut response)
                .take(8192)
                .read_to_string(&mut text);
            return Err(format!("HTTP {}: {text}", response.status()));
        }
        if !response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|s| s.to_str().ok())
            .is_some_and(|s| s.to_ascii_lowercase().starts_with("text/event-stream"))
        {
            return Err("server did not return text/event-stream".into());
        }
        let mut reader = BufReader::new(response);
        let mut line = Vec::new();
        let mut data = String::new();
        let mut at = 0.0;
        while read_line(&mut reader, &mut line)? {
            let text = std::str::from_utf8(&line)
                .map_err(|e| format!("invalid SSE UTF-8: {e}"))?
                .trim_end_matches(['\r', '\n']);
            if text.is_empty() {
                if !data.is_empty() {
                    consume(&data, at, &mut sample)?;
                    data.clear();
                    if sample.done {
                        break;
                    }
                }
            } else if let Some(value) = text.strip_prefix("data:") {
                at = started.elapsed().as_secs_f64();
                if !data.is_empty() {
                    data.push('\n');
                }
                if data.len() + value.len() > 16 * 1024 * 1024 {
                    return Err("SSE event exceeds 16 MiB".into());
                }
                data.push_str(value.strip_prefix(' ').unwrap_or(value));
            }
        }
        if !data.is_empty() && !sample.done {
            consume(&data, at, &mut sample)?;
        }
        if !sample.done {
            return Err("stream closed before [DONE]".into());
        }
        if sample.prompt_tokens.is_none() || sample.completion_tokens.is_none() {
            return Err("stream has no valid usage.prompt_tokens and usage.completion_tokens; token counts are never estimated".into());
        }
        if sample.completion_tokens.is_some_and(|n| n > 0) && sample.events.is_empty() {
            return Err(
                "server reports generated tokens but no observable content/token event".into(),
            );
        }
        Ok(())
    })();
    sample.elapsed = started.elapsed().as_secs_f64();
    sample.error = result.err();
    sample
}

fn batch(
    client: &Client,
    o: &Options,
    model: &str,
    case: &Case,
    words: usize,
    seed: u64,
    control: &str,
) -> (Vec<Sample>, f64) {
    let next = AtomicUsize::new(0);
    let ready = Barrier::new(case.concurrency + 1);
    let mut samples = Vec::with_capacity(case.requests);
    let elapsed = thread::scope(|scope| {
        let mut workers = Vec::new();
        for _ in 0..case.concurrency {
            workers.push(scope.spawn(|| {
                ready.wait();
                ready.wait();
                let mut result = Vec::new();
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= case.requests {
                        break;
                    }
                    let mut state =
                        seed.wrapping_add((index as u64).wrapping_mul(0x9e3779b97f4a7c15));
                    result.push(request(
                        client,
                        o,
                        model,
                        case,
                        control,
                        Work {
                            words,
                            seed: mix(&mut state),
                            index,
                        },
                    ));
                }
                result
            }));
        }
        ready.wait();
        let start = Instant::now();
        ready.wait();
        for worker in workers {
            if let Ok(mut result) = worker.join() {
                samples.append(&mut result);
            }
        }
        start.elapsed().as_secs_f64()
    });
    let found = samples.iter().map(|s| s.index).collect::<BTreeSet<_>>();
    for index in 0..case.requests {
        if !found.contains(&index) {
            samples.push(Sample {
                index,
                words,
                error: Some("benchmark worker panicked".into()),
                ..Sample::default()
            });
        }
    }
    samples.sort_by_key(|s| s.index);
    (samples, elapsed)
}
fn repetition(samples: &[Sample], elapsed: f64, index: usize) -> Value {
    let good = samples
        .iter()
        .filter(|s| s.error.is_none())
        .collect::<Vec<_>>();
    let ttft = summary(good.iter().filter_map(|s| s.ttft()).collect());
    let prefill = summary(good.iter().filter_map(|s| s.prefill_rate()).collect());
    let decode = summary(good.iter().filter_map(|s| s.decode_rate()).collect());
    let intervals = summary(good.iter().flat_map(|s| s.intervals()).collect());
    let tokens = good
        .iter()
        .try_fold(0u64, |total, s| total.checked_add(s.completion_tokens?));
    let prompts = good
        .iter()
        .try_fold(0u64, |total, s| total.checked_add(s.prompt_tokens?));
    let valid = good.len() == samples.len() && tokens.is_some() && prompts.is_some();
    json!({"index":index,"elapsed_seconds":elapsed,"requests":samples.iter().map(|s| s.value()).collect::<Vec<_>>(),
        "errors":samples.len()-good.len(),"successful_requests":good.len(),"prompt_tokens":prompts,"completion_tokens":tokens,
        "observed_throughput_tokens_per_second":tokens.and_then(|n| rate(n as f64,elapsed)),
        "metrics":{"throughput_tokens_per_second":if valid { tokens.and_then(|n| rate(n as f64,elapsed)) } else { None },
            "prefill_tokens_per_second":if valid { prefill["median"].clone() } else { Value::Null },
            "decode_tokens_per_second":if valid { decode["median"].clone() } else { Value::Null },
            "ttft_p50_seconds":if valid { ttft["median"].clone() } else { Value::Null },
            "ttft_p95_seconds":if valid { ttft["p95"].clone() } else { Value::Null },
            "content_event_interval_median_ms":if valid { intervals["median"].clone() } else { Value::Null },
            "content_event_interval_mean_ms":if valid { intervals["mean"].clone() } else { Value::Null },
            "content_event_interval_p05_ms":if valid { intervals["p05"].clone() } else { Value::Null },
            "content_event_interval_p95_ms":if valid { intervals["p95"].clone() } else { Value::Null }}})
}

fn discover(
    client: &Client,
    o: &Options,
    warnings: &mut Vec<String>,
) -> Result<(String, Value), String> {
    let mut req = client.get(format!("{}/v1/models", o.root));
    if let Some(key) = &o.api_key {
        req = req.bearer_auth(key);
    }
    let result = req
        .send()
        .and_then(|r| r.error_for_status())
        .and_then(|r| r.json::<Value>());
    match result {
        Ok(v) => {
            let reported = v["data"]
                .as_array()
                .and_then(|a| a.first())
                .and_then(|m| m["id"].as_str());
            let model = o
                .model
                .clone()
                .or_else(|| reported.map(str::to_owned))
                .ok_or("/v1/models returned no model; specify --model")?;
            Ok((model, v))
        }
        Err(e) => {
            warnings.push(format!("model discovery failed: {}", e.without_url()));
            Ok((
                o.model
                    .clone()
                    .ok_or("model discovery failed; specify --model to continue")?,
                Value::Null,
            ))
        }
    }
}

pub fn run(args: &[String]) -> Result<(), String> {
    if args.iter().any(|s| s == "--help" || s == "-h") {
        println!("{HELP}");
        return Ok(());
    }
    let o = parse(args)?;
    let client = Client::builder()
        .timeout(Duration::from_secs(o.timeout))
        .connect_timeout(Duration::from_secs(o.timeout.min(20)))
        .user_agent(format!("fllm-bench/{VERSION}"))
        .build()
        .map_err(|e| e.to_string())?;
    let mut warnings = vec!["SSE content events may contain multiple tokens; their intervals are not guaranteed per-token intervals. Decode rate uses server usage and excludes the first observed event; absent first-event token IDs it subtracts one token.".into(),
        "Host metadata describes the benchmark client. For remote servers, record server hardware, software and launch flags in --notes.".into(),
        "Accepted min_tokens/ignore_eos fields may be ignored by a server. Actual usage and finish_reason remain authoritative.".into()];
    let timestamp = chrono::Utc::now().to_rfc3339();
    let (model, models) = discover(&client, &o, &mut warnings)?;
    let host = host_info();
    let mut control = if o.length_control == "auto" {
        "both".to_owned()
    } else {
        o.length_control.clone()
    };
    let mut negotiate = o.length_control == "auto";
    let mut results = Vec::new();
    let mut any_error = false;
    println!(
        "fLLM benchmark {VERSION}; model {model}; endpoint {}; client-observed streaming timings",
        o.endpoint
    );
    println!(
        "Started {timestamp}; client Git {} (dirty={})",
        env!("FLLM_GIT_COMMIT"),
        env!("FLLM_GIT_DIRTY")
    );
    println!(
        "Client: {} {}; CPU {}; logical cores {}; available cores {}; RAM {} bytes",
        std::env::consts::OS,
        std::env::consts::ARCH,
        host["cpu_model"].as_str().unwrap_or("unknown"),
        host["logical_cores"],
        host["available_parallelism"],
        host["ram_bytes"]
    );
    println!(
        "Client CPU features: {}; GPU inventory: {}",
        host["linux_cpu_features"], host["gpu_inventory"]
    );
    println!("Parameters: {}", parameters(&o));
    println!(
        "{:<46} {:>22} {:>22} {:>22} {:>18} {:>7}",
        "Scenario",
        "Prefill t/s med[min,max]",
        "Decode t/s med[min,max]",
        "Total t/s med[min,max]",
        "TTFT p50/p95 ms",
        "Errors"
    );
    for case in cases(&o) {
        eprintln!("[bench] calibrating {}", case.id);
        let mut words = case.target;
        let mut calibration = Vec::new();
        let mut probe = case.clone();
        probe.gen = 1;
        let mut state = case.seed;
        let mut first = request(
            &client,
            &o,
            &model,
            &probe,
            &control,
            Work {
                words,
                seed: mix(&mut state),
                index: 0,
            },
        );
        if negotiate {
            if matches!(first.status, Some(400 | 422)) {
                warnings.push("Length-control probe rejected with HTTP 400/422; subsequent requests use max_tokens only.".into());
                calibration.push(first.value());
                control = "none".into();
                first = request(
                    &client,
                    &o,
                    &model,
                    &probe,
                    &control,
                    Work {
                        words,
                        seed: mix(&mut state),
                        index: 1,
                    },
                );
            }
            negotiate = false;
        }
        if first.error.is_none() {
            if let Some(actual) = first.prompt_tokens.filter(|n| *n > 0) {
                words = ((words as f64 * case.target as f64 / actual as f64).round() as usize)
                    .clamp(1, 1_000_000);
            }
        } else {
            warnings.push(format!(
                "{} calibration failed; unadjusted word count used: {}",
                case.id,
                first.error.as_deref().unwrap_or_default()
            ));
        }
        calibration.push(first.value());
        let adjusted = request(
            &client,
            &o,
            &model,
            &probe,
            &control,
            Work {
                words,
                seed: mix(&mut state),
                index: 2,
            },
        );
        if let Some(error) = &adjusted.error {
            warnings.push(format!("{} adjusted calibration failed: {error}", case.id));
        }
        calibration.push(adjusted.value());
        let mut warmups = Vec::new();
        for i in 0..o.warmup {
            eprintln!("[bench] {} warmup {}/{}", case.id, i + 1, o.warmup);
            let (samples, elapsed) =
                batch(&client, &o, &model, &case, words, mix(&mut state), &control);
            if samples.iter().any(|s| s.error.is_some()) {
                warnings.push(format!(
                    "{} warmup {} had request errors; see warmups in JSON",
                    case.id,
                    i + 1
                ));
            }
            warmups.push(repetition(&samples, elapsed, i));
        }
        let mut repeats = Vec::new();
        for i in 0..o.repeats {
            eprintln!("[bench] {} repetition {}/{}", case.id, i + 1, o.repeats);
            let (samples, elapsed) =
                batch(&client, &o, &model, &case, words, mix(&mut state), &control);
            for sample in samples.iter().filter(|s| s.error.is_some()) {
                eprintln!(
                    "[bench] {} request {}: {}",
                    case.id,
                    sample.index,
                    sample.error.as_deref().unwrap_or_default()
                );
            }
            repeats.push(repetition(&samples, elapsed, i));
        }
        let mut metrics = serde_json::Map::new();
        for name in [
            "prefill_tokens_per_second",
            "decode_tokens_per_second",
            "throughput_tokens_per_second",
            "ttft_p50_seconds",
            "ttft_p95_seconds",
            "content_event_interval_median_ms",
            "content_event_interval_mean_ms",
            "content_event_interval_p05_ms",
            "content_event_interval_p95_ms",
        ] {
            metrics.insert(
                name.into(),
                summary(
                    repeats
                        .iter()
                        .filter_map(|r| r["metrics"][name].as_f64())
                        .collect(),
                ),
            );
        }
        let errors: u64 = repeats.iter().filter_map(|r| r["errors"].as_u64()).sum();
        any_error |= errors > 0;
        let measured = repeats
            .iter()
            .flat_map(|r| r["requests"].as_array().into_iter().flatten())
            .collect::<Vec<_>>();
        let result = json!({"id":case.id,"scenario":case.kind,"target_prompt_tokens":case.target,"requested_completion_tokens":case.gen,
            "concurrency":case.concurrency,"requests_per_repeat":case.requests,"prompt_words":words,"case_seed":case.seed,
            "shared_prefix_words":if o.shared_prefix && case.kind == "concurrency" { words.saturating_sub(8).min(words*9/10) } else { 0 },
            "calibration":calibration,"warmups":warmups,"repetitions":repeats,"errors":errors,"metrics":metrics,
            "actual_prompt_tokens":summary(measured.iter().filter_map(|s| s["prompt_tokens"].as_f64()).collect()),
            "actual_completion_tokens":summary(measured.iter().filter_map(|s| s["completion_tokens"].as_f64()).collect())});
        println!(
            "{:<46} {:>22} {:>22} {:>22} {:>8}/{:<8} {:>7}",
            case.id,
            display_metric(&result, "prefill_tokens_per_second"),
            display_metric(&result, "decode_tokens_per_second"),
            display_metric(&result, "throughput_tokens_per_second"),
            number(
                result["metrics"]["ttft_p50_seconds"]["median"]
                    .as_f64()
                    .map(|n| n * 1000.0)
            ),
            number(
                result["metrics"]["ttft_p95_seconds"]["median"]
                    .as_f64()
                    .map(|n| n * 1000.0)
            ),
            errors
        );
        println!("  Usage per request: prompt {} [{},{}]; completion {} [{},{}]. Content-event interval ms (repeat medians): median {}, mean {}, p05 {}, p95 {}.",
            number(result["actual_prompt_tokens"]["median"].as_f64()),number(result["actual_prompt_tokens"]["min"].as_f64()),number(result["actual_prompt_tokens"]["max"].as_f64()),
            number(result["actual_completion_tokens"]["median"].as_f64()),number(result["actual_completion_tokens"]["min"].as_f64()),number(result["actual_completion_tokens"]["max"].as_f64()),
            number(result["metrics"]["content_event_interval_median_ms"]["median"].as_f64()),number(result["metrics"]["content_event_interval_mean_ms"]["median"].as_f64()),
            number(result["metrics"]["content_event_interval_p05_ms"]["median"].as_f64()),number(result["metrics"]["content_event_interval_p95_ms"]["median"].as_f64()));
        results.push(result);
    }
    let server_commit = models["git_commit"]
        .as_str()
        .or_else(|| models["commit"].as_str())
        .or_else(|| models.pointer("/data/0/git_commit").and_then(Value::as_str));
    let report = json!({"schema":1,"bench_version":VERSION,"timestamp":timestamp,"metadata":{"server_url":o.root,"requested_model":model,
        "reported_models":models,"server_git_commit":server_commit,"client_git_commit":env!("FLLM_GIT_COMMIT"),"client_git_dirty":env!("FLLM_GIT_DIRTY") == "true","host":host},
        "parameters":parameters(&o),"length_control_effective":control,"warnings":warnings,"results":results});
    if let Some(path) = &o.output {
        let file =
            fs::File::create(path).map_err(|e| format!("cannot create {}: {e}", path.display()))?;
        serde_json::to_writer_pretty(&file, &report).map_err(|e| e.to_string())?;
        (&file).write_all(b"\n").map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
    }
    for warning in report["warnings"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
    {
        eprintln!("[bench] {warning}");
    }
    if any_error {
        Err("benchmark completed with measured request errors; full results were retained".into())
    } else {
        Ok(())
    }
}

fn number(n: Option<f64>) -> String {
    n.map_or_else(|| "n/a".into(), |n| format!("{n:.2}"))
}
fn display_metric(result: &Value, metric: &str) -> String {
    let v = &result["metrics"][metric];
    if v["median"].as_f64().is_none() {
        return "n/a".into();
    }
    format!(
        "{}[{},{}]",
        number(v["median"].as_f64()),
        number(v["min"].as_f64()),
        number(v["max"].as_f64())
    )
}

// Optional inventory commands have bounded execution time and bounded output.
fn command(program: &str, args: &[&str]) -> Option<String> {
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) if status.success() => {
                let mut text = String::new();
                child
                    .stdout
                    .take()?
                    .take(65536)
                    .read_to_string(&mut text)
                    .ok()?;
                return (!text.trim().is_empty()).then(|| text.trim().to_owned());
            }
            Ok(Some(_)) => return None,
            Ok(None) if start.elapsed() < Duration::from_secs(3) => {
                thread::sleep(Duration::from_millis(20))
            }
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
}
fn host_info() -> Value {
    let mut cpu = None::<String>;
    let mut ram = None::<u64>;
    let mut features = BTreeSet::new();
    let mut logical_cores = None::<usize>;
    if cfg!(target_os = "linux") {
        if let Ok(text) = fs::read_to_string("/proc/cpuinfo") {
            logical_cores = Some(
                text.lines()
                    .filter(|s| s.starts_with("processor") && s.contains(':'))
                    .count(),
            );
            for line in text.lines() {
                let Some((key, value)) = line.split_once(':') else {
                    continue;
                };
                let key = key.trim();
                let value = value.trim();
                if ["model name", "Hardware", "Processor"].contains(&key) && cpu.is_none() {
                    cpu = Some(value.into());
                }
                if key == "flags" || key == "Features" {
                    for flag in value.split_whitespace() {
                        if flag == "avx2"
                            || flag.starts_with("avx512")
                            || ["neon", "asimd", "dotprod", "asimddp", "i8mm"].contains(&flag)
                        {
                            features.insert(flag.to_owned());
                            if flag == "asimd" {
                                features.insert("neon".into());
                            }
                        }
                    }
                }
            }
        }
        ram = fs::read_to_string("/proc/meminfo").ok().and_then(|s| {
            s.lines().find_map(|l| {
                l.strip_prefix("MemTotal:").and_then(|l| {
                    l.split_whitespace()
                        .next()?
                        .parse::<u64>()
                        .ok()?
                        .checked_mul(1024)
                })
            })
        });
    } else if cfg!(target_os = "macos") {
        cpu = command("sysctl", &["-n", "machdep.cpu.brand_string"])
            .or_else(|| command("sysctl", &["-n", "hw.model"]));
        ram = command("sysctl", &["-n", "hw.memsize"]).and_then(|s| s.parse().ok());
        logical_cores = command("sysctl", &["-n", "hw.logicalcpu"]).and_then(|s| s.parse().ok());
    } else if cfg!(target_os = "windows") {
        cpu = command(
            "reg",
            &[
                "query",
                r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                "/v",
                "ProcessorNameString",
            ],
        )
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.split_once("REG_SZ").map(|(_, v)| v.trim().to_owned()))
        });
        ram = command(
            "powershell",
            &[
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
            ],
        )
        .and_then(|s| s.parse().ok());
        logical_cores = std::env::var("NUMBER_OF_PROCESSORS")
            .ok()
            .and_then(|s| s.parse().ok());
    }
    let mut gpus = Vec::new();
    for (program, args) in [
        (
            "nvidia-smi",
            vec!["--query-gpu=name", "--format=csv,noheader"],
        ),
        ("rocm-smi", vec!["--showproductname"]),
        ("vulkaninfo", vec!["--summary"]),
    ] {
        if let Some(output) = command(program, &args) {
            gpus.push(json!({"source":program,"output":output}));
        }
    }
    json!({"scope":"benchmark_client","os":std::env::consts::OS,"architecture":std::env::consts::ARCH,
        "cpu_model":cpu,"logical_cores":logical_cores,"available_parallelism":thread::available_parallelism().ok().map(|n| n.get()),
        "ram_bytes":ram,"linux_cpu_features":features,"gpu_inventory":gpus})
}

struct Report {
    file: String,
    data: Value,
}
fn load_report(path: &str) -> Result<Report, String> {
    let file = fs::File::open(path).map_err(|e| format!("read {path}: {e}"))?;
    let data: Value = serde_json::from_reader(BufReader::new(file))
        .map_err(|e| format!("invalid JSON in {path}: {e}"))?;
    if data["schema"].as_u64() != Some(1) {
        return Err(format!("{path}: unsupported benchmark schema (expected 1)"));
    }
    let rows = data["results"]
        .as_array()
        .filter(|r| !r.is_empty())
        .ok_or_else(|| format!("{path}: no results"))?;
    let mut ids = BTreeSet::new();
    for row in rows {
        let id = row["id"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| format!("{path}: result has no id"))?;
        if !ids.insert(id) || !row["metrics"].is_object() {
            return Err(format!("{path}: duplicate id or missing metrics for {id}"));
        }
        if !["prefill", "decode-single", "decode-context", "concurrency"]
            .contains(&row["scenario"].as_str().unwrap_or_default())
        {
            return Err(format!("{path}: unknown scenario for {id}"));
        }
    }
    Ok(Report {
        file: path.into(),
        data,
    })
}
fn markdown(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('|', "&#124;")
        .replace(['\r', '\n'], " ")
}
fn primary(row: &Value) -> &'static str {
    match row["scenario"].as_str() {
        Some("prefill") => "prefill_tokens_per_second",
        Some("concurrency") => "throughput_tokens_per_second",
        _ => "decode_tokens_per_second",
    }
}
fn row<'a>(report: &'a Report, id: &str) -> Option<&'a Value> {
    report.data["results"]
        .as_array()?
        .iter()
        .find(|r| r["id"].as_str() == Some(id))
}
fn comparable(base: &Report, other: &Report) -> Result<(), String> {
    if base.data["bench_version"] != other.data["bench_version"] {
        return Err("benchmark versions differ".into());
    }
    for parameter in [
        "endpoint",
        "scenarios",
        "prompt_lens",
        "context_lens",
        "concurrency",
        "gen_len",
        "input_len",
        "requests",
        "shared_prefix",
        "warmup",
        "repeats",
        "seed",
        "length_control_requested",
        "temperature",
        "top_p",
        "n",
        "stream",
        "include_usage",
        "prompt_generator",
    ] {
        let a = base.data["parameters"].get(parameter);
        let b = other.data["parameters"].get(parameter);
        if a.is_none() || a != b {
            return Err(format!("incompatible or missing parameter: {parameter}"));
        }
    }
    if base.data["length_control_effective"].is_null()
        || base.data["length_control_effective"] != other.data["length_control_effective"]
    {
        return Err("effective length-control modes differ or are missing".into());
    }
    Ok(())
}
fn comparable_lengths(base: &Value, other: &Value) -> Result<(), String> {
    for field in ["actual_prompt_tokens", "actual_completion_tokens"] {
        match (
            base[field]["median"].as_f64(),
            other[field]["median"].as_f64(),
        ) {
            (Some(a), Some(b)) if a > 0.0 && (b / a - 1.0).abs() <= 0.10 => {}
            _ => {
                return Err(format!(
                    "{field} medians missing or differ by more than 10%"
                ))
            }
        }
    }
    Ok(())
}
fn complete_metric(report: &Report, row: &Value, metric: &str) -> Result<f64, String> {
    let id = row["id"].as_str().unwrap_or("?");
    let repeats = report.data["parameters"]["repeats"]
        .as_u64()
        .filter(|n| *n > 0)
        .ok_or("missing repeat count")?;
    if row["errors"].as_u64() != Some(0) {
        return Err(format!(
            "{id}: measured request errors or missing error count"
        ));
    }
    let observations = row["repetitions"]
        .as_array()
        .filter(|a| a.len() as u64 == repeats)
        .ok_or_else(|| format!("{id}: incomplete repetitions"))?;
    if observations
        .iter()
        .any(|r| r["errors"].as_u64() != Some(0) || r["metrics"][metric].as_f64().is_none())
    {
        return Err(format!(
            "{id}: repetition has errors or an unavailable primary metric"
        ));
    }
    let request_count = row["requests_per_repeat"]
        .as_u64()
        .filter(|n| *n > 0)
        .ok_or_else(|| format!("{id}: missing request count"))?;
    for observation in observations {
        let requests = observation["requests"]
            .as_array()
            .filter(|a| a.len() as u64 == request_count)
            .ok_or_else(|| format!("{id}: missing raw requests"))?;
        if requests.iter().any(|s| {
            s["done"].as_bool() != Some(true)
                || !s["error"].is_null()
                || s["prompt_tokens"].as_u64().is_none()
                || s["completion_tokens"].as_u64().is_none()
        }) {
            return Err(format!("{id}: raw request is incomplete or has no usage"));
        }
    }
    let summary = &row["metrics"][metric];
    if summary["count"].as_u64() != Some(repeats) {
        return Err(format!(
            "{id}: primary metric does not cover every repetition"
        ));
    }
    summary["median"]
        .as_f64()
        .filter(|n| n.is_finite() && *n > 0.0)
        .ok_or_else(|| format!("{id}: invalid primary metric"))
}

pub fn report(args: &[String]) -> Result<(), String> {
    if args.iter().any(|s| s == "--help" || s == "-h") {
        println!("{REPORT_HELP}");
        return Ok(());
    }
    let mut paths = Vec::new();
    let mut baseline = None;
    let mut current = None;
    let mut threshold = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            flag @ ("--baseline" | "--current" | "--max-regression") => {
                i += 1;
                let value = args
                    .get(i)
                    .ok_or_else(|| format!("missing value for {flag}"))?;
                match flag {
                    "--baseline" if baseline.is_none() => baseline = Some(value.clone()),
                    "--current" if current.is_none() => current = Some(value.clone()),
                    "--max-regression" if threshold.is_none() => {
                        threshold = Some(
                            value
                                .parse::<f64>()
                                .ok()
                                .filter(|n| n.is_finite() && (0.0..=100.0).contains(n))
                                .ok_or("--max-regression must be between 0 and 100")?,
                        )
                    }
                    _ => return Err(format!("duplicate report option: {flag}")),
                }
            }
            flag if flag.starts_with('-') => return Err(format!("unknown report option: {flag}")),
            path => paths.push(path.to_owned()),
        }
        i += 1;
    }
    let regression = baseline.is_some() || current.is_some() || threshold.is_some();
    if regression {
        if !paths.is_empty() || baseline.is_none() || current.is_none() || threshold.is_none() {
            return Err(REPORT_HELP.into());
        }
        paths = vec![baseline.unwrap_or_default(), current.unwrap_or_default()];
    }
    if paths.len() < 2 {
        return Err(REPORT_HELP.into());
    }
    let reports = paths
        .iter()
        .map(|p| load_report(p))
        .collect::<Result<Vec<_>, _>>()?;
    let base = &reports[0];
    let mut problems = Vec::new();
    println!("Client-observed rates; values are median [min, max] over repetitions. SSE cadence is a content-event proxy.\n");
    print!("| Scenario / metric |");
    for report in &reports {
        let label = report.data["parameters"]["label"]
            .as_str()
            .unwrap_or(&report.file);
        print!(" {} |", markdown(label));
    }
    println!();
    print!("| --- |");
    for _ in &reports {
        print!(" --- |");
    }
    println!();
    let mut all_rows = BTreeMap::new();
    for report in &reports {
        for result in report.data["results"].as_array().into_iter().flatten() {
            all_rows
                .entry(result["id"].as_str().unwrap_or_default().to_owned())
                .or_insert(result);
        }
    }
    for (id, definition) in &all_rows {
        let metric = primary(definition);
        let baseline_value = row(base, id)
            .and_then(|r| r["metrics"][metric]["median"].as_f64())
            .filter(|n| *n > 0.0);
        print!("| {} / {} |", markdown(id), metric);
        for (index, report) in reports.iter().enumerate() {
            let cell = if let Some(r) = row(report, id) {
                let value = r["metrics"][metric]["median"].as_f64();
                let delta = if index == 0 {
                    String::new()
                } else if let Err(e) = comparable(base, report) {
                    format!(" [incomparable: {}]", markdown(&e))
                } else if row(base, id).is_none_or(|r| complete_metric(base, r, metric).is_err())
                    || complete_metric(report, r, metric).is_err()
                {
                    " [incomplete]".into()
                } else if let Some(Err(e)) =
                    row(base, id).map(|base_row| comparable_lengths(base_row, r))
                {
                    format!(" [incomparable: {}]", markdown(&e))
                } else if let (Some(a), Some(b)) = (baseline_value, value) {
                    let change = (b / a - 1.0) * 100.0;
                    format!(
                        " {change:+.2}% [{}]",
                        if change < 0.0 {
                            "red: worse"
                        } else if change > 0.0 {
                            "green: better"
                        } else {
                            "equal"
                        }
                    )
                } else {
                    " [unavailable]".into()
                };
                format!(
                    "{}{delta}; errors {}",
                    display_metric(r, metric),
                    r["errors"]
                        .as_u64()
                        .map_or_else(|| "?".into(), |n| n.to_string())
                )
            } else {
                "missing".into()
            };
            print!(" {cell} |");
        }
        println!();
    }
    if regression {
        let newer = &reports[1];
        if let Err(e) = comparable(base, newer) {
            problems.push(e);
        }
        for definition in base.data["results"].as_array().into_iter().flatten() {
            let id = definition["id"].as_str().unwrap_or_default();
            let metric = primary(definition);
            let Some(new_row) = row(newer, id) else {
                problems.push(format!("missing current case: {id}"));
                continue;
            };
            for field in [
                "scenario",
                "target_prompt_tokens",
                "requested_completion_tokens",
                "concurrency",
                "requests_per_repeat",
                "case_seed",
                "shared_prefix_words",
            ] {
                // Calibrated word counts can differ between tokenizers; shared-prefix
                // length is expressed as a fraction in parameters, not exact words.
                if field == "shared_prefix_words" {
                    continue;
                }
                if definition[field].is_null() || definition[field] != new_row[field] {
                    problems.push(format!("{id}: incompatible case field {field}"));
                }
            }
            match (
                complete_metric(base, definition, metric),
                complete_metric(newer, new_row, metric),
            ) {
                (Ok(a), Ok(b)) => {
                    let loss = (1.0 - b / a) * 100.0;
                    if loss - threshold.unwrap_or(0.0) > 1e-9 {
                        problems.push(format!(
                            "{id}: {metric} regressed {loss:.2}% (limit {:.2}%)",
                            threshold.unwrap_or(0.0)
                        ));
                    }
                }
                (Err(e), _) => problems.push(format!("baseline: {e}")),
                (_, Err(e)) => problems.push(format!("current: {e}")),
            }
            if let Err(e) = comparable_lengths(definition, new_row) {
                problems.push(format!("{id}: workload cannot be compared fairly: {e}"));
            }
        }
        if !problems.is_empty() {
            for p in &problems {
                eprintln!("[bench-report] {p}");
            }
            return Err(format!(
                "benchmark regression check failed ({} issues)",
                problems.len()
            ));
        }
        println!(
            "\nRegression check passed (maximum allowed loss {:.2}%).",
            threshold.unwrap_or(0.0)
        );
    }
    Ok(())
}
