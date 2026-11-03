//! `fllm-rs bench-serve`: throughput benchmark of a running OpenAI-compatible server in the manner of `trtllm-bench` / `vllm bench serve`:
//! input and output length distributions, a request rate (open loop, Poisson arrivals) or a concurrency (closed loop), a sweep over several
//! of either, time to first token, inter-token latency, time per output token and end-to-end percentiles, goodput against service-level
//! objectives, a JSON and a Markdown table, and a suggestion for `--max-seqs` / `--max-batched-tokens` derived from the sweep.
//!
//! Token counts come from the server's `usage` (never estimated). Prompts are made of common words, one word is roughly one token; the
//! mean of the reported `prompt_tokens` is in the results. `bench` (src/bench.rs) is the fixed-scenario benchmark that is also used for
//! engine comparisons; this one is for sizing a deployment.
use crate::bench::{mix, quantile, stream_sample, Sample, WORDS};
use reqwest::blocking::Client;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub const HELP: &str = "usage: fllm-rs bench-serve [OPTIONS]
  --url URL                 server root or /v1 (default http://127.0.0.1:8000)
  --model ID                default: the first model of GET /v1/models
  --api-key KEY             or FLLM_API_KEY; never written to results
  --endpoint chat|completions   (default chat)
  --input-len DIST          prompt length in words (about tokens): N | uniform:LO-HI | normal:MEAN,STD | choices:A,B,C   (default 512)
  --output-len DIST         generated tokens, same forms (default 128)
  --num-prompts N           requests per measured point (default: 8 x concurrency, 16 to 256)
  --concurrency LIST        closed loop: C requests in flight at all times; a list sweeps, e.g. 1,4,16,64
  --request-rate LIST       open loop: Poisson arrivals at R requests per second; a list sweeps; inf sends all at once
  --max-in-flight N         cap on simultaneous requests in open-loop mode (default 1024)
  --slo-ttft-ms MS --slo-tpot-ms MS --slo-e2e-ms MS    goodput counts the requests that meet every given objective
  --warmup N                requests before the first point, not measured (default 2)
  --no-length-control       do not send min_tokens / ignore_eos (the output length is then whatever the model produces)
  --seed N                  prompt and arrival seed (default 42)
  --timeout SECONDS         per request (default 600)
  --suggest                 print --max-seqs / --max-batched-tokens suggested by the sweep
  --json PATH --markdown PATH --label TEXT --details
Without --concurrency and --request-rate one point at concurrency 8 is measured. Definitions: TTFT = first content event; ITL = gap between
content events; TPOT = (end-to-end - TTFT) / (output tokens - 1) per request; goodput = requests that meet all objectives per second.";

#[derive(Clone, Debug, PartialEq)]
pub enum LenDist {
    Fixed(usize),
    Uniform(usize, usize),
    Normal { mean: f64, std: f64 },
    Choices(Vec<usize>),
}

impl LenDist {
    pub fn parse(text: &str) -> Result<Self, String> {
        let text = text.trim();
        let number = |t: &str| t.trim().parse::<usize>().ok().filter(|n| *n > 0).ok_or_else(|| format!("'{t}' is not a positive whole number"));
        if let Some(rest) = text.strip_prefix("uniform:") {
            let (lo, hi) = rest.split_once('-').ok_or("uniform:LO-HI")?;
            let (lo, hi) = (number(lo)?, number(hi)?);
            if lo > hi {
                return Err(format!("uniform:{lo}-{hi}: LO is larger than HI"));
            }
            Ok(Self::Uniform(lo, hi))
        } else if let Some(rest) = text.strip_prefix("normal:") {
            let (mean, std) = rest.split_once(',').ok_or("normal:MEAN,STD")?;
            let mean: f64 = mean.trim().parse().map_err(|_| "normal: MEAN is not a number")?;
            let std: f64 = std.trim().parse().map_err(|_| "normal: STD is not a number")?;
            if mean < 1.0 || std < 0.0 {
                return Err("normal:MEAN,STD needs MEAN >= 1 and STD >= 0".into());
            }
            Ok(Self::Normal { mean, std })
        } else if let Some(rest) = text.strip_prefix("choices:") {
            let list: Vec<usize> = rest.split(',').map(number).collect::<Result<_, _>>()?;
            if list.is_empty() {
                return Err("choices: needs at least one length".into());
            }
            Ok(Self::Choices(list))
        } else {
            Ok(Self::Fixed(number(text)?))
        }
    }

    /// One length from the distribution (at least 1).
    pub fn sample(&self, state: &mut u64) -> usize {
        let unit = |state: &mut u64| (mix(state) >> 11) as f64 / (1u64 << 53) as f64;
        match self {
            Self::Fixed(n) => *n,
            Self::Uniform(lo, hi) => lo + (mix(state) as usize) % (hi - lo + 1),
            Self::Normal { mean, std } => {
                // Box-Muller
                let (u1, u2) = (unit(state).max(1e-12), unit(state));
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                (mean + std * z).round().max(1.0) as usize
            }
            Self::Choices(list) => list[(mix(state) as usize) % list.len()],
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Self::Fixed(n) => n.to_string(),
            Self::Uniform(lo, hi) => format!("uniform {lo}-{hi}"),
            Self::Normal { mean, std } => format!("normal mean {mean}, std {std}"),
            Self::Choices(list) => format!("one of {list:?}"),
        }
    }
}

/// Arrival times in seconds of `n` requests at `rate` per second with exponential gaps (a Poisson process); all zero for an infinite rate.
pub fn poisson_arrivals(n: usize, rate: f64, state: &mut u64) -> Vec<f64> {
    if !rate.is_finite() || rate <= 0.0 {
        return vec![0.0; n];
    }
    let mut t = 0.0;
    (0..n)
        .map(|_| {
            let u = ((mix(state) >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
            t += -u.ln() / rate;
            t
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq)]
pub enum Load {
    Concurrency(usize),
    Rate(f64),
}

impl Load {
    pub fn label(&self) -> String {
        match self {
            Load::Concurrency(c) => format!("concurrency {c}"),
            Load::Rate(r) if r.is_finite() => format!("{r} req/s"),
            Load::Rate(_) => "all at once".to_owned(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Slo {
    pub ttft_ms: Option<f64>,
    pub tpot_ms: Option<f64>,
    pub e2e_ms: Option<f64>,
}

/// What was measured for one request.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Rec {
    pub ok: bool,
    pub error: Option<String>,
    pub prompt_tokens: u64,
    pub output_tokens: u64,
    /// Seconds from the start of the run to the request's start (arrival).
    pub started: f64,
    pub ttft_s: Option<f64>,
    pub e2e_s: f64,
    /// Gaps between consecutive content events, milliseconds.
    pub itl_ms: Vec<f64>,
}

impl Rec {
    pub(crate) fn from_sample(s: &Sample, started: f64) -> Self {
        let ttft = s.events.first().copied();
        let itl_ms = s.events.windows(2).map(|w| (w[1] - w[0]) * 1000.0).collect();
        Self {
            ok: s.error.is_none() && s.done,
            error: s.error.clone(),
            prompt_tokens: s.prompt_tokens.unwrap_or(0),
            output_tokens: s.completion_tokens.unwrap_or(0),
            started,
            ttft_s: ttft,
            e2e_s: s.elapsed,
            itl_ms,
        }
    }

    /// Time per output token (ms) after the first one.
    pub fn tpot_ms(&self) -> Option<f64> {
        let ttft = self.ttft_s?;
        (self.output_tokens > 1).then(|| (self.e2e_s - ttft) * 1000.0 / (self.output_tokens - 1) as f64)
    }

    pub fn meets(&self, slo: &Slo) -> bool {
        self.ok
            && slo.ttft_ms.is_none_or(|m| self.ttft_s.is_some_and(|t| t * 1000.0 <= m))
            && slo.tpot_ms.is_none_or(|m| self.tpot_ms().is_none_or(|t| t <= m))
            && slo.e2e_ms.is_none_or(|m| self.e2e_s * 1000.0 <= m)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stats {
    pub count: usize,
    pub mean: f64,
    pub p50: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
}

pub fn stats(mut values: Vec<f64>) -> Option<Stats> {
    values.retain(|v| v.is_finite());
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let q = |p: f64| quantile(&values, p).unwrap_or(0.0);
    Some(Stats {
        count: values.len(),
        mean: values.iter().sum::<f64>() / values.len() as f64,
        p50: q(0.5),
        p90: q(0.9),
        p95: q(0.95),
        p99: q(0.99),
        max: values[values.len() - 1],
    })
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Summary {
    pub load: String,
    pub requests: usize,
    pub failed: usize,
    /// First arrival to last completion, seconds.
    pub duration_s: f64,
    pub request_throughput: f64,
    pub output_tokens_per_s: f64,
    pub total_tokens_per_s: f64,
    pub mean_prompt_tokens: f64,
    pub mean_output_tokens: f64,
    pub ttft_ms: Option<Stats>,
    pub tpot_ms: Option<Stats>,
    pub itl_ms: Option<Stats>,
    pub e2e_ms: Option<Stats>,
    /// Requests meeting the objectives per second (equals the request throughput without objectives).
    pub goodput: f64,
    pub errors: Vec<String>,
}

pub fn summarize(load: &Load, recs: &[Rec], slo: &Slo) -> Summary {
    let ok: Vec<&Rec> = recs.iter().filter(|r| r.ok).collect();
    let first = recs.iter().map(|r| r.started).fold(f64::INFINITY, f64::min);
    let last = recs.iter().map(|r| r.started + r.e2e_s).fold(0.0, f64::max);
    let duration = if recs.is_empty() { 0.0 } else { (last - first).max(1e-9) };
    let out_tokens: u64 = ok.iter().map(|r| r.output_tokens).sum();
    let in_tokens: u64 = ok.iter().map(|r| r.prompt_tokens).sum();
    let mut errors: Vec<String> = Vec::new();
    for r in recs.iter().filter(|r| !r.ok) {
        let e = r.error.clone().unwrap_or_else(|| "unknown error".into());
        if !errors.contains(&e) && errors.len() < 5 {
            errors.push(e);
        }
    }
    let n_ok = ok.len().max(1) as f64;
    Summary {
        load: load.label(),
        requests: recs.len(),
        failed: recs.len() - ok.len(),
        duration_s: duration,
        request_throughput: ok.len() as f64 / duration,
        output_tokens_per_s: out_tokens as f64 / duration,
        total_tokens_per_s: (out_tokens + in_tokens) as f64 / duration,
        mean_prompt_tokens: in_tokens as f64 / n_ok,
        mean_output_tokens: out_tokens as f64 / n_ok,
        ttft_ms: stats(ok.iter().filter_map(|r| r.ttft_s.map(|t| t * 1000.0)).collect()),
        tpot_ms: stats(ok.iter().filter_map(|r| r.tpot_ms()).collect()),
        itl_ms: stats(ok.iter().flat_map(|r| r.itl_ms.iter().copied()).collect()),
        e2e_ms: stats(ok.iter().map(|r| r.e2e_s * 1000.0).collect()),
        goodput: recs.iter().filter(|r| r.meets(slo)).count() as f64 / duration,
        errors,
    }
}

fn stat_json(s: &Option<Stats>) -> Value {
    match s {
        None => Value::Null,
        Some(s) => json!({"count":s.count,"mean":s.mean,"p50":s.p50,"p90":s.p90,"p95":s.p95,"p99":s.p99,"max":s.max}),
    }
}

impl Summary {
    pub fn to_json(&self) -> Value {
        json!({"load":self.load,"requests":self.requests,"failed":self.failed,"duration_s":self.duration_s,
            "request_throughput":self.request_throughput,"output_tokens_per_s":self.output_tokens_per_s,"total_tokens_per_s":self.total_tokens_per_s,
            "mean_prompt_tokens":self.mean_prompt_tokens,"mean_output_tokens":self.mean_output_tokens,
            "ttft_ms":stat_json(&self.ttft_ms),"tpot_ms":stat_json(&self.tpot_ms),"itl_ms":stat_json(&self.itl_ms),"e2e_ms":stat_json(&self.e2e_ms),
            "goodput_req_per_s":self.goodput,"errors":self.errors})
    }
}

fn cell(s: &Option<Stats>, f: impl Fn(&Stats) -> f64) -> String {
    s.as_ref().map_or_else(|| "-".to_owned(), |s| format!("{:.1}", f(s)))
}

pub fn markdown(rows: &[Summary]) -> String {
    let mut out = String::from(
        "| load | ok/total | req/s | out tok/s | total tok/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 | goodput req/s |\n\
         | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n",
    );
    for r in rows {
        out.push_str(&format!(
            "| {} | {}/{} | {:.2} | {:.1} | {:.1} | {} | {} | {} | {} | {} | {} | {} | {} | {:.2} |\n",
            r.load,
            r.requests - r.failed,
            r.requests,
            r.request_throughput,
            r.output_tokens_per_s,
            r.total_tokens_per_s,
            cell(&r.ttft_ms, |s| s.p50),
            cell(&r.ttft_ms, |s| s.p99),
            cell(&r.tpot_ms, |s| s.p50),
            cell(&r.tpot_ms, |s| s.p99),
            cell(&r.itl_ms, |s| s.p50),
            cell(&r.itl_ms, |s| s.p99),
            cell(&r.e2e_ms, |s| s.p50),
            cell(&r.e2e_ms, |s| s.p99),
            r.goodput
        ));
    }
    out.push_str("\nTimes in milliseconds. TTFT: first content event; TPOT: (end-to-end - TTFT) / (output tokens - 1); ITL: gap between content events.\n");
    out
}

/// What a sweep says about the settings of a server.
#[derive(Clone, Debug, PartialEq)]
pub struct Suggestion {
    pub max_seqs: usize,
    pub max_batched_tokens: usize,
    pub reason: String,
}

/// `--max-seqs` / `--max-batched-tokens` from a concurrency sweep: the smallest concurrency whose goodput reaches 90% of the best point
/// (more sequences only add latency), rounded up to a power of two; the prefill chunk is sized so that one chunk stalls the decoding
/// sequences for about five inter-token gaps, from the prefill speed the lightest point shows. A heuristic from measurements, not a guarantee.
pub fn suggest(points: &[(usize, Summary)]) -> Option<Suggestion> {
    let usable: Vec<&(usize, Summary)> = points.iter().filter(|(_, s)| s.failed == 0 && s.requests > 0).collect();
    if usable.is_empty() {
        return None;
    }
    let best = usable.iter().map(|(_, s)| s.goodput).fold(0.0, f64::max);
    let knee = usable.iter().filter(|(_, s)| s.goodput >= 0.9 * best).map(|(c, _)| *c).min()?;
    let lightest = usable.iter().min_by_key(|(c, _)| *c)?;
    let ttft = lightest.1.ttft_ms.as_ref().map_or(0.0, |s| s.p50) / 1000.0;
    let prefill_rate = if ttft > 0.0 { lightest.1.mean_prompt_tokens / ttft } else { 0.0 };
    let at_knee = usable.iter().find(|(c, _)| *c == knee)?;
    let itl = at_knee.1.itl_ms.as_ref().map_or(0.0, |s| s.p50);
    let chunk = if prefill_rate > 0.0 && itl > 0.0 { prefill_rate * 5.0 * itl / 1000.0 } else { 8192.0 };
    let chunk = ((chunk / 256.0).round() as usize * 256).clamp(512, 32768);
    Some(Suggestion {
        max_seqs: knee.next_power_of_two(),
        max_batched_tokens: chunk,
        reason: format!(
            "goodput peaks at {best:.2} req/s; {knee} concurrent requests reach 90% of it; prefill runs at about {prefill_rate:.0} tokens/s and the median gap between tokens at the knee is {itl:.1} ms"
        ),
    })
}

struct Options {
    root: String,
    model: Option<String>,
    api_key: Option<String>,
    chat: bool,
    input: LenDist,
    output: LenDist,
    num_prompts: Option<usize>,
    concurrency: Vec<usize>,
    rates: Vec<f64>,
    max_in_flight: usize,
    slo: Slo,
    warmup: usize,
    length_control: bool,
    seed: u64,
    timeout: u64,
    suggest: bool,
    json: Option<String>,
    markdown: Option<String>,
    label: String,
    details: bool,
}

fn parse(args: &[String]) -> Result<Options, String> {
    let mut o = Options {
        root: "http://127.0.0.1:8000".into(),
        model: None,
        api_key: std::env::var("FLLM_API_KEY").ok().filter(|k| !k.is_empty()),
        chat: true,
        input: LenDist::Fixed(512),
        output: LenDist::Fixed(128),
        num_prompts: None,
        concurrency: Vec::new(),
        rates: Vec::new(),
        max_in_flight: 1024,
        slo: Slo::default(),
        warmup: 2,
        length_control: true,
        seed: 42,
        timeout: 600,
        suggest: false,
        json: None,
        markdown: None,
        label: String::new(),
        details: false,
    };
    let mut i = 0;
    let value = |i: &mut usize, name: &str| -> Result<String, String> {
        *i += 1;
        args.get(*i).cloned().ok_or_else(|| format!("{name} needs a value"))
    };
    let num = |t: &str, name: &str| -> Result<f64, String> { t.trim().parse::<f64>().map_err(|_| format!("{name}: '{t}' is not a number")) };
    while i < args.len() {
        let flag = args[i].clone();
        match flag.as_str() {
            "--url" => o.root = value(&mut i, &flag)?.trim_end_matches('/').trim_end_matches("/v1").to_owned(),
            "--model" => o.model = Some(value(&mut i, &flag)?),
            "--api-key" => o.api_key = Some(value(&mut i, &flag)?),
            "--endpoint" => {
                o.chat = match value(&mut i, &flag)?.as_str() {
                    "chat" => true,
                    "completions" => false,
                    other => return Err(format!("--endpoint: '{other}' (chat or completions)")),
                }
            }
            "--input-len" => o.input = LenDist::parse(&value(&mut i, &flag)?)?,
            "--output-len" => o.output = LenDist::parse(&value(&mut i, &flag)?)?,
            "--num-prompts" => o.num_prompts = Some(num(&value(&mut i, &flag)?, &flag)? as usize),
            "--concurrency" => {
                o.concurrency = value(&mut i, &flag)?.split(',').map(|t| num(t, "--concurrency").map(|n| n as usize)).collect::<Result<_, _>>()?;
                if o.concurrency.contains(&0) {
                    return Err("--concurrency values must be at least 1".into());
                }
            }
            "--request-rate" => {
                o.rates = value(&mut i, &flag)?
                    .split(',')
                    .map(|t| if t.trim() == "inf" { Ok(f64::INFINITY) } else { num(t, "--request-rate").and_then(|r| if r > 0.0 { Ok(r) } else { Err("--request-rate must be positive".to_owned()) }) })
                    .collect::<Result<_, _>>()?
            }
            "--max-in-flight" => o.max_in_flight = (num(&value(&mut i, &flag)?, &flag)? as usize).max(1),
            "--slo-ttft-ms" => o.slo.ttft_ms = Some(num(&value(&mut i, &flag)?, &flag)?),
            "--slo-tpot-ms" => o.slo.tpot_ms = Some(num(&value(&mut i, &flag)?, &flag)?),
            "--slo-e2e-ms" => o.slo.e2e_ms = Some(num(&value(&mut i, &flag)?, &flag)?),
            "--warmup" => o.warmup = num(&value(&mut i, &flag)?, &flag)? as usize,
            "--no-length-control" => o.length_control = false,
            "--seed" => o.seed = num(&value(&mut i, &flag)?, &flag)? as u64,
            "--timeout" => o.timeout = (num(&value(&mut i, &flag)?, &flag)? as u64).max(1),
            "--suggest" => o.suggest = true,
            "--json" => o.json = Some(value(&mut i, &flag)?),
            "--markdown" => o.markdown = Some(value(&mut i, &flag)?),
            "--label" => o.label = value(&mut i, &flag)?,
            "--details" => o.details = true,
            "--help" | "-h" => return Err(String::new()),
            other => return Err(format!("unknown option {other}\n{HELP}")),
        }
        i += 1;
    }
    if o.concurrency.is_empty() && o.rates.is_empty() {
        o.concurrency = vec![8];
    }
    Ok(o)
}

fn prompt_text(words: usize, seed: u64) -> String {
    let mut state = seed;
    let mut text = format!("Request {seed:016x}: ");
    for i in 0..words {
        if i > 0 {
            text.push(' ');
        }
        text.push_str(WORDS[mix(&mut state) as usize % WORDS.len()]);
    }
    text
}

struct Job {
    words: usize,
    out_len: usize,
    seed: u64,
}

fn jobs(o: &Options, n: usize, state: &mut u64) -> Vec<Job> {
    (0..n).map(|_| Job { words: o.input.sample(state), out_len: o.output.sample(state), seed: mix(state) }).collect()
}

fn one_request(client: &Client, o: &Options, model: &str, job: &Job, run_start: Instant) -> Rec {
    let text = prompt_text(job.words, job.seed);
    let mut body = json!({"model":model,"max_tokens":job.out_len,"temperature":0,"stream":true,"stream_options":{"include_usage":true}});
    if o.chat {
        body["messages"] = json!([{"role":"user","content":text}]);
    } else {
        body["prompt"] = json!(text);
    }
    if o.length_control {
        body["min_tokens"] = json!(job.out_len);
        body["ignore_eos"] = json!(true);
    }
    let url = format!("{}/v1/{}", o.root, if o.chat { "chat/completions" } else { "completions" });
    let mut req = client.post(url).json(&body);
    if let Some(key) = &o.api_key {
        req = req.bearer_auth(key);
    }
    let started = Instant::now();
    let offset = run_start.elapsed().as_secs_f64();
    let mut sample = Sample::default();
    let result = stream_sample(req, started, &mut sample);
    sample.elapsed = started.elapsed().as_secs_f64();
    sample.error = result.err();
    Rec::from_sample(&sample, offset)
}

fn run_point(client: &Client, o: &Options, model: &str, load: &Load, n: usize, state: &mut u64) -> Vec<Rec> {
    let list = jobs(o, n, state);
    let recs = Mutex::new(Vec::with_capacity(n));
    let next = AtomicUsize::new(0);
    let in_flight = AtomicUsize::new(0);
    let run_start = Instant::now();
    std::thread::scope(|scope| match load {
        Load::Concurrency(c) => {
            for _ in 0..(*c).min(n) {
                scope.spawn(|| loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    let Some(job) = list.get(k) else { break };
                    let rec = one_request(client, o, model, job, run_start);
                    recs.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(rec);
                });
            }
        }
        Load::Rate(rate) => {
            let arrivals = poisson_arrivals(n, *rate, state);
            for (job, at) in list.iter().zip(arrivals) {
                let wait = at - run_start.elapsed().as_secs_f64();
                if wait > 0.0 {
                    std::thread::sleep(Duration::from_secs_f64(wait));
                }
                while in_flight.load(Ordering::Relaxed) >= o.max_in_flight {
                    std::thread::sleep(Duration::from_millis(1));
                }
                in_flight.fetch_add(1, Ordering::Relaxed);
                let (recs, in_flight) = (&recs, &in_flight);
                scope.spawn(move || {
                    let rec = one_request(client, o, model, job, run_start);
                    recs.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(rec);
                    in_flight.fetch_sub(1, Ordering::Relaxed);
                });
            }
        }
    });
    let mut out = recs.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner);
    out.sort_by(|a, b| a.started.total_cmp(&b.started));
    out
}

fn discover(client: &Client, o: &Options) -> Result<String, String> {
    if let Some(m) = &o.model {
        return Ok(m.clone());
    }
    let mut req = client.get(format!("{}/v1/models", o.root));
    if let Some(key) = &o.api_key {
        req = req.bearer_auth(key);
    }
    let v: Value = req.send().map_err(|e| format!("cannot reach {}: {}", o.root, e.without_url()))?.json().map_err(|e| format!("GET /v1/models: {e}"))?;
    v["data"][0]["id"].as_str().map(str::to_owned).ok_or_else(|| "the server lists no model; pass --model".to_owned())
}

pub fn run(args: &[String]) -> Result<(), String> {
    let o = match parse(args) {
        Ok(o) => o,
        Err(e) if e.is_empty() => {
            println!("{HELP}");
            return Ok(());
        }
        Err(e) => return Err(e),
    };
    let client = Client::builder().timeout(Duration::from_secs(o.timeout)).build().map_err(|e| e.to_string())?;
    let model = discover(&client, &o)?;
    let mut state = o.seed;
    println!("bench-serve: {} model {model}, input {} words, output {} tokens{}", o.root, o.input.describe(), o.output.describe(), if o.label.is_empty() { String::new() } else { format!(", {}", o.label) });
    for k in 0..o.warmup {
        let job = Job { words: 32, out_len: 8, seed: mix(&mut state) ^ k as u64 };
        let rec = one_request(&client, &o, &model, &job, Instant::now());
        if let Some(e) = rec.error {
            return Err(format!("warm-up request failed: {e}"));
        }
    }
    let mut loads: Vec<Load> = o.concurrency.iter().map(|c| Load::Concurrency(*c)).collect();
    loads.extend(o.rates.iter().map(|r| Load::Rate(*r)));
    let mut rows = Vec::new();
    let mut sweep = Vec::new();
    let mut details = Vec::new();
    for load in &loads {
        let n = o.num_prompts.unwrap_or_else(|| match load {
            Load::Concurrency(c) => (8 * c).clamp(16, 256),
            Load::Rate(r) if r.is_finite() => ((r * 30.0) as usize).clamp(16, 512),
            Load::Rate(_) => 64,
        });
        eprintln!("measuring {} with {n} requests ...", load.label());
        let recs = run_point(&client, &o, &model, load, n, &mut state);
        let summary = summarize(load, &recs, &o.slo);
        if let Load::Concurrency(c) = load {
            sweep.push((*c, summary.clone()));
        }
        if o.details {
            details.push(json!(recs.iter().map(|r| json!({"started":r.started,"ok":r.ok,"error":r.error,"prompt_tokens":r.prompt_tokens,"output_tokens":r.output_tokens,"ttft_s":r.ttft_s,"e2e_s":r.e2e_s})).collect::<Vec<_>>()));
        }
        rows.push(summary);
    }
    let table = markdown(&rows);
    println!("\n{table}");
    for r in rows.iter().filter(|r| r.failed > 0) {
        println!("{}: {} of {} requests failed, e.g. {}", r.load, r.failed, r.requests, r.errors.first().map_or("?", String::as_str));
    }
    let suggestion = if o.suggest { suggest(&sweep) } else { None };
    if o.suggest {
        match &suggestion {
            Some(s) => println!("suggested server settings: --max-seqs {} --max-batched-tokens {}\n  ({})", s.max_seqs, s.max_batched_tokens, s.reason),
            None => println!("no suggestion: it needs a --concurrency sweep without failed requests"),
        }
    }
    if let Some(path) = &o.markdown {
        std::fs::write(path, &table).map_err(|e| format!("{path}: {e}"))?;
    }
    if let Some(path) = &o.json {
        let doc = json!({"tool":"fllm-rs bench-serve","label":o.label,"url":o.root,"model":model,"input_len":o.input.describe(),"output_len":o.output.describe(),
            "length_control":o.length_control,"seed":o.seed,"slo":{"ttft_ms":o.slo.ttft_ms,"tpot_ms":o.slo.tpot_ms,"e2e_ms":o.slo.e2e_ms},
            "points":rows.iter().map(Summary::to_json).collect::<Vec<_>>(),
            "suggestion":suggestion.as_ref().map(|s| json!({"max_seqs":s.max_seqs,"max_batched_tokens":s.max_batched_tokens,"reason":s.reason})),
            "requests":details});
        std::fs::write(path, serde_json::to_vec_pretty(&doc).map_err(|e| e.to_string())?).map_err(|e| format!("{path}: {e}"))?;
    }
    if rows.iter().any(|r| r.failed == r.requests) {
        return Err("every request failed".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;

    #[test]
    fn length_distributions_parse_and_sample_in_range() {
        assert_eq!(LenDist::parse("512").unwrap(), LenDist::Fixed(512));
        assert_eq!(LenDist::parse("uniform:100-200").unwrap(), LenDist::Uniform(100, 200));
        assert_eq!(LenDist::parse("choices:128, 512,2048").unwrap(), LenDist::Choices(vec![128, 512, 2048]));
        assert!(LenDist::parse("uniform:200-100").is_err() && LenDist::parse("0").is_err() && LenDist::parse("normal:5").is_err() && LenDist::parse("abc").is_err());
        let mut state = 7;
        let uniform = LenDist::Uniform(100, 200);
        assert!((0..2000).map(|_| uniform.sample(&mut state)).all(|n| (100..=200).contains(&n)));
        let choices = LenDist::Choices(vec![1, 10]);
        assert!((0..200).map(|_| choices.sample(&mut state)).all(|n| n == 1 || n == 10));
        let normal = LenDist::Normal { mean: 300.0, std: 50.0 };
        let samples: Vec<usize> = (0..5000).map(|_| normal.sample(&mut state)).collect();
        let mean = samples.iter().sum::<usize>() as f64 / samples.len() as f64;
        assert!((mean - 300.0).abs() < 5.0, "mean {mean}");
        assert!(samples.iter().all(|n| *n >= 1));
    }

    #[test]
    fn poisson_arrivals_have_the_requested_rate() {
        let mut state = 99;
        let arrivals = poisson_arrivals(4000, 20.0, &mut state);
        assert!(arrivals.windows(2).all(|w| w[1] >= w[0]));
        let rate = arrivals.len() as f64 / arrivals[arrivals.len() - 1];
        assert!((rate - 20.0).abs() < 1.5, "rate {rate}");
        assert!(poisson_arrivals(5, f64::INFINITY, &mut state).iter().all(|t| *t == 0.0));
    }

    fn rec(started: f64, ttft: f64, e2e: f64, out: u64) -> Rec {
        Rec { ok: true, error: None, prompt_tokens: 100, output_tokens: out, started, ttft_s: Some(ttft), e2e_s: e2e, itl_ms: vec![10.0; out as usize - 1] }
    }

    #[test]
    fn percentiles_and_summary_numbers() {
        let s = stats((1..=100).map(f64::from).collect()).unwrap();
        assert_eq!((s.p50, s.max, s.count), (50.5, 100.0, 100));
        assert!((s.p99 - 99.01).abs() < 1e-9 && (s.mean - 50.5).abs() < 1e-9);
        assert!(stats(vec![]).is_none());
        // 4 requests of 11 tokens: TTFT 0.1 s, then 10 tokens in 1 s -> TPOT 100 ms
        let recs: Vec<Rec> = (0..4).map(|k| rec(f64::from(k), 0.1, 1.1, 11)).collect();
        let sum = summarize(&Load::Concurrency(1), &recs, &Slo::default());
        assert_eq!((sum.requests, sum.failed), (4, 0));
        assert!((sum.duration_s - 4.1).abs() < 1e-9);
        assert!((sum.output_tokens_per_s - 44.0 / 4.1).abs() < 1e-9);
        assert!((sum.tpot_ms.as_ref().unwrap().p50 - 100.0).abs() < 1e-9);
        assert!((sum.ttft_ms.as_ref().unwrap().p50 - 100.0).abs() < 1e-9);
        assert_eq!(sum.goodput, sum.request_throughput);
    }

    #[test]
    fn goodput_counts_only_requests_that_meet_every_objective() {
        let mut recs: Vec<Rec> = (0..10).map(|k| rec(f64::from(k) * 0.1, 0.1, 1.1, 11)).collect();
        recs[0].ttft_s = Some(0.9);
        recs[1].e2e_s = 3.0;
        recs[2].ok = false;
        recs[2].error = Some("HTTP 500".into());
        let slo = Slo { ttft_ms: Some(500.0), tpot_ms: Some(150.0), e2e_ms: Some(2000.0) };
        let sum = summarize(&Load::Rate(10.0), &recs, &slo);
        assert_eq!(sum.failed, 1);
        assert_eq!(sum.errors, ["HTTP 500"]);
        let met = recs.iter().filter(|r| r.meets(&slo)).count();
        assert_eq!(met, 7); // minus the slow first token, the slow end-to-end and the failed request (recs[1]'s TPOT is also over 150 ms)
        assert!((sum.goodput - 7.0 / sum.duration_s).abs() < 1e-9);
        assert!(sum.goodput < sum.request_throughput);
    }

    #[test]
    fn the_suggestion_finds_the_knee_and_sizes_the_prefill_chunk() {
        let point = |c: usize, goodput: f64, itl: f64| {
            let mut s = Summary { requests: 32, goodput, mean_prompt_tokens: 1000.0, ..Summary::default() };
            s.ttft_ms = stats(vec![250.0]); // 1000 tokens in 250 ms = 4000 tokens/s
            s.itl_ms = stats(vec![itl]);
            (c, s)
        };
        let points = vec![point(1, 1.0, 10.0), point(4, 3.5, 12.0), point(16, 6.0, 20.0), point(64, 6.3, 60.0)];
        let s = suggest(&points).unwrap();
        assert_eq!(s.max_seqs, 16); // 6.0 >= 0.9 * 6.3
        // 4000 tok/s * 5 gaps * 20 ms = 400 tokens -> rounded to 256s and clamped to at least 512
        assert_eq!(s.max_batched_tokens, 512);
        let mut failing = points.clone();
        failing[2].1.failed = 3;
        assert_eq!(suggest(&failing).unwrap().max_seqs, 64); // 6.3 is the best usable point, 4 gives 3.5 < 5.67
        assert!(suggest(&[]).is_none());
    }

    #[test]
    fn markdown_has_one_row_per_point() {
        let sum = summarize(&Load::Concurrency(2), &[rec(0.0, 0.1, 1.0, 5)], &Slo::default());
        let text = markdown(&[sum.clone(), sum]);
        assert_eq!(text.lines().filter(|l| l.starts_with("| concurrency 2")).count(), 2);
        assert!(text.contains("goodput req/s") && text.contains("TTFT p99"));
    }

    /// A server that streams `tokens` content events 20 ms apart and a usage chunk, for every request.
    fn mock_server(tokens: usize) -> (String, std::thread::JoinHandle<()>, std::sync::Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let served = std::sync::Arc::new(AtomicUsize::new(0));
        let counter = served.clone();
        let handle = std::thread::spawn(move || {
            listener.set_nonblocking(false).unwrap();
            let mut workers = Vec::new();
            for stream in listener.incoming() {
                let mut stream = stream.unwrap();
                let counter = counter.clone();
                workers.push(std::thread::spawn(move || {
                    let mut reader = BufReader::new(stream.try_clone().unwrap());
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    let mut length = 0usize;
                    loop {
                        let mut h = String::new();
                        reader.read_line(&mut h).unwrap();
                        if h.trim().is_empty() {
                            break;
                        }
                        if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") {
                            length = v.trim().parse().unwrap_or(0);
                        }
                    }
                    let mut body = vec![0u8; length];
                    reader.read_exact(&mut body).unwrap();
                    let request: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
                    if line.starts_with("GET /v1/models") {
                        let text = r#"{"data":[{"id":"mock"}]}"#;
                        write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}", text.len()).unwrap();
                        return;
                    }
                    let n = request["max_tokens"].as_u64().unwrap_or(tokens as u64) as usize;
                    write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n").unwrap();
                    for _ in 0..n {
                        std::thread::sleep(Duration::from_millis(20));
                        write!(stream, "data: {}\n\n", json!({"model":"mock","choices":[{"index":0,"delta":{"content":"x"}}]})).unwrap();
                        stream.flush().unwrap();
                    }
                    write!(stream, "data: {}\n\n", json!({"choices":[],"usage":{"prompt_tokens":40,"completion_tokens":n}})).unwrap();
                    write!(stream, "data: [DONE]\n\n").unwrap();
                    counter.fetch_add(1, Ordering::Relaxed);
                }));
                if workers.len() >= 64 {
                    break;
                }
            }
        });
        (format!("http://{addr}"), handle, served)
    }

    #[test]
    fn a_closed_loop_run_against_a_mock_server_measures_what_it_should() {
        let (root, _server, served) = mock_server(6);
        let o = parse(&["--url".into(), root.clone(), "--input-len".into(), "uniform:10-20".into(), "--output-len".into(), "6".into()]).unwrap();
        let client = Client::builder().timeout(Duration::from_secs(20)).build().unwrap();
        assert_eq!(discover(&client, &o).unwrap(), "mock");
        let mut state = 5;
        let recs = run_point(&client, &o, "mock", &Load::Concurrency(4), 8, &mut state);
        assert_eq!(recs.len(), 8);
        assert!(recs.iter().all(|r| r.ok), "{:?}", recs.iter().filter_map(|r| r.error.clone()).collect::<Vec<_>>());
        assert!(recs.iter().all(|r| r.output_tokens == 6 && r.prompt_tokens == 40));
        let sum = summarize(&Load::Concurrency(4), &recs, &Slo::default());
        // events are 20 ms apart: TTFT >= 20 ms, ITL median close to 20 ms, TPOT about 20 ms
        assert!(sum.ttft_ms.as_ref().unwrap().p50 >= 19.0);
        let itl = sum.itl_ms.as_ref().unwrap().p50;
        assert!((15.0..60.0).contains(&itl), "itl {itl}");
        assert_eq!(served.load(Ordering::Relaxed), 8);
        // four at a time: about two rounds of ~120 ms, far less than eight sequential requests
        assert!(sum.duration_s < 1.5, "{}", sum.duration_s);
    }

    #[test]
    fn an_open_loop_run_sends_requests_at_the_arrival_times() {
        let (root, _server, _served) = mock_server(3);
        let o = parse(&["--url".into(), root, "--output-len".into(), "3".into(), "--request-rate".into(), "50".into()]).unwrap();
        let client = Client::builder().timeout(Duration::from_secs(20)).build().unwrap();
        let mut state = 11;
        let recs = run_point(&client, &o, "mock", &Load::Rate(50.0), 10, &mut state);
        assert_eq!(recs.len(), 10);
        assert!(recs.iter().all(|r| r.ok));
        assert!(recs.windows(2).all(|w| w[1].started >= w[0].started));
        // 10 arrivals at 50/s span about 200 ms; they are not all sent at time zero
        assert!(recs[9].started > 0.05, "{}", recs[9].started);
    }

    #[test]
    fn unreachable_servers_and_bad_options_are_reported() {
        assert!(parse(&["--endpoint".into(), "grpc".into()]).is_err());
        assert!(parse(&["--concurrency".into(), "0".into()]).is_err());
        assert!(parse(&["--bogus".into()]).is_err());
        let o = parse(&[]).unwrap();
        assert_eq!(o.concurrency, vec![8]);
        let o = parse(&["--request-rate".into(), "inf,4".into()]).unwrap();
        assert_eq!(o.rates.len(), 2);
        assert!(o.rates[0].is_infinite());
        let client = Client::builder().timeout(Duration::from_secs(2)).build().unwrap();
        let o = parse(&["--url".into(), "http://127.0.0.1:1".into()]).unwrap();
        assert!(discover(&client, &o).is_err());
    }
}
