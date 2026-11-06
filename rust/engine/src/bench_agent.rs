//! `fllm-rs bench-agent`: the load of chat front ends and coding agents against any OpenAI-compatible server (fLLM, vLLM, SGLang).
//!
//! `bench-serve` sends independent requests. An agent sends *conversations*: every user has a long system prompt (the tool list) that is the
//! same for all users, and every turn sends the whole history again, the earlier answers included. What a server can reuse between the turns
//! (its prefix cache) decides how long a turn takes, and a benchmark of single prompts cannot show it. This one measures each turn on its own,
//! so that turn 1 (nothing to reuse but the system prompt) and turn 2 and later (the history is reusable) can be read side by side.
//!
//! **Which answers go into the history.** `--answers fixed` (the default) puts a text made from the seed there, the same for every server:
//! all servers see byte-identical prompts, so a difference in the time of a turn is the server and not the text it happened to write.
//! `--answers server` sends back what the server really answered, which is what a client does; then two servers see different prompts as
//! soon as their answers differ, and the comparison says so in its header.
//!
//! **Cold and warm.** The conversations run twice with the same seed: the first pass finds the caches as they are (cold for a fresh server),
//! the second finds what the first left behind (warm). Both are printed and labelled; a number without the label is not comparable.
use crate::bench::{mix, stream_sample, Sample, WORDS};
use crate::bench_serve::{stats, Rec, Stats};
use reqwest::blocking::Client;
use serde_json::{json, Value};
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub const HELP: &str = "usage: fllm-rs bench-agent [OPTIONS]
  --url URL                 server root or /v1 (default http://127.0.0.1:8000)
  --model ID                default: the first model of GET /v1/models
  --api-key KEY             or FLLM_API_KEY; never written to results
  --users N                 conversations that run at the same time (default 8)
  --turns T                 turns per conversation (default 6)
  --system-words W          words of the system prompt shared by all users, the tool list of an agent (default 1500)
  --question-words Q        words of every user message (default 40)
  --answer-tokens A         tokens the server is asked to generate per turn (default 64)
  --think-ms MS             pause of a user between two turns (default 0)
  --answers fixed|server    what the history holds as the earlier answers (default fixed: identical prompts for every server)
  --passes N                times the same conversations are run, the first one cold (default 2)
  --seed N                  seed of all texts (default 42)
  --timeout SECONDS         per request (default 600)
  --json PATH --markdown PATH --label TEXT
A turn is measured from sending the request to the first content event (TTFT) and to the end. The prompt of turn k is the system prompt,
the k questions and the k-1 answers before it; `cached` is what the server reports in usage.prompt_tokens_details.cached_tokens (a server that
does not report it shows 0, which means 'unknown' there).";

#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    pub root: String,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub users: usize,
    pub turns: usize,
    pub system_words: usize,
    pub question_words: usize,
    pub answer_tokens: usize,
    pub think_ms: u64,
    pub fixed_answers: bool,
    pub passes: usize,
    pub seed: u64,
    pub timeout: u64,
    pub json: Option<String>,
    pub markdown: Option<String>,
    pub label: String,
}

pub fn parse(args: &[String]) -> Result<Options, String> {
    let mut o = Options {
        root: "http://127.0.0.1:8000".into(),
        model: None,
        api_key: std::env::var("FLLM_API_KEY").ok().filter(|k| !k.is_empty()),
        users: 8,
        turns: 6,
        system_words: 1500,
        question_words: 40,
        answer_tokens: 64,
        think_ms: 0,
        fixed_answers: true,
        passes: 2,
        seed: 42,
        timeout: 600,
        json: None,
        markdown: None,
        label: String::new(),
    };
    let mut i = 0;
    let value = |i: &mut usize, name: &str| -> Result<String, String> {
        *i += 1;
        args.get(*i).cloned().ok_or_else(|| format!("{name} needs a value"))
    };
    let count = |t: &str, name: &str, min: usize| -> Result<usize, String> {
        let n = t.trim().parse::<usize>().map_err(|_| format!("{name}: '{t}' is not a whole number"))?;
        if n < min {
            return Err(format!("{name} must be at least {min}"));
        }
        Ok(n)
    };
    while i < args.len() {
        let flag = args[i].clone();
        match flag.as_str() {
            "--url" => o.root = value(&mut i, &flag)?.trim_end_matches('/').trim_end_matches("/v1").to_owned(),
            "--model" => o.model = Some(value(&mut i, &flag)?),
            "--api-key" => o.api_key = Some(value(&mut i, &flag)?),
            "--users" => o.users = count(&value(&mut i, &flag)?, &flag, 1)?,
            "--turns" => o.turns = count(&value(&mut i, &flag)?, &flag, 1)?,
            "--system-words" => o.system_words = count(&value(&mut i, &flag)?, &flag, 1)?,
            "--question-words" => o.question_words = count(&value(&mut i, &flag)?, &flag, 1)?,
            "--answer-tokens" => o.answer_tokens = count(&value(&mut i, &flag)?, &flag, 1)?,
            "--think-ms" => o.think_ms = count(&value(&mut i, &flag)?, &flag, 0)? as u64,
            "--answers" => {
                o.fixed_answers = match value(&mut i, &flag)?.as_str() {
                    "fixed" => true,
                    "server" => false,
                    other => return Err(format!("--answers: '{other}' (fixed or server)")),
                }
            }
            "--passes" => o.passes = count(&value(&mut i, &flag)?, &flag, 1)?,
            "--seed" => o.seed = count(&value(&mut i, &flag)?, &flag, 0)? as u64,
            "--timeout" => o.timeout = count(&value(&mut i, &flag)?, &flag, 1)? as u64,
            "--json" => o.json = Some(value(&mut i, &flag)?),
            "--markdown" => o.markdown = Some(value(&mut i, &flag)?),
            "--label" => o.label = value(&mut i, &flag)?,
            "--help" | "-h" => return Err(String::new()),
            other => return Err(format!("unknown option {other}\n{HELP}")),
        }
        i += 1;
    }
    Ok(o)
}

/// `words` words from a stream seeded by `seed`: the same seed gives the same text on every machine and for every server.
fn text(words: usize, seed: u64, tag: &str) -> String {
    let mut state = seed;
    let mut out = format!("{tag} {seed:016x}:");
    for _ in 0..words {
        out.push(' ');
        out.push_str(WORDS[mix(&mut state) as usize % WORDS.len()]);
    }
    out
}

/// The system prompt of every conversation: the same for all users and all passes (a tool list does not change).
pub fn system_prompt(o: &Options) -> String {
    text(o.system_words, o.seed ^ 0x5157, "System")
}

/// The question of `user` in `turn`. Different for every user and turn, the same in every pass.
pub fn question(o: &Options, user: usize, turn: usize) -> String {
    text(o.question_words, o.seed ^ ((user as u64 + 1) << 20) ^ ((turn as u64 + 1) << 40), "Question")
}

/// The answer put into the history in the fixed mode: about `answer_tokens` words, a function of the seed, the user and the turn only.
pub fn fixed_answer(o: &Options, user: usize, turn: usize) -> String {
    text(o.answer_tokens, o.seed ^ ((user as u64 + 1) << 24) ^ ((turn as u64 + 1) << 44) ^ 0xa115, "Answer")
}

/// The messages of `turn` (0 based) of `user`: the system prompt, then question, answer, question, ... up to the new question.
pub fn messages(o: &Options, user: usize, turn: usize, answers: &[String]) -> Value {
    let mut list = vec![json!({"role": "system", "content": system_prompt(o)})];
    for t in 0..=turn {
        list.push(json!({"role": "user", "content": question(o, user, t)}));
        if let Some(answer) = answers.get(t).filter(|_| t < turn) {
            list.push(json!({"role": "assistant", "content": answer}));
        }
    }
    Value::Array(list)
}

/// One measured turn.
#[derive(Debug, Clone)]
pub struct TurnRec {
    pub pass: usize,
    pub user: usize,
    pub turn: usize,
    pub rec: Rec,
    /// `usage.prompt_tokens_details.cached_tokens`, when the server reports it.
    pub cached: u64,
}

fn one_turn(client: &Client, o: &Options, model: &str, messages: Value, run_start: Instant) -> (Rec, u64, String) {
    let body = json!({"model": model, "messages": messages, "max_tokens": o.answer_tokens, "min_tokens": o.answer_tokens, "ignore_eos": true,
                      "temperature": 0, "stream": true, "stream_options": {"include_usage": true}});
    let mut req = client.post(format!("{}/v1/chat/completions", o.root)).json(&body);
    if let Some(key) = &o.api_key {
        req = req.bearer_auth(key);
    }
    let started = Instant::now();
    let offset = run_start.elapsed().as_secs_f64();
    let mut sample = Sample::default();
    let result = stream_sample(req, started, &mut sample);
    sample.elapsed = started.elapsed().as_secs_f64();
    sample.error = result.err();
    let cached = sample.usage.as_ref().and_then(|u| u.pointer("/prompt_tokens_details/cached_tokens")).and_then(Value::as_u64).unwrap_or(0);
    let answer = std::mem::take(&mut sample.text);
    (Rec::from_sample(&sample, offset), cached, answer)
}

/// Runs every conversation of one pass, `users` at a time, each user's turns in order.
pub fn run_pass(client: &Client, o: &Options, model: &str, pass: usize) -> Vec<TurnRec> {
    let records = Mutex::new(Vec::new());
    let run_start = Instant::now();
    std::thread::scope(|scope| {
        for user in 0..o.users {
            let records = &records;
            scope.spawn(move || {
                let mut answers: Vec<String> = Vec::new();
                for turn in 0..o.turns {
                    if turn > 0 && o.think_ms > 0 {
                        std::thread::sleep(Duration::from_millis(o.think_ms));
                    }
                    let (rec, cached, answer) = one_turn(client, o, model, messages(o, user, turn, &answers), run_start);
                    let failed = !rec.ok;
                    records.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(TurnRec { pass, user, turn, rec, cached });
                    if failed {
                        break; // the history of the later turns would not be the one that was asked for
                    }
                    answers.push(if o.fixed_answers { fixed_answer(o, user, turn) } else { answer });
                }
            });
        }
    });
    let mut out = records.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner);
    out.sort_by_key(|r| (r.user, r.turn));
    out
}

/// What is printed for the turns of one pass that share a turn number.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TurnRow {
    pub pass: usize,
    pub turn: usize,
    pub requests: usize,
    pub failed: usize,
    pub prompt_tokens: f64,
    pub cached_share: f64,
    pub ttft_ms: Option<Stats>,
    pub e2e_ms: Option<Stats>,
}

pub fn rows(records: &[TurnRec]) -> Vec<TurnRow> {
    let mut keys: Vec<(usize, usize)> = records.iter().map(|r| (r.pass, r.turn)).collect();
    keys.sort_unstable();
    keys.dedup();
    keys.into_iter()
        .map(|(pass, turn)| {
            let group: Vec<&TurnRec> = records.iter().filter(|r| r.pass == pass && r.turn == turn).collect();
            let ok: Vec<&&TurnRec> = group.iter().filter(|r| r.rec.ok).collect();
            let prompt: f64 = ok.iter().map(|r| r.rec.prompt_tokens as f64).sum();
            let cached: f64 = ok.iter().map(|r| r.cached.min(r.rec.prompt_tokens) as f64).sum();
            TurnRow {
                pass,
                turn,
                requests: group.len(),
                failed: group.len() - ok.len(),
                prompt_tokens: if ok.is_empty() { 0.0 } else { prompt / ok.len() as f64 },
                cached_share: if prompt > 0.0 { cached / prompt } else { 0.0 },
                ttft_ms: stats(ok.iter().filter_map(|r| r.rec.ttft_s.map(|t| t * 1000.0)).collect()),
                e2e_ms: stats(ok.iter().map(|r| r.rec.e2e_s * 1000.0).collect()),
            }
        })
        .collect()
}

fn cell(s: &Option<Stats>, f: impl Fn(&Stats) -> f64) -> String {
    s.as_ref().map_or_else(|| "-".to_owned(), |s| format!("{:.0}", f(s)))
}

pub fn markdown(rows: &[TurnRow], passes: usize) -> String {
    let mut out = String::from("| pass | turn | requests | failed | prompt tokens | cached | TTFT p50 ms | TTFT p99 ms | e2e p50 ms |\n| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    for r in rows {
        let name = if r.pass == 0 { "1 (cold)".to_owned() } else if passes > 1 { format!("{} (warm)", r.pass + 1) } else { format!("{}", r.pass + 1) };
        out.push_str(&format!(
            "| {name} | {} | {} | {} | {:.0} | {:.0} % | {} | {} | {} |\n",
            r.turn + 1,
            r.requests,
            r.failed,
            r.prompt_tokens,
            100.0 * r.cached_share,
            cell(&r.ttft_ms, |s| s.p50),
            cell(&r.ttft_ms, |s| s.p99),
            cell(&r.e2e_ms, |s| s.p50),
        ));
    }
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
    println!(
        "bench-agent: {} model {model}{}: {} users x {} turns, system prompt {} words, question {} words, answer {} tokens, think {} ms, answers {}, {} passes{}",
        o.root,
        if o.label.is_empty() { String::new() } else { format!(" [{}]", o.label) },
        o.users,
        o.turns,
        o.system_words,
        o.question_words,
        o.answer_tokens,
        o.think_ms,
        if o.fixed_answers { "fixed (identical prompts on every server)" } else { "from the server (prompts differ between servers)" },
        o.passes,
        if o.passes > 1 { "; pass 1 is cold, the later ones find what the first left in the caches" } else { "" },
    );
    let mut all = Vec::new();
    for pass in 0..o.passes {
        eprintln!("pass {} of {} ...", pass + 1, o.passes);
        let records = run_pass(&client, &o, &model, pass);
        if let Some(e) = records.iter().find_map(|r| r.rec.error.clone()) {
            eprintln!("note: a request failed: {e}");
        }
        all.extend(records);
    }
    let table = rows(&all);
    let md = markdown(&table, o.passes);
    println!("\n{md}");
    if let Some(path) = &o.markdown {
        std::fs::write(path, &md).map_err(|e| format!("{path}: {e}"))?;
    }
    if let Some(path) = &o.json {
        let doc = json!({"tool": "fllm-rs bench-agent", "label": o.label, "url": o.root, "model": model, "users": o.users, "turns": o.turns,
            "system_words": o.system_words, "question_words": o.question_words, "answer_tokens": o.answer_tokens, "think_ms": o.think_ms,
            "answers": if o.fixed_answers { "fixed" } else { "server" }, "passes": o.passes, "seed": o.seed,
            "rows": table.iter().map(|r| json!({"pass": r.pass + 1, "turn": r.turn + 1, "requests": r.requests, "failed": r.failed,
                "prompt_tokens": r.prompt_tokens, "cached_share": r.cached_share,
                "ttft_ms": r.ttft_ms.as_ref().map(|s| json!({"p50": s.p50, "p99": s.p99, "mean": s.mean, "max": s.max})),
                "e2e_ms": r.e2e_ms.as_ref().map(|s| json!({"p50": s.p50, "p99": s.p99, "mean": s.mean, "max": s.max}))})).collect::<Vec<_>>()});
        std::fs::write(path, serde_json::to_vec_pretty(&doc).map_err(|e| e.to_string())?).map_err(|e| format!("{path}: {e}"))?;
    }
    if all.iter().all(|r| !r.rec.ok) {
        return Err("every request failed".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    fn options(extra: &[&str]) -> Options {
        let mut args: Vec<String> = ["--users", "3", "--turns", "4", "--system-words", "50", "--question-words", "5", "--answer-tokens", "6"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        args.extend(extra.iter().map(|s| (*s).to_owned()));
        parse(&args).unwrap()
    }

    #[test]
    fn the_history_grows_by_one_question_and_one_answer_per_turn_and_starts_the_same_for_everybody() {
        let o = options(&[]);
        let answers: Vec<String> = (0..3).map(|t| fixed_answer(&o, 1, t)).collect();
        let sizes: Vec<usize> = (0..4).map(|t| messages(&o, 1, t, &answers).as_array().unwrap().len()).collect();
        assert_eq!(sizes, vec![2, 4, 6, 8], "system + question; then question, answer, question; ...");
        // every user starts with the same system prompt: that is what a prefix cache can share between conversations
        let a = messages(&o, 0, 0, &[]);
        let b = messages(&o, 2, 0, &[]);
        assert_eq!(a[0], b[0]);
        assert_ne!(a[1], b[1], "but the questions differ");
        // turn k+1 starts with turn k, byte for byte: what a prefix cache needs to reuse the earlier turn
        let t1 = serde_json::to_string(&messages(&o, 1, 1, &answers)).unwrap();
        let t2 = serde_json::to_string(&messages(&o, 1, 2, &answers)).unwrap();
        assert!(t2.starts_with(&t1[..t1.len() - 2]), "turn 3 does not start with turn 2");
    }

    #[test]
    fn fixed_answers_make_the_prompts_of_two_runs_identical() {
        let o = options(&[]);
        let first = messages(&o, 2, 3, &(0..3).map(|t| fixed_answer(&o, 2, t)).collect::<Vec<_>>());
        let second = messages(&o, 2, 3, &(0..3).map(|t| fixed_answer(&o, 2, t)).collect::<Vec<_>>());
        assert_eq!(first, second);
        let other_seed = options(&["--seed", "7"]);
        assert_ne!(messages(&other_seed, 2, 3, &(0..3).map(|t| fixed_answer(&other_seed, 2, t)).collect::<Vec<_>>()), first);
    }

    #[test]
    fn options_are_checked() {
        assert!(parse(&["--users".into(), "0".into()]).is_err());
        assert!(parse(&["--turns".into(), "x".into()]).is_err());
        assert!(parse(&["--answers".into(), "maybe".into()]).is_err());
        assert!(parse(&["--bogus".into()]).is_err());
        let o = parse(&["--answers".into(), "server".into(), "--think-ms".into(), "0".into()]).unwrap();
        assert!(!o.fixed_answers);
        assert_eq!(o.passes, 2);
    }

    /// A server with a prefix cache of its own, as simple as it gets: it remembers the longest prompt prefix (in characters) it has seen for
    /// the same user and reports what it could reuse as `cached_tokens`. Every word counts as one token; a turn takes 10 ms per uncached token.
    fn mock_agent_server() -> (String, Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let log = seen.clone();
        std::thread::spawn(move || {
            let mut workers = Vec::new();
            for stream in listener.incoming() {
                let mut stream = stream.unwrap();
                stream.set_nodelay(true).unwrap();
                let log = log.clone();
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
                    let prompt: String = request["messages"].as_array().unwrap().iter().map(|m| m["content"].as_str().unwrap().to_owned() + "\n").collect();
                    let words = prompt.split_whitespace().count();
                    let mut seen = log.lock().unwrap();
                    let best = seen.iter().map(|old| old.chars().zip(prompt.chars()).take_while(|(a, b)| a == b).count()).max().unwrap_or(0);
                    seen.push(prompt.clone());
                    drop(seen);
                    let cached_words = prompt[..best].split_whitespace().count().saturating_sub(1).min(words);
                    std::thread::sleep(Duration::from_millis(((words - cached_words) as u64).min(400)));
                    write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n").unwrap();
                    for piece in ["a", "b", "c"] {
                        write!(stream, "data: {}\n\n", json!({"model": "mock", "choices": [{"index": 0, "delta": {"content": piece}}]})).unwrap();
                        stream.flush().unwrap();
                    }
                    write!(stream, "data: {}\n\n", json!({"choices": [], "usage": {"prompt_tokens": words, "completion_tokens": 3,
                        "prompt_tokens_details": {"cached_tokens": cached_words}}})).unwrap();
                    write!(stream, "data: [DONE]\n\n").unwrap();
                }));
                if workers.len() >= 256 {
                    break;
                }
            }
        });
        (format!("http://{addr}"), seen)
    }

    #[test]
    fn a_run_against_a_server_with_a_prefix_cache_shows_the_later_turns_faster_and_the_warm_pass_faster_still() {
        let (root, seen) = mock_agent_server();
        let o = parse(&["--url".into(), root, "--users".into(), "2".into(), "--turns".into(), "3".into(), "--system-words".into(), "60".into(),
                        "--question-words".into(), "6".into(), "--answer-tokens".into(), "5".into(), "--passes".into(), "2".into()])
            .unwrap();
        let client = Client::builder().timeout(Duration::from_secs(30)).build().unwrap();
        let model = discover(&client, &o).unwrap();
        let mut all = run_pass(&client, &o, &model, 0);
        all.extend(run_pass(&client, &o, &model, 1));
        assert_eq!(all.len(), 2 * 2 * 3, "users x turns x passes");
        assert!(all.iter().all(|r| r.rec.ok), "{:?}", all.iter().filter_map(|r| r.rec.error.clone()).collect::<Vec<_>>());
        let table = rows(&all);
        let by = |pass: usize, turn: usize| table.iter().find(|r| r.pass == pass && r.turn == turn).unwrap();
        // the cold first turn has nothing to reuse but the other user's system prompt at best; later turns reuse the whole history
        assert!(by(0, 1).cached_share > by(0, 0).cached_share, "turn 2 reuses more than turn 1");
        assert!(by(0, 2).prompt_tokens > by(0, 0).prompt_tokens, "the prompt grows");
        // the warm pass repeats byte-identical prompts: everything but the end is cached
        assert!(by(1, 0).cached_share > 0.9, "warm turn 1: {}", by(1, 0).cached_share);
        assert!(by(1, 0).ttft_ms.as_ref().unwrap().p50 < by(0, 0).ttft_ms.as_ref().unwrap().p50, "the warm first turn is faster than the cold one");
        // the fixed answers are what the server saw, not what it answered ("abc")
        assert!(seen.lock().unwrap().iter().all(|p| !p.contains("\nabc\n")), "a fixed answer was replaced by the server's");
        let md = markdown(&table, 2);
        assert!(md.contains("1 (cold)") && md.contains("2 (warm)"));
    }

    #[test]
    fn the_server_answers_go_into_the_history_when_asked_to() {
        let (root, seen) = mock_agent_server();
        let o = parse(&["--url".into(), root, "--users".into(), "1".into(), "--turns".into(), "2".into(), "--system-words".into(), "10".into(),
                        "--question-words".into(), "3".into(), "--answer-tokens".into(), "3".into(), "--answers".into(), "server".into(),
                        "--passes".into(), "1".into()])
            .unwrap();
        let client = Client::builder().timeout(Duration::from_secs(30)).build().unwrap();
        let all = run_pass(&client, &o, "mock", 0);
        assert!(all.iter().all(|r| r.rec.ok));
        let prompts = seen.lock().unwrap().clone();
        assert!(prompts.iter().any(|p| p.contains("\nabc\n")), "the server's answer (abc) is not in the history of turn 2: {prompts:?}");
    }

    #[test]
    fn a_failed_turn_ends_that_conversation_and_is_counted() {
        let o = parse(&["--url".into(), "http://127.0.0.1:1".into(), "--users".into(), "2".into(), "--turns".into(), "3".into(), "--passes".into(), "1".into()]).unwrap();
        let client = Client::builder().timeout(Duration::from_secs(2)).build().unwrap();
        let all = run_pass(&client, &o, "mock", 0);
        assert_eq!(all.len(), 2, "one failed first turn per user, no later turns");
        let table = rows(&all);
        assert_eq!((table[0].requests, table[0].failed), (2, 2));
    }
}
