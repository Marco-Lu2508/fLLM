//! Delivery of scheduled-task results: besides the page, a task can send its result to a place you read anyway.
//!
//! A task's `deliver` field is one of
//!
//! * `webhook:https://host/path`: a JSON `POST` `{task, ok, text, time}` (Home Assistant, n8n, Matrix bridges ...),
//! * `ntfy:https://ntfy.host/topic`: the text as the body, the task name as the title (a self-hosted ntfy works),
//! * `telegram:<chat id or @channel>`: a message through the Bot API; the bot token is never stored in the task, it is
//!   read from the environment variable `FLLM_TELEGRAM_TOKEN` of the server.
//!
//! The target is set by the owner on this machine (the schedule routes accept requests from this machine only), so
//! private addresses are allowed on purpose: a ntfy or webhook inside the home network is the common case. Errors
//! never contain the bot token (it sits in the request URL, which is removed from them). The request is built by a
//! pure function ([`build`]) so every kind is tested without a network; one test also sends to a local listener.
use serde_json::json;
use std::time::Duration;

pub const TELEGRAM_TOKEN_ENV: &str = "FLLM_TELEGRAM_TOKEN";
const MAX_TARGET_CHARS: usize = 500;
/// The Bot API refuses messages above 4096 characters.
const TELEGRAM_LIMIT: usize = 4000;
const BODY_LIMIT: usize = 20_000;

#[derive(Debug, Clone, PartialEq)]
pub enum Deliver {
    Webhook(String),
    Ntfy(String),
    Telegram(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

fn valid_url(url: &str) -> Result<(), String> {
    if url.chars().count() > MAX_TARGET_CHARS || url.chars().any(char::is_whitespace) {
        return Err("the address has spaces or is too long".into());
    }
    let rest = url.strip_prefix("https://").or_else(|| url.strip_prefix("http://")).ok_or("the address must begin with http:// or https://")?;
    if rest.is_empty() || rest.starts_with('/') {
        return Err("the address has no host".into());
    }
    Ok(())
}

fn valid_chat(chat: &str) -> bool {
    if let Some(channel) = chat.strip_prefix('@') {
        return (5..=64).contains(&channel.len()) && channel.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    }
    let digits = chat.strip_prefix('-').unwrap_or(chat);
    !digits.is_empty() && digits.len() <= 20 && digits.chars().all(|c| c.is_ascii_digit())
}

/// `None` for an empty text (no delivery), the target for a valid one, a reason for anything else.
pub fn parse(text: &str) -> Result<Option<Deliver>, String> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(None);
    }
    let (kind, rest) = text.split_once(':').ok_or("deliver is webhook:URL, ntfy:URL or telegram:CHAT_ID")?;
    match kind {
        "webhook" => valid_url(rest).map(|()| Some(Deliver::Webhook(rest.to_owned()))),
        "ntfy" => valid_url(rest).map(|()| Some(Deliver::Ntfy(rest.to_owned()))),
        "telegram" if valid_chat(rest) => Ok(Some(Deliver::Telegram(rest.to_owned()))),
        "telegram" => Err("the Telegram chat is a number (a group has a minus sign) or @channelname".into()),
        other => Err(format!("unknown delivery '{other}' (webhook, ntfy or telegram)")),
    }
}

/// HTTP header values must be visible ASCII; anything else becomes `?`.
fn ascii_header(text: &str) -> String {
    text.chars().map(|c| if c.is_ascii() && !c.is_ascii_control() { c } else { '?' }).take(120).collect()
}

fn clip(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_owned();
    }
    let mut cut: String = text.chars().take(max.saturating_sub(20)).collect();
    cut.push_str("\n[... cut ...]");
    cut
}

/// The request for one result. `token` is the Telegram bot token (only read for a Telegram target).
pub fn build(target: &Deliver, task_name: &str, ok: bool, text: &str, now: i64, token: Option<&str>) -> Result<Request, String> {
    match target {
        Deliver::Webhook(url) => Ok(Request {
            url: url.clone(),
            headers: vec![("content-type".into(), "application/json".into())],
            body: json!({"task": task_name, "ok": ok, "text": clip(text, BODY_LIMIT), "time": now}).to_string(),
        }),
        Deliver::Ntfy(url) => Ok(Request {
            url: url.clone(),
            headers: vec![
                ("Title".into(), ascii_header(task_name)),
                ("Tags".into(), if ok { "white_check_mark" } else { "warning" }.into()),
                ("Markdown".into(), "yes".into()),
            ],
            body: clip(text, BODY_LIMIT),
        }),
        Deliver::Telegram(chat) => {
            let token = token.filter(|t| !t.trim().is_empty()).ok_or_else(|| format!("set {TELEGRAM_TOKEN_ENV} in the environment of the server to deliver to Telegram"))?;
            if token.chars().any(|c| c.is_whitespace() || c == '/') {
                return Err(format!("{TELEGRAM_TOKEN_ENV} is not a bot token"));
            }
            let head = if ok { format!("{task_name}\n\n") } else { format!("{task_name} (failed)\n\n") };
            Ok(Request {
                url: format!("https://api.telegram.org/bot{token}/sendMessage"),
                headers: vec![("content-type".into(), "application/json".into())],
                body: json!({"chat_id": chat, "text": clip(&format!("{head}{text}"), TELEGRAM_LIMIT), "disable_web_page_preview": true}).to_string(),
            })
        }
    }
}

/// Sends a built request. Redirects are not followed; the URL is removed from errors because it can hold a bot token.
pub fn send(request: &Request) -> Result<(), String> {
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(20))
        .redirect(reqwest::redirect::Policy::none())
        .user_agent("fllm-schedule/0.1")
        .build()
        .map_err(|e| e.without_url().to_string())?;
    let mut builder = client.post(&request.url).body(request.body.clone());
    for (name, value) in &request.headers {
        builder = builder.header(name.as_str(), value.as_str());
    }
    let response = builder.send().map_err(|e| e.without_url().to_string())?;
    let status = response.status();
    if status.is_success() {
        Ok(())
    } else {
        Err(format!("the receiver answered HTTP {status}"))
    }
}

/// Delivers one result and says how it went: `"sent"` or `"failed: reason"` (stored with the result).
pub fn deliver(target: &Deliver, task_name: &str, ok: bool, text: &str, now: i64) -> String {
    let token = std::env::var(TELEGRAM_TOKEN_ENV).ok();
    match build(target, task_name, ok, text, now, token.as_deref()).and_then(|request| send(&request)) {
        Ok(()) => "sent".to_owned(),
        Err(e) => format!("failed: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    #[test]
    fn targets_are_parsed_and_bad_ones_refused() {
        assert_eq!(parse("").unwrap(), None);
        assert_eq!(parse("  ").unwrap(), None);
        assert_eq!(parse("webhook:https://h.example/x").unwrap(), Some(Deliver::Webhook("https://h.example/x".into())));
        assert_eq!(parse("ntfy:http://10.7.0.5:8080/fllm").unwrap(), Some(Deliver::Ntfy("http://10.7.0.5:8080/fllm".into())));
        assert_eq!(parse("telegram:123456").unwrap(), Some(Deliver::Telegram("123456".into())));
        assert_eq!(parse("telegram:-1001234").unwrap(), Some(Deliver::Telegram("-1001234".into())));
        assert_eq!(parse("telegram:@my_channel").unwrap(), Some(Deliver::Telegram("@my_channel".into())));
        for bad in ["webhook", "webhook:ftp://x", "webhook:https://", "webhook:https:///path", "webhook:https://a b", "ntfy:topic", "telegram:abc", "telegram:@x", "telegram:", "mail:a@b.c", "telegram:12 34"] {
            assert!(parse(bad).is_err(), "{bad} must be refused");
        }
        assert!(parse(&format!("webhook:https://h/{}", "a".repeat(600))).is_err());
    }

    #[test]
    fn a_webhook_gets_json_with_the_result() {
        let r = build(&Deliver::Webhook("https://h/x".into()), "Morning", true, "all fine", 42, None).unwrap();
        assert_eq!(r.url, "https://h/x");
        let body: Value = serde_json::from_str(&r.body).unwrap();
        assert_eq!(body, json!({"task": "Morning", "ok": true, "text": "all fine", "time": 42}));
    }

    #[test]
    fn ntfy_gets_the_text_and_an_ascii_title() {
        let r = build(&Deliver::Ntfy("https://n/t".into()), "Täglich\nBericht", false, "Ergebnis: ü", 0, None).unwrap();
        assert_eq!(r.body, "Ergebnis: ü");
        let title = &r.headers.iter().find(|(n, _)| n == "Title").unwrap().1;
        assert_eq!(title, "T?glich?Bericht", "non-ASCII and control characters cannot go into a header");
        assert!(r.headers.iter().any(|(n, v)| n == "Tags" && v == "warning"));
    }

    #[test]
    fn telegram_needs_a_token_and_keeps_it_out_of_the_body() {
        let t = Deliver::Telegram("123".into());
        let err = build(&t, "n", true, "x", 0, None).unwrap_err();
        assert!(err.contains(TELEGRAM_TOKEN_ENV));
        assert!(build(&t, "n", true, "x", 0, Some("  ")).is_err());
        assert!(build(&t, "n", true, "x", 0, Some("12/34")).is_err(), "a token with a slash would change the path");
        let r = build(&t, "Report", true, "hello", 0, Some("123:ABC")).unwrap();
        assert_eq!(r.url, "https://api.telegram.org/bot123:ABC/sendMessage");
        let body: Value = serde_json::from_str(&r.body).unwrap();
        assert_eq!(body["chat_id"], "123");
        assert_eq!(body["text"], "Report\n\nhello");
        assert!(!r.body.contains("ABC"), "the token belongs in the URL only");
        let failed = build(&t, "Report", false, "boom", 0, Some("1:a")).unwrap();
        assert!(serde_json::from_str::<Value>(&failed.body).unwrap()["text"].as_str().unwrap().starts_with("Report (failed)"));
    }

    #[test]
    fn long_results_are_cut_for_telegram_and_marked() {
        let r = build(&Deliver::Telegram("1".into()), "n", true, &"x".repeat(9000), 0, Some("1:a")).unwrap();
        let text = serde_json::from_str::<Value>(&r.body).unwrap()["text"].as_str().unwrap().to_owned();
        assert!(text.chars().count() <= 4096 && text.ends_with("[... cut ...]"), "{}", text.chars().count());
        assert_eq!(clip("short", 100), "short");
    }

    /// One request to a real local listener: method, headers and body arrive as built, and the answer decides the result.
    fn one_shot(status: &'static str) -> (String, std::thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/hook", listener.local_addr().unwrap());
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            let mut got = Vec::new();
            let mut buf = [0u8; 4096];
            loop {
                let n = stream.read(&mut buf).unwrap_or(0);
                got.extend_from_slice(&buf[..n]);
                let text = String::from_utf8_lossy(&got).into_owned();
                if let Some((head, body)) = text.split_once("\r\n\r\n") {
                    let len = head.lines().find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap_or(0))).unwrap_or(0);
                    if body.len() >= len {
                        break;
                    }
                }
                if n == 0 {
                    break;
                }
            }
            stream.write_all(format!("HTTP/1.1 {status}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n").as_bytes()).unwrap();
            String::from_utf8_lossy(&got).into_owned()
        });
        (url, handle)
    }

    #[test]
    fn a_result_really_arrives_at_a_local_receiver() {
        let (url, received) = one_shot("200 OK");
        let outcome = deliver(&Deliver::Webhook(url), "Nightly", true, "report text", 7);
        assert_eq!(outcome, "sent");
        let request = received.join().unwrap();
        assert!(request.starts_with("POST /hook "), "{request}");
        assert!(request.to_ascii_lowercase().contains("content-type: application/json"));
        let body: Value = serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!((body["task"].as_str(), body["text"].as_str(), body["ok"].as_bool()), (Some("Nightly"), Some("report text"), Some(true)));
    }

    #[test]
    fn a_refusing_receiver_and_a_dead_one_are_reported_as_failed() {
        let (url, received) = one_shot("500 Internal Server Error");
        let outcome = deliver(&Deliver::Ntfy(url), "n", true, "x", 0);
        received.join().unwrap();
        assert!(outcome.starts_with("failed:") && outcome.contains("500"), "{outcome}");
        let closed = {
            let l = TcpListener::bind("127.0.0.1:0").unwrap();
            format!("http://{}/", l.local_addr().unwrap())
        };
        let outcome = deliver(&Deliver::Webhook(closed.clone()), "n", true, "x", 0);
        assert!(outcome.starts_with("failed:") && !outcome.contains(&closed), "the address is not echoed: {outcome}");
    }

    #[test]
    fn errors_do_not_leak_the_bot_token() {
        let secret = "999:SECRET-TOKEN";
        let request = Request { url: format!("http://127.0.0.1:1/bot{secret}/sendMessage"), headers: vec![], body: "{}".into() };
        let err = send(&request).unwrap_err();
        assert!(!err.contains("SECRET-TOKEN"), "{err}");
    }
}
