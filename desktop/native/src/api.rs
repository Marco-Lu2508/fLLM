//! The server side: what the window asks an fLLM server over HTTP (`/v1/models`, `/v1/chat/completions` with a stream).

use futures::stream::{self, Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::pin::Pin;

/// The loaded model, as the server lists it.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelInfo {
    pub id: String,
    pub max_context: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WireMessage {
    pub role: String,
    pub content: String,
}

/// One step of a streamed reply.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// Text of the answer.
    Delta(String),
    /// Text of the reasoning (`reasoning_content`), shown apart from the answer.
    Reasoning(String),
    /// Token counts at the end (`stream_options.include_usage`).
    Usage { prompt: u64, completion: u64 },
    /// The server's own timings: tokens per second of the answer, and of reading the prompt.
    Timings { tokens_per_second: f64, prompt_per_second: f64, prompt_tokens: u64 },
    Done,
    Error(String),
}

/// What one `data:` line of the stream means. `None`: a comment, an empty line or something this window does not use.
pub fn parse_sse_line(line: &str) -> Option<Vec<StreamEvent>> {
    let data = line.strip_prefix("data:")?.trim();
    if data.is_empty() {
        return None;
    }
    if data == "[DONE]" {
        return Some(vec![StreamEvent::Done]);
    }
    let v: serde_json::Value = serde_json::from_str(data).ok()?;
    if let Some(err) = v.get("error") {
        let msg = err.get("message").and_then(|m| m.as_str()).or_else(|| err.as_str()).unwrap_or("the server reported an error");
        return Some(vec![StreamEvent::Error(msg.to_string())]);
    }
    let mut out = Vec::new();
    if let Some(delta) = v.pointer("/choices/0/delta") {
        if let Some(r) = delta.get("reasoning_content").and_then(|c| c.as_str()).filter(|s| !s.is_empty()) {
            out.push(StreamEvent::Reasoning(r.to_string()));
        }
        if let Some(c) = delta.get("content").and_then(|c| c.as_str()).filter(|s| !s.is_empty()) {
            out.push(StreamEvent::Delta(c.to_string()));
        }
    }
    if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
        out.push(StreamEvent::Usage {
            prompt: u.get("prompt_tokens").and_then(|n| n.as_u64()).unwrap_or(0),
            completion: u.get("completion_tokens").and_then(|n| n.as_u64()).unwrap_or(0),
        });
    }
    if let Some(t) = v.get("timings") {
        out.push(StreamEvent::Timings {
            tokens_per_second: t.get("predicted_per_second").and_then(|n| n.as_f64()).unwrap_or(0.0),
            prompt_per_second: t.get("prompt_per_second").and_then(|n| n.as_f64()).unwrap_or(0.0),
            prompt_tokens: t.get("prompt_n").and_then(|n| n.as_u64()).unwrap_or(0),
        });
    }
    Some(out)
}

/// Splits received bytes into lines and keeps the unfinished tail for the next chunk. It keeps bytes, not text: a chunk can end in the
/// middle of a multi-byte character, and that character must arrive whole.
#[derive(Default)]
pub struct LineBuffer {
    pending: Vec<u8>,
}

impl LineBuffer {
    pub fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.pending.extend_from_slice(chunk);
        let mut lines = Vec::new();
        while let Some(i) = self.pending.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.pending.drain(..=i).collect();
            lines.push(String::from_utf8_lossy(&line).trim_end_matches(['\r', '\n']).to_string());
        }
        lines
    }
}

pub async fn fetch_model(base: &str) -> Result<ModelInfo, String> {
    let http = reqwest::Client::builder().timeout(std::time::Duration::from_secs(5)).build().map_err(|e| e.to_string())?;
    let v: serde_json::Value = http
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .map_err(|e| format!("no server at {base}: {e}"))?
        .error_for_status()
        .map_err(|e| e.to_string())?
        .json()
        .await
        .map_err(|e| e.to_string())?;
    let first = v.pointer("/data/0").ok_or(crate::models::NO_MODEL)?;
    Ok(ModelInfo {
        id: first.get("id").and_then(|s| s.as_str()).ok_or("the model has no id")?.to_string(),
        max_context: first.get("max_model_len").and_then(|n| n.as_u64()).unwrap_or(4096),
    })
}

type ByteStream = Pin<Box<dyn Stream<Item = reqwest::Result<Vec<u8>>> + Send>>;

/// Starts a chat request and gives the events of its stream. The request is dropped, and the server stops generating, when the stream is dropped.
pub fn chat_stream(base: String, model: String, messages: Vec<WireMessage>) -> impl Stream<Item = StreamEvent> + Send {
    struct State {
        bytes: Option<ByteStream>,
        request: Option<(String, serde_json::Value)>,
        lines: LineBuffer,
        queue: VecDeque<StreamEvent>,
        finished: bool,
    }
    let body = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": true,
        "stream_options": { "include_usage": true },
        "temperature": 0.7,
    });
    let state = State { bytes: None, request: Some((format!("{base}/v1/chat/completions"), body)), lines: LineBuffer::default(), queue: VecDeque::new(), finished: false };
    stream::unfold(state, |mut st| async move {
        loop {
            if let Some(ev) = st.queue.pop_front() {
                if matches!(ev, StreamEvent::Done | StreamEvent::Error(_)) {
                    st.finished = true;
                }
                return Some((ev, st));
            }
            if st.finished {
                return None;
            }
            if let Some((url, body)) = st.request.take() {
                let sent = reqwest::Client::new().post(url).json(&body).send().await;
                match sent {
                    Ok(resp) if resp.status().is_success() => {
                        let s = resp.bytes_stream().map(|r| r.map(|b| b.to_vec()));
                        st.bytes = Some(Box::pin(s));
                    }
                    Ok(resp) => {
                        let status = resp.status();
                        let text = resp.text().await.unwrap_or_default();
                        let msg = serde_json::from_str::<serde_json::Value>(&text)
                            .ok()
                            .and_then(|v| v.pointer("/error/message").and_then(|m| m.as_str()).map(str::to_string))
                            .unwrap_or_else(|| format!("the server answered {status}"));
                        st.queue.push_back(StreamEvent::Error(msg));
                    }
                    Err(e) => st.queue.push_back(StreamEvent::Error(format!("the server could not be reached: {e}"))),
                }
                continue;
            }
            match st.bytes.as_mut()?.next().await {
                Some(Ok(chunk)) => {
                    for line in st.lines.push(&chunk) {
                        if let Some(events) = parse_sse_line(&line) {
                            st.queue.extend(events);
                        }
                    }
                }
                Some(Err(e)) => st.queue.push_back(StreamEvent::Error(format!("the connection broke: {e}"))),
                None => st.queue.push_back(StreamEvent::Done),
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_content_delta_is_text() {
        let ev = parse_sse_line(r#"data: {"choices":[{"index":0,"delta":{"content":"Hallo"}}]}"#).unwrap();
        assert_eq!(ev, vec![StreamEvent::Delta("Hallo".into())]);
    }

    #[test]
    fn reasoning_stays_apart_from_the_answer() {
        let ev = parse_sse_line(r#"data: {"choices":[{"delta":{"reasoning_content":"hm","content":""}}]}"#).unwrap();
        assert_eq!(ev, vec![StreamEvent::Reasoning("hm".into())]);
    }

    #[test]
    fn usage_and_timings_come_at_the_end() {
        let ev = parse_sse_line(
            r#"data: {"choices":[],"usage":{"prompt_tokens":922,"completion_tokens":52},"timings":{"predicted_per_second":5.9,"prompt_per_second":66.0,"prompt_n":922}}"#,
        )
        .unwrap();
        assert_eq!(
            ev,
            vec![
                StreamEvent::Usage { prompt: 922, completion: 52 },
                StreamEvent::Timings { tokens_per_second: 5.9, prompt_per_second: 66.0, prompt_tokens: 922 }
            ]
        );
    }

    #[test]
    fn done_comments_and_junk() {
        assert_eq!(parse_sse_line("data: [DONE]"), Some(vec![StreamEvent::Done]));
        assert_eq!(parse_sse_line(": keep-alive"), None);
        assert_eq!(parse_sse_line(""), None);
        assert_eq!(parse_sse_line("data: {not json"), None);
    }

    #[test]
    fn an_error_object_ends_the_stream_with_its_message() {
        let ev = parse_sse_line(r#"data: {"error":{"message":"context length exceeded"}}"#).unwrap();
        assert_eq!(ev, vec![StreamEvent::Error("context length exceeded".into())]);
    }

    #[test]
    fn lines_are_cut_across_chunks_and_a_multibyte_character_is_not_lost() {
        let mut b = LineBuffer::default();
        assert!(b.push(b"data: {\"a\"").is_empty());
        assert_eq!(b.push(b":1}\r\ndata: x\n"), vec![r#"data: {"a":1}"#.to_string(), "data: x".to_string()]);
        let all = "data: ä\n".as_bytes();
        assert!(b.push(&all[..7]).is_empty());
        assert_eq!(b.push(&all[7..]), vec!["data: ä".to_string()]);
    }
}
