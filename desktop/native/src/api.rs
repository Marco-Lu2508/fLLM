//! The server side: what the window asks an fLLM server over HTTP (`/v1/models`, `/v1/chat/completions` with a stream).

use futures::stream::{self, Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::pin::Pin;

/// The loaded model, as the server lists it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelInfo {
    pub id: String,
    pub max_context: u64,
    /// What the model can do with its thinking: nothing, a switch, an effort.
    pub thinking: Thinking,
    pub vision: bool,
    /// Name, architecture, size and template, for the system prompt.
    pub detail: crate::prompts::ModelDetail,
}

impl ModelInfo {
    pub fn new(id: &str, max_context: u64) -> ModelInfo {
        ModelInfo { id: id.into(), max_context, ..Default::default() }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ThinkControl {
    #[default]
    None,
    /// A switch in the chat template (`enable_thinking`), with levels that cut the thinking off after a number of tokens.
    Toggle,
    /// The model takes a reasoning effort.
    Effort,
    /// The model always thinks; nothing to switch.
    Always,
}

/// What the loaded model can do with thinking (`capabilities.thinking` of `/v1/models`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Thinking {
    pub control: ThinkControl,
    pub variable: String,
    pub levels: Vec<String>,
    pub default: String,
}

pub const THINKING_LEVELS: [&str; 4] = ["off", "low", "medium", "high"];

impl Thinking {
    pub fn from_caps(raw: Option<&serde_json::Value>) -> Thinking {
        let Some(raw) = raw else { return Thinking::default() };
        if raw.as_bool() == Some(true) {
            // an older server only said that the template mentions thinking: a plain switch
            return Thinking { control: ThinkControl::Toggle, variable: "enable_thinking".into(), levels: vec!["off".into(), "high".into()], default: "off".into() };
        }
        let Some(o) = raw.as_object() else { return Thinking::default() };
        let supported = o.get("supported").and_then(|b| b.as_bool()).unwrap_or(false);
        let control = match o.get("control").and_then(|s| s.as_str()) {
            Some("toggle") => ThinkControl::Toggle,
            Some("effort") => ThinkControl::Effort,
            Some("always") => ThinkControl::Always,
            Some("none") => ThinkControl::None,
            _ if supported => ThinkControl::Toggle,
            _ => ThinkControl::None,
        };
        let levels: Vec<String> = o.get("levels").and_then(|l| l.as_array()).into_iter().flatten().filter_map(|l| l.as_str()).filter(|l| THINKING_LEVELS.contains(l)).map(str::to_string).collect();
        if !supported || control == ThinkControl::None || levels.is_empty() {
            return Thinking::default();
        }
        let default = o.get("default").and_then(|d| d.as_str()).filter(|d| levels.iter().any(|l| l == d)).map(str::to_string).unwrap_or_else(|| levels[0].clone());
        Thinking { control, variable: o.get("variable").and_then(|v| v.as_str()).unwrap_or_default().to_string(), levels, default }
    }

    pub fn supported(&self) -> bool {
        self.control != ThinkControl::None
    }

    /// The level to use: the saved one when this model has it, else the model's default.
    pub fn level_for(&self, saved: &str) -> String {
        if self.levels.iter().any(|l| l == saved) {
            saved.to_string()
        } else {
            self.default.clone()
        }
    }

    /// The level after `level` in the list (the chip steps through them).
    pub fn next(&self, level: &str) -> String {
        if !self.supported() || self.levels.len() < 2 {
            return level.to_string();
        }
        let i = self.levels.iter().position(|l| l == level).unwrap_or(0);
        self.levels[(i + 1) % self.levels.len()].clone()
    }

    /// The fields of a request that carry the level; only what the model understands.
    pub fn fields(&self, level: &str) -> serde_json::Map<String, serde_json::Value> {
        let mut out = serde_json::Map::new();
        if level.is_empty() {
            return out;
        }
        match self.control {
            ThinkControl::Effort => {
                out.insert("reasoning_effort".into(), level.into());
            }
            ThinkControl::Toggle => {
                if level != "off" {
                    out.insert("reasoning_effort".into(), level.into());
                }
                if !self.variable.is_empty() {
                    out.insert("chat_template_kwargs".into(), serde_json::json!({ self.variable.clone(): level != "off" }));
                }
            }
            ThinkControl::None | ThinkControl::Always => {}
        }
        out
    }

    /// The label of the chip and its help.
    pub fn chip(&self, level: &str) -> (String, String) {
        if self.control == ThinkControl::Always {
            return ("Thinking".into(), "This model always thinks before it answers; there is nothing to switch.".into());
        }
        let name = match level {
            "off" => "Off",
            "low" => "Low",
            "medium" => "Medium",
            "high" => "High",
            _ => "",
        };
        let label = if level == "off" || name.is_empty() { "Thinking".to_string() } else { format!("Thinking: {name}") };
        let budgets = if self.control == ThinkControl::Toggle { " Low and Medium stop the thinking after about 512 and 2048 tokens and let the model answer." } else { "" };
        (label, format!("Thinking: {}. Click to change.{budgets}", if name.is_empty() { "default" } else { name }))
    }
}

/// The settings of a request. `None` leaves the choice to the server (the defaults the model ships with).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChatParams {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    /// More fields of the request, put in as they are: top_k, min_p, penalties, seed, stop, the thinking fields.
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// The JSON of a streamed chat request.
pub fn chat_body(model: &str, messages: &[WireMessage], params: ChatParams) -> serde_json::Value {
    chat_body_with_tools(model, messages, params, None)
}

/// The same with the tools the model may call (the list the server's agent offers, in the OpenAI `tools` format).
pub fn chat_body_with_tools(model: &str, messages: &[WireMessage], params: ChatParams, tools: Option<&serde_json::Value>) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": true,
        "stream_options": { "include_usage": true },
    });
    if let Some(t) = params.temperature {
        body["temperature"] = serde_json::json!(t);
    }
    if let Some(p) = params.top_p {
        body["top_p"] = serde_json::json!(p);
    }
    if let Some(n) = params.max_tokens {
        body["max_tokens"] = serde_json::json!(n);
    }
    for (k, v) in &params.extra {
        body[k.as_str()] = v.clone();
    }
    if let Some(tools) = tools.filter(|t| t.as_array().is_some_and(|a| !a.is_empty())) {
        body["tools"] = tools.clone();
        body["tool_choice"] = serde_json::json!("auto");
    }
    body
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WireMessage {
    pub role: String,
    pub content: String,
    /// On an assistant message: the calls the model made (OpenAI format).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
    /// On a `tool` message: the call it answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl WireMessage {
    pub fn new(role: &str, content: impl Into<String>) -> WireMessage {
        WireMessage { role: role.into(), content: content.into(), tool_calls: None, tool_call_id: None }
    }
}

/// One step of a streamed reply.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// Text of the answer.
    Delta(String),
    /// Text of the reasoning (`reasoning_content`), shown apart from the answer.
    Reasoning(String),
    /// A piece of a tool call: the calls arrive in parts, `index` says which call a part belongs to (the id and the name come with the first).
    ToolCall { index: usize, id: Option<String>, name: Option<String>, arguments: String },
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
        for (i, call) in delta.get("tool_calls").and_then(|c| c.as_array()).into_iter().flatten().enumerate() {
            out.push(StreamEvent::ToolCall {
                index: call.get("index").and_then(|n| n.as_u64()).map(|n| n as usize).unwrap_or(i),
                id: call.get("id").and_then(|s| s.as_str()).filter(|s| !s.is_empty()).map(str::to_string),
                name: call.pointer("/function/name").and_then(|s| s.as_str()).filter(|s| !s.is_empty()).map(str::to_string),
                arguments: call.pointer("/function/arguments").and_then(|s| s.as_str()).unwrap_or_default().to_string(),
            });
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
    let resp = http.get(format!("{base}/v1/models")).send().await.map_err(|e| format!("no server at {base}: {e}"))?;
    // the Studio of an fLLM server that has unloaded its model answers only its own pages: `/v1/models` is not a route then
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Err(crate::models::NO_MODEL.into());
    }
    let v: serde_json::Value = resp.error_for_status().map_err(|e| e.to_string())?.json().await.map_err(|e| e.to_string())?;
    let first = v.pointer("/data/0").ok_or(crate::models::NO_MODEL)?;
    let id = first.get("id").and_then(|s| s.as_str()).ok_or("the model has no id")?.to_string();
    let caps = first.get("capabilities");
    let mut info = ModelInfo {
        max_context: first.get("max_model_len").and_then(|n| n.as_u64()).unwrap_or(4096),
        thinking: Thinking::from_caps(caps.and_then(|c| c.get("thinking"))),
        vision: caps.and_then(|c| c.get("vision")).and_then(|b| b.as_bool()).unwrap_or(false),
        detail: crate::prompts::ModelDetail { name: id.clone(), ..Default::default() },
        id,
    };
    // name, architecture, size and template: for the system prompt; a server without the route just gives less
    if let Ok(resp) = http.get(format!("{base}/api/studio/status")).send().await {
        if let Ok(v) = resp.json::<serde_json::Value>().await {
            if let Some(m) = v.get("model") {
                info.detail = crate::prompts::ModelDetail {
                    name: m.get("name").and_then(|s| s.as_str()).unwrap_or(&info.id).to_string(),
                    arch: m.get("arch").and_then(|s| s.as_str()).unwrap_or_default().to_string(),
                    parameters: m.get("parameters").and_then(|n| n.as_f64()).unwrap_or(0.0),
                    chat_template: m.get("chat_template").and_then(|s| s.as_str()).unwrap_or_default().to_string(),
                    thinks: info.thinking.supported(),
                };
            }
        }
    }
    Ok(info)
}

/// Lets the server open a file dialog (`POST /api/studio/pick-file` with `kind: "file"`) and reads the
/// chosen file there, so an exported assistant can be imported. The answer is the text of the file, or
/// why it could not be read; no timeout, because a dialog waits for the person at the computer.
pub async fn pick_assistant_file(base: String) -> Result<String, String> {
    let http = reqwest::Client::new();
    let resp = http
        .post(format!("{base}/api/studio/pick-file"))
        .json(&serde_json::json!({ "kind": "file" }))
        .send()
        .await
        .map_err(|e| format!("no server at {base}: {e}"))?;
    let status = resp.status();
    let data: serde_json::Value = resp.json().await.unwrap_or(serde_json::Value::Null);
    if !status.is_success() {
        let message = data
            .pointer("/error/message")
            .or_else(|| data.get("error"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| format!("the server answered {status}"));
        return Err(message);
    }
    match data.get("text").and_then(serde_json::Value::as_str) {
        Some(text) => Ok(text.to_string()),
        // the dialog was closed: no file, and no line about it (the Studio is silent then too)
        None if data.get("path").is_some_and(serde_json::Value::is_null) => Err(String::new()),
        None => Err("The server did not send the file.".into()),
    }
}

type ByteStream = Pin<Box<dyn Stream<Item = reqwest::Result<Vec<u8>>> + Send>>;

/// Starts a chat request and gives the events of its stream. The request is dropped, and the server stops generating, when the stream is dropped.
pub fn chat_stream(base: String, model: String, messages: Vec<WireMessage>, params: ChatParams) -> impl Stream<Item = StreamEvent> + Send {
    chat_stream_with_tools(base, model, messages, params, None)
}

/// `chat_stream` with tools the model may call: its calls arrive as `StreamEvent::ToolCall` parts.
pub fn chat_stream_with_tools(base: String, model: String, messages: Vec<WireMessage>, params: ChatParams, tools: Option<serde_json::Value>) -> impl Stream<Item = StreamEvent> + Send {
    struct State {
        bytes: Option<ByteStream>,
        request: Option<(String, serde_json::Value)>,
        lines: LineBuffer,
        queue: VecDeque<StreamEvent>,
        finished: bool,
    }
    let body = chat_body_with_tools(&model, &messages, params, tools.as_ref());
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
    fn automatic_sampling_sends_nothing_and_manual_sampling_sends_what_was_set() {
        let m = [WireMessage::new("user", "Hi")];
        let auto = chat_body("tiny", &m, ChatParams::default());
        assert!(auto.get("temperature").is_none() && auto.get("top_p").is_none() && auto.get("max_tokens").is_none(), "{auto}");
        assert_eq!(auto["stream"], true);
        let manual = chat_body("tiny", &m, ChatParams { temperature: Some(0.25), top_p: Some(0.5), max_tokens: Some(300), ..Default::default() });
        assert_eq!((manual["temperature"].as_f64(), manual["top_p"].as_f64(), manual["max_tokens"].as_u64()), (Some(0.25), Some(0.5), Some(300)));
    }

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
