//! Model Context Protocol client. The Studio can start MCP servers (programs that offer tools over JSON-RPC 2.0 on
//! their standard input and output, one message per line), list their tools and call them. The list of servers is
//! `mcp.json` next to `studio.json`, in the common `{"mcpServers": {"name": {"command", "args", "env"}}}` format.
//!
//! The protocol layer (`Client`) talks to a `Transport`, so it is tested against an in-memory transport; `StdioTransport`
//! is the real one (a child process with reader threads). Everything here is blocking; the routes run it on blocking threads.
use super::*;
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

/// The newest protocol revision this client offers; the version the server answers with is accepted.
const PROTOCOL_VERSION: &str = "2025-11-25";
const DEFAULT_CALL_TIMEOUT_SECS: u64 = 60;
const MAX_CALL_TIMEOUT_SECS: u64 = 3600;
/// Starting a server may download it first (`npx`, `uvx`), so the handshake gets a long time.
const INITIALIZE_TIMEOUT: Duration = Duration::from_secs(120);
const LIST_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_PAGES: usize = 50;
const STDERR_LIMIT: usize = 8 * 1024;
const MAX_BACKOFF_SECS: u64 = 60;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------------------------------------------------------
// Protocol layer
// ---------------------------------------------------------------------------------------------------------------

pub(super) enum Recv {
    Line(String),
    Timeout,
    Closed,
}

/// A line-based duplex channel to one MCP server.
pub(super) trait Transport: Send {
    fn send(&mut self, line: &str) -> Result<(), String>;
    fn recv(&mut self, timeout: Duration) -> Recv;
}

#[derive(Debug, PartialEq)]
pub(super) enum McpError {
    /// The server answered with a JSON-RPC error.
    Rpc(String),
    Timeout,
    /// The connection is gone (the process ended or its pipe broke).
    Closed(String),
}

impl McpError {
    fn message(&self) -> String {
        match self {
            McpError::Rpc(m) => m.clone(),
            McpError::Timeout => "the MCP server did not answer in time".to_owned(),
            McpError::Closed(m) => format!("the MCP server connection is closed: {m}"),
        }
    }
}

pub(super) struct Client<T: Transport> {
    transport: T,
    next_id: u64,
    pub protocol: String,
}

impl<T: Transport> Client<T> {
    pub fn new(transport: T) -> Self {
        Client { transport, next_id: 1, protocol: String::new() }
    }

    fn notify(&mut self, method: &str, params: Value) -> Result<(), McpError> {
        let message = json!({"jsonrpc": "2.0", "method": method, "params": params});
        self.transport.send(&message.to_string()).map_err(McpError::Closed)
    }

    /// Sends a request and waits for its answer. Notifications, noise on the pipe and requests of the server
    /// (`ping` is answered, everything else is refused) are handled on the way.
    pub fn request(&mut self, method: &str, params: Value, timeout: Duration) -> Result<Value, McpError> {
        let id = self.next_id;
        self.next_id += 1;
        let message = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        self.transport.send(&message.to_string()).map_err(McpError::Closed)?;
        let deadline = Instant::now() + timeout;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                let _ = self.notify("notifications/cancelled", json!({"requestId": id, "reason": "timeout"}));
                return Err(McpError::Timeout);
            }
            let line = match self.transport.recv(left) {
                Recv::Line(line) => line,
                Recv::Timeout => continue,
                Recv::Closed => return Err(McpError::Closed("the server ended".to_owned())),
            };
            let Ok(value) = serde_json::from_str::<Value>(line.trim()) else { continue };
            let has_method = value.get("method").and_then(Value::as_str);
            let reply_id = value.get("id").cloned();
            match (has_method, reply_id) {
                (Some(m), Some(rid)) => {
                    let reply = if m == "ping" {
                        json!({"jsonrpc": "2.0", "id": rid, "result": {}})
                    } else {
                        json!({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "method not supported by this client"}})
                    };
                    self.transport.send(&reply.to_string()).map_err(McpError::Closed)?;
                }
                (Some(_), None) => {}
                (None, Some(rid)) if rid == json!(id) => {
                    if let Some(error) = value.get("error") {
                        let text = error.get("message").and_then(Value::as_str).unwrap_or("error without a message");
                        let code = error.get("code").and_then(Value::as_i64).map(|c| format!(" (code {c})")).unwrap_or_default();
                        return Err(McpError::Rpc(format!("{text}{code}")));
                    }
                    return Ok(value.get("result").cloned().unwrap_or(Value::Null));
                }
                _ => {}
            }
        }
    }

    pub fn initialize(&mut self, timeout: Duration) -> Result<(), McpError> {
        let result = self.request(
            "initialize",
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "fllm-studio", "version": env!("CARGO_PKG_VERSION")},
            }),
            timeout,
        )?;
        self.protocol = result
            .get("protocolVersion")
            .and_then(Value::as_str)
            .unwrap_or(PROTOCOL_VERSION)
            .to_owned();
        self.notify("notifications/initialized", json!({}))
    }

    /// All tools of the server, following `nextCursor` pages.
    pub fn list_tools(&mut self, timeout: Duration) -> Result<Vec<Value>, McpError> {
        let mut tools = Vec::new();
        let mut cursor: Option<String> = None;
        for _ in 0..MAX_PAGES {
            let params = match &cursor {
                Some(c) => json!({"cursor": c}),
                None => json!({}),
            };
            let result = self.request("tools/list", params, timeout)?;
            for tool in result.get("tools").and_then(Value::as_array).into_iter().flatten() {
                let Some(name) = tool.get("name").and_then(Value::as_str) else { continue };
                tools.push(json!({
                    "name": name,
                    "description": tool.get("description").and_then(Value::as_str).unwrap_or_default(),
                    "input_schema": tool.get("inputSchema").cloned().unwrap_or_else(|| json!({"type": "object"})),
                }));
            }
            match result.get("nextCursor").and_then(Value::as_str) {
                Some(next) if !next.is_empty() => cursor = Some(next.to_owned()),
                _ => break,
            }
        }
        Ok(tools)
    }

    /// The resources the server publishes (`resources/list`). A server without support answers with an error, and that
    /// is not a failure: an empty list is the honest answer.
    pub fn list_resources(&mut self, timeout: Duration) -> Result<Vec<Value>, McpError> {
        let mut out = Vec::new();
        let mut cursor: Option<String> = None;
        for _ in 0..MAX_PAGES {
            let params = match &cursor {
                Some(c) => json!({"cursor": c}),
                None => json!({}),
            };
            let result = match self.request("resources/list", params, timeout) {
                Ok(value) => value,
                Err(e) => {
                    let text = e.message();
                    if text.contains("-32601") || text.to_ascii_lowercase().contains("method not found") {
                        return Ok(Vec::new());
                    }
                    return Err(e);
                }
            };
            for item in result.get("resources").and_then(Value::as_array).into_iter().flatten() {
                let Some(uri) = item.get("uri").and_then(Value::as_str) else { continue };
                out.push(json!({
                    "uri": uri,
                    "name": item.get("name").and_then(Value::as_str).unwrap_or(uri),
                    "description": item.get("description").and_then(Value::as_str).unwrap_or_default(),
                    "mime_type": item.get("mimeType").and_then(Value::as_str).unwrap_or_default(),
                }));
            }
            match result.get("nextCursor").and_then(Value::as_str) {
                Some(next) if !next.is_empty() => cursor = Some(next.to_owned()),
                _ => break,
            }
        }
        Ok(out)
    }

    /// The text of one resource (`resources/read`); a binary body is noted, not guessed.
    pub fn read_resource(&mut self, uri: &str, timeout: Duration) -> Result<String, McpError> {
        let result = self.request("resources/read", json!({"uri": uri}), timeout)?;
        let mut parts: Vec<String> = Vec::new();
        for item in result.get("contents").and_then(Value::as_array).into_iter().flatten() {
            if let Some(text) = item.get("text").and_then(Value::as_str) {
                parts.push(text.to_owned());
            } else if let Some(blob) = item.get("blob").and_then(Value::as_str) {
                parts.push(format!("[binary content, {} bytes base64]", blob.len()));
            }
        }
        Ok(if parts.is_empty() { "(the resource is empty)".to_owned() } else { parts.join("\n") })
    }

    pub fn call_tool(&mut self, name: &str, arguments: Value, timeout: Duration) -> Result<(String, bool), McpError> {
        let result = self.request("tools/call", json!({"name": name, "arguments": arguments}), timeout)?;
        Ok(result_text(&result))
    }
}

/// The text of a `tools/call` result: text blocks joined by newlines, images and audio noted, resources by their text.
fn result_text(result: &Value) -> (String, bool) {
    let mut parts: Vec<String> = Vec::new();
    for block in result.get("content").and_then(Value::as_array).into_iter().flatten() {
        match block.get("type").and_then(Value::as_str).unwrap_or_default() {
            "text" => parts.push(block.get("text").and_then(Value::as_str).unwrap_or_default().to_owned()),
            "image" => parts.push("[image]".to_owned()),
            "audio" => parts.push("[audio]".to_owned()),
            "resource" => {
                let resource = block.get("resource");
                match resource.and_then(|r| r.get("text")).and_then(Value::as_str) {
                    Some(text) => parts.push(text.to_owned()),
                    None => parts.push(format!(
                        "[resource {}]",
                        resource.and_then(|r| r.get("uri")).and_then(Value::as_str).unwrap_or_default()
                    )),
                }
            }
            "resource_link" => parts.push(format!("[resource {}]", block.get("uri").and_then(Value::as_str).unwrap_or_default())),
            other => parts.push(format!("[{other}]")),
        }
    }
    if parts.is_empty() {
        if let Some(structured) = result.get("structuredContent") {
            parts.push(structured.to_string());
        }
    }
    (parts.join("\n"), result.get("isError").and_then(Value::as_bool).unwrap_or(false))
}

// ---------------------------------------------------------------------------------------------------------------
// Stdio transport
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub(super) struct ServerConfig {
    pub command: String,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
    /// Set for a remote server: the MCP endpoint (Streamable HTTP). Exactly one of `command` and `url` is set.
    pub url: Option<String>,
    /// Extra HTTP headers for a remote server; `${NAME}` is expanded from the environment when the request is made.
    pub headers: BTreeMap<String, String>,
    pub timeout: Duration,
}

impl ServerConfig {
    fn from_value(value: &Value) -> Result<Self, String> {
        let command = value.get("command").and_then(Value::as_str).unwrap_or_default().trim().to_owned();
        let url = value
            .get("url")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|u| !u.is_empty())
            .map(str::to_owned);
        if let Some(url) = &url {
            if !(url.starts_with("http://") || url.starts_with("https://")) {
                return Err("the url of a remote MCP server must start with http:// or https://".to_owned());
            }
        }
        if url.is_none() && command.is_empty() {
            return Err("either a command (a local server) or a url (a remote server) is needed".to_owned());
        }
        let args = match value.get("args") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(items)) => items
                .iter()
                .map(|a| a.as_str().map(str::to_owned).ok_or("every argument must be a string"))
                .collect::<Result<Vec<_>, _>>()?,
            Some(_) => return Err("args must be a list of strings".to_owned()),
        };
        let env = match value.get("env") {
            None | Some(Value::Null) => BTreeMap::new(),
            Some(Value::Object(map)) => map
                .iter()
                .map(|(k, v)| v.as_str().map(|v| (k.clone(), v.to_owned())).ok_or("every environment value must be a string"))
                .collect::<Result<_, _>>()?,
            Some(_) => return Err("env must be an object of strings".to_owned()),
        };
        let headers = match value.get("headers") {
            None | Some(Value::Null) => BTreeMap::new(),
            Some(Value::Object(map)) => map
                .iter()
                .map(|(k, v)| v.as_str().map(|v| (k.clone(), v.to_owned())).ok_or("every header value must be a string"))
                .collect::<Result<_, _>>()?,
            Some(_) => return Err("headers must be an object of strings".to_owned()),
        };
        let seconds = value
            .get("timeout_seconds")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_CALL_TIMEOUT_SECS)
            .clamp(1, MAX_CALL_TIMEOUT_SECS);
        Ok(ServerConfig { command, args, env, url, headers, timeout: Duration::from_secs(seconds) })
    }
}

/// Keeps only the last `STDERR_LIMIT` bytes of the text of a server's error output.
fn push_bounded(buffer: &mut String, text: &str) {
    buffer.push_str(text);
    if buffer.len() > STDERR_LIMIT {
        let mut cut = buffer.len() - STDERR_LIMIT;
        while !buffer.is_char_boundary(cut) {
            cut += 1;
        }
        buffer.drain(..cut);
    }
}

pub(super) struct StdioTransport {
    child: Child,
    stdin: ChildStdin,
    lines: mpsc::Receiver<String>,
}

fn build_command(config: &ServerConfig) -> Command {
    let lower = config.command.to_ascii_lowercase();
    // On Windows `npx`, `uvx` and friends are .cmd shims that only the command interpreter can start.
    let shim = cfg!(windows) && !lower.ends_with(".exe") && !lower.ends_with(".com");
    let mut cmd = if shim {
        let mut c = Command::new("cmd");
        c.arg("/c").arg(&config.command);
        c
    } else {
        Command::new(&config.command)
    };
    cmd.args(&config.args).envs(&config.env);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    cmd
}

impl StdioTransport {
    fn spawn(config: &ServerConfig, stderr_log: Arc<Mutex<String>>) -> Result<Self, String> {
        let mut child = build_command(config)
            .spawn()
            .map_err(|e| format!("cannot start `{}`: {e}", config.command))?;
        let stdin = child.stdin.take().ok_or("no standard input")?;
        let stdout = child.stdout.take().ok_or("no standard output")?;
        let stderr = child.stderr.take().ok_or("no standard error")?;
        let (tx, lines) = mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut buf = Vec::new();
            loop {
                buf.clear();
                match reader.read_until(b'\n', &mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        if tx.send(String::from_utf8_lossy(&buf).into_owned()).is_err() {
                            break;
                        }
                    }
                }
            }
        });
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stderr);
            let mut buf = Vec::new();
            loop {
                buf.clear();
                match reader.read_until(b'\n', &mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => push_bounded(&mut lock(&stderr_log), &String::from_utf8_lossy(&buf)),
                }
            }
        });
        Ok(StdioTransport { child, stdin, lines })
    }
}

impl Transport for StdioTransport {
    fn send(&mut self, line: &str) -> Result<(), String> {
        self.stdin
            .write_all(line.as_bytes())
            .and_then(|()| self.stdin.write_all(b"\n"))
            .and_then(|()| self.stdin.flush())
            .map_err(|e| e.to_string())
    }

    fn recv(&mut self, timeout: Duration) -> Recv {
        match self.lines.recv_timeout(timeout) {
            Ok(line) => Recv::Line(line),
            Err(RecvTimeoutError::Timeout) => Recv::Timeout,
            Err(RecvTimeoutError::Disconnected) => Recv::Closed,
        }
    }
}

impl Drop for StdioTransport {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

// ---------------------------------------------------------------------------------------------------------------
// HTTP transport (MCP "Streamable HTTP": one POST per message, the answer as JSON or as an event stream)
// ---------------------------------------------------------------------------------------------------------------

/// Expands `${NAME}` to the value of the environment variable, so a token can live in the environment instead of
/// `mcp.json`. A missing variable becomes an empty string (the server then answers with its own 401, which is clearer
/// than a config error here).
pub(super) fn expand_env(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("${") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        match after.find('}') {
            Some(end) => {
                let name = &after[..end];
                out.push_str(&std::env::var(name).unwrap_or_default());
                rest = &after[end + 1..];
            }
            None => {
                out.push_str(&rest[start..]);
                return out;
            }
        }
    }
    out.push_str(rest);
    out
}

/// A remote MCP server reached over HTTP. Every message is one POST; the server answers with a single JSON message, an
/// array of them, or an SSE stream whose `data:` lines are the messages. The `Mcp-Session-Id` the server hands out on the
/// first answer is sent with every later request.
pub(super) struct HttpTransport {
    url: String,
    headers: BTreeMap<String, String>,
    http: reqwest::blocking::Client,
    session: Option<String>,
    queue: std::collections::VecDeque<String>,
}

impl HttpTransport {
    fn new(config: &ServerConfig) -> Result<Self, String> {
        let url = config.url.clone().ok_or_else(|| "the url is missing".to_owned())?;
        let http = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(120))
            .user_agent(format!("fllm-studio/{}", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| format!("cannot build the HTTP client: {e}"))?;
        Ok(HttpTransport { url, headers: config.headers.clone(), http, session: None, queue: std::collections::VecDeque::new() })
    }
}

impl Transport for HttpTransport {
    fn send(&mut self, line: &str) -> Result<(), String> {
        let mut request = self
            .http
            .post(&self.url)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .header(reqwest::header::ACCEPT, "application/json, text/event-stream")
            .body(line.to_owned());
        for (name, value) in &self.headers {
            request = request.header(name.as_str(), expand_env(value));
        }
        if let Some(session) = &self.session {
            request = request.header("mcp-session-id", session.clone());
        }
        let response = request.send().map_err(|e| format!("the MCP server did not answer: {e}"))?;
        let status = response.status();
        if let Some(session) = response.headers().get("mcp-session-id").and_then(|v| v.to_str().ok()) {
            self.session = Some(session.to_owned());
        }
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_ascii_lowercase();
        let body = response.text().map_err(|e| format!("cannot read the answer of the MCP server: {e}"))?;
        if !status.is_success() {
            let detail: String = body.trim().chars().take(300).collect();
            return Err(format!("the MCP server answered {status}: {detail}"));
        }
        if body.trim().is_empty() {
            return Ok(()); // 202 Accepted: a notification has no answer
        }
        if content_type.contains("text/event-stream") {
            for raw in body.lines() {
                if let Some(data) = raw.trim().strip_prefix("data:") {
                    let data = data.trim();
                    if !data.is_empty() {
                        self.queue.push_back(data.to_owned());
                    }
                }
            }
            return Ok(());
        }
        match serde_json::from_str::<Value>(&body) {
            Ok(Value::Array(items)) => {
                for item in items {
                    self.queue.push_back(item.to_string());
                }
                Ok(())
            }
            Ok(other) => {
                self.queue.push_back(other.to_string());
                Ok(())
            }
            Err(e) => Err(format!("the MCP server sent something that is not JSON: {e}")),
        }
    }

    fn recv(&mut self, timeout: Duration) -> Recv {
        if let Some(line) = self.queue.pop_front() {
            return Recv::Line(line);
        }
        // the answer arrived with the POST: a short wait keeps `Client::request` from spinning until its deadline
        std::thread::sleep(timeout.min(Duration::from_millis(25)));
        Recv::Timeout
    }
}

/// Either transport, so one `Client` type serves local and remote servers.
pub(super) enum AnyTransport {
    Stdio(StdioTransport),
    Http(HttpTransport),
}

impl Transport for AnyTransport {
    fn send(&mut self, line: &str) -> Result<(), String> {
        match self {
            AnyTransport::Stdio(t) => t.send(line),
            AnyTransport::Http(t) => t.send(line),
        }
    }

    fn recv(&mut self, timeout: Duration) -> Recv {
        match self {
            AnyTransport::Stdio(t) => t.recv(timeout),
            AnyTransport::Http(t) => t.recv(timeout),
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Server registry
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
enum State {
    Starting,
    Ready,
    Failed,
}

impl State {
    fn name(self) -> &'static str {
        match self {
            State::Starting => "starting",
            State::Ready => "ready",
            State::Failed => "failed",
        }
    }
}

struct Status {
    state: State,
    message: String,
    tools: Vec<Value>,
    failures: u32,
    retry_at: Instant,
    protocol: String,
}

struct Entry {
    name: String,
    raw: Value,
    config: Result<ServerConfig, String>,
    status: Mutex<Status>,
    client: Mutex<Option<Client<AnyTransport>>>,
    stderr: Arc<Mutex<String>>,
}

fn backoff(failures: u32) -> Duration {
    Duration::from_secs((1u64 << failures.min(6)).min(MAX_BACKOFF_SECS))
}

impl Entry {
    fn new(name: &str, raw: Value) -> Arc<Entry> {
        let config = ServerConfig::from_value(&raw);
        let (state, message) = match &config {
            Ok(_) => (State::Starting, String::new()),
            Err(e) => (State::Failed, e.clone()),
        };
        Arc::new(Entry {
            name: name.to_owned(),
            raw,
            config,
            status: Mutex::new(Status {
                state,
                message,
                tools: Vec::new(),
                failures: 0,
                retry_at: Instant::now(),
                protocol: String::new(),
            }),
            client: Mutex::new(None),
            stderr: Arc::new(Mutex::new(String::new())),
        })
    }

    fn fail(&self, message: String) {
        let mut status = lock(&self.status);
        status.failures += 1;
        status.retry_at = Instant::now() + backoff(status.failures - 1);
        status.state = State::Failed;
        status.message = message;
    }

    /// Starts the process or connects over HTTP, then does the handshake and the tool listing. The caller holds the
    /// `client` lock.
    fn start_locked(&self, slot: &mut Option<Client<AnyTransport>>) -> Result<(), String> {
        *slot = None;
        let config = self.config.clone()?;
        {
            let mut status = lock(&self.status);
            status.state = State::Starting;
            status.message.clear();
        }
        lock(&self.stderr).clear();
        let outcome = (|| {
            let transport = match &config.url {
                Some(_) => AnyTransport::Http(HttpTransport::new(&config)?),
                None => AnyTransport::Stdio(StdioTransport::spawn(&config, self.stderr.clone())?),
            };
            let mut client = Client::new(transport);
            client.initialize(INITIALIZE_TIMEOUT).map_err(|e| e.message())?;
            let tools = client.list_tools(LIST_TIMEOUT).map_err(|e| e.message())?;
            Ok::<_, String>((client, tools))
        })();
        match outcome {
            Ok((client, tools)) => {
                let mut status = lock(&self.status);
                status.state = State::Ready;
                status.message.clear();
                status.tools = tools;
                status.failures = 0;
                status.protocol = client.protocol.clone();
                *slot = Some(client);
                Ok(())
            }
            Err(error) => {
                let tail = lock(&self.stderr).trim().lines().last().map(str::to_owned);
                let message = match tail {
                    Some(t) if !t.is_empty() => format!("{error} ({t})"),
                    _ => error,
                };
                lock(&self.status).tools.clear();
                self.fail(message.clone());
                Err(message)
            }
        }
    }

    fn start(&self) -> Result<(), String> {
        let mut slot = lock(&self.client);
        self.start_locked(&mut slot)
    }

    /// Runs `f` on a running client, starting the server when it is down — the same way `call` does it.
    fn with_client<T>(&self, f: impl FnOnce(&mut Client<AnyTransport>, Duration) -> Result<T, McpError>) -> Result<T, String> {
        let timeout = self.config.clone()?.timeout;
        let mut slot = lock(&self.client);
        if slot.is_none() {
            let (wait, message) = {
                let status = lock(&self.status);
                (status.retry_at.saturating_duration_since(Instant::now()), status.message.clone())
            };
            if !wait.is_zero() {
                return Err(format!("the MCP server `{}` is not running ({message}); next start in {} s", self.name, wait.as_secs() + 1));
            }
            self.start_locked(&mut slot)?;
        }
        let client = slot.as_mut().ok_or("the MCP server is not running")?;
        match f(client, timeout) {
            Ok(value) => Ok(value),
            Err(McpError::Closed(why)) => {
                *slot = None;
                self.fail(format!("the server ended: {why}"));
                Err(format!("the MCP server `{}` ended while it was asked", self.name))
            }
            Err(e) => Err(e.message()),
        }
    }

    fn resources(&self) -> Result<Vec<Value>, String> {
        self.with_client(|client, timeout| client.list_resources(timeout))
    }

    fn read_resource(&self, uri: &str) -> Result<String, String> {
        self.with_client(|client, timeout| client.read_resource(uri, timeout))
    }

    fn call(&self, tool: &str, arguments: Value) -> Result<(String, bool), String> {
        let tool = tool.to_owned();
        self.with_client(move |client, timeout| client.call_tool(&tool, arguments, timeout))
    }

    fn describe(&self) -> Value {
        let status = lock(&self.status);
        let (command, args, env_keys, url, headers, timeout) = match &self.config {
            Ok(c) => (
                c.command.clone(),
                c.args.clone(),
                c.env.keys().cloned().collect::<Vec<_>>(),
                c.url.clone().unwrap_or_default(),
                c.headers.iter().map(|(k, v)| json!([k, v])).collect::<Vec<_>>(),
                c.timeout.as_secs(),
            ),
            Err(_) => (
                self.raw.get("command").and_then(Value::as_str).unwrap_or_default().to_owned(),
                Vec::new(),
                Vec::new(),
                self.raw.get("url").and_then(Value::as_str).unwrap_or_default().to_owned(),
                Vec::new(),
                DEFAULT_CALL_TIMEOUT_SECS,
            ),
        };
        json!({
            "name": self.name,
            "command": command,
            "args": args,
            "url": url,
            "headers": headers,
            // the values stay in mcp.json: they often hold secrets
            "env_keys": env_keys,
            "timeout_seconds": timeout,
            "state": status.state.name(),
            "message": status.message,
            "protocol": status.protocol,
            "tool_count": status.tools.len(),
            "tools": status.tools,
            "stderr": lock(&self.stderr).clone(),
        })
    }
}

#[derive(Default)]
struct Registry {
    entries: BTreeMap<String, Arc<Entry>>,
    /// Text of `mcp.json` as last read (`None`: never read).
    seen: Option<String>,
    config_error: Option<String>,
}

fn registry() -> &'static Mutex<Registry> {
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Registry::default()))
}

fn start_in_background(entry: &Arc<Entry>) {
    if entry.config.is_ok() {
        let entry = entry.clone();
        let _ = std::thread::Builder::new().name("mcp-start".into()).spawn(move || {
            let _ = entry.start();
        });
    }
}

/// Parses the text of `mcp.json` into the raw server objects.
fn parse_config(text: &str) -> Result<BTreeMap<String, Value>, String> {
    if text.trim().is_empty() {
        return Ok(BTreeMap::new());
    }
    let root: Value = serde_json::from_str(text).map_err(|e| format!("mcp.json is not valid JSON: {e}"))?;
    match root.get("mcpServers") {
        None | Some(Value::Null) => Ok(BTreeMap::new()),
        Some(Value::Object(map)) => Ok(map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()),
        Some(_) => Err("\"mcpServers\" in mcp.json must be an object".to_owned()),
    }
}

/// Reads `mcp.json` when it is new or changed: new servers start, changed ones restart, removed ones stop.
fn sync() {
    let text = std::fs::read_to_string(crate::hub::mcp_file()).unwrap_or_default();
    let mut reg = lock(registry());
    if reg.seen.as_deref() == Some(text.as_str()) {
        return;
    }
    reg.seen = Some(text.clone());
    let wanted = match parse_config(&text) {
        Ok(wanted) => {
            reg.config_error = None;
            wanted
        }
        Err(error) => {
            reg.config_error = Some(error);
            return;
        }
    };
    reg.entries.retain(|name, _| wanted.contains_key(name));
    for (name, raw) in wanted {
        if reg.entries.get(&name).is_some_and(|e| e.raw == raw) {
            continue;
        }
        let entry = Entry::new(&name, raw);
        start_in_background(&entry);
        reg.entries.insert(name, entry);
    }
}

fn entry_named(name: &str) -> Result<Arc<Entry>, String> {
    sync();
    lock(registry()).entries.get(name).cloned().ok_or_else(|| format!("no MCP server named `{name}`"))
}

/// Stops every server (the process is ending).
pub(in crate::server) fn shutdown() {
    let entries: Vec<Arc<Entry>> = lock(registry()).entries.values().cloned().collect();
    for entry in entries {
        // a call in progress holds the lock; the process ends anyway and closing the pipe stops the server
        if let Ok(mut slot) = entry.client.try_lock() {
            *slot = None;
        }
    }
}

fn valid_name(name: &str) -> bool {
    !name.is_empty() && name.len() <= 64 && name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

fn read_file_value() -> Result<Value, String> {
    match std::fs::read_to_string(crate::hub::mcp_file()) {
        Ok(text) if !text.trim().is_empty() => {
            serde_json::from_str(&text).map_err(|e| format!("mcp.json is not valid JSON ({e}); fix or delete the file first"))
        }
        _ => Ok(json!({})),
    }
}

fn write_file_value(root: &Value) -> Result<(), String> {
    let path = crate::hub::mcp_file();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let text = serde_json::to_string_pretty(root).map_err(|e| e.to_string())?;
    std::fs::write(&path, text).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

/// Adds or updates a server. An environment value of `null` keeps the saved value of that variable.
fn upsert_server(body: &Value) -> Result<(), String> {
    let name = body.get("name").and_then(Value::as_str).unwrap_or_default().trim().to_owned();
    if !valid_name(&name) {
        return Err("the name must be 1-64 letters, digits, `-`, `_` or `.`".to_owned());
    }
    let mut root = read_file_value()?;
    let servers = root
        .as_object_mut()
        .ok_or("mcp.json must hold an object")?
        .entry("mcpServers")
        .or_insert_with(|| json!({}));
    let servers = servers.as_object_mut().ok_or("\"mcpServers\" in mcp.json must be an object")?;
    let mut server = servers.get(&name).and_then(Value::as_object).cloned().unwrap_or_default();
    for key in ["command", "args", "url", "timeout_seconds"] {
        if let Some(v) = body.get(key) {
            server.insert(key.to_owned(), v.clone());
        }
    }
    // a remote server does not need the stdio fields, and a local one does not need the url
    if body.get("url").and_then(Value::as_str).map(|u| !u.trim().is_empty()).unwrap_or(false) {
        server.remove("command");
        server.remove("args");
        server.remove("env");
    }
    for key in ["env", "headers"] {
        if let Some(Value::Object(values)) = body.get(key) {
            let old = server.get(key).and_then(Value::as_object).cloned().unwrap_or_default();
            let mut merged = serde_json::Map::new();
            for (k, v) in values {
                if k.trim().is_empty() {
                    continue;
                }
                match v {
                    Value::Null => {
                        if let Some(kept) = old.get(k) {
                            merged.insert(k.clone(), kept.clone());
                        }
                    }
                    other => {
                        merged.insert(k.clone(), other.clone());
                    }
                }
            }
            if merged.is_empty() {
                server.remove(key);
            } else {
                server.insert(key.to_owned(), Value::Object(merged));
            }
        }
    }
    let server = Value::Object(server);
    ServerConfig::from_value(&server)?;
    servers.insert(name, server);
    write_file_value(&root)?;
    sync();
    Ok(())
}

fn remove_server(name: &str) -> Result<(), String> {
    let mut root = read_file_value()?;
    let removed = root
        .get_mut("mcpServers")
        .and_then(Value::as_object_mut)
        .and_then(|servers| servers.shift_remove(name));
    if removed.is_none() {
        return Err(format!("no MCP server named `{name}`"));
    }
    write_file_value(&root)?;
    sync();
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------------------------------------------

#[allow(clippy::result_large_err)]
async fn blocking<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> Result<T, Response> {
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))
}

fn name_of(body: &Value) -> String {
    body.get("name").or_else(|| body.get("server")).and_then(Value::as_str).unwrap_or_default().to_owned()
}

/// `GET /api/studio/mcp/servers`: the configured servers with their state.
pub(in crate::server) async fn servers() -> Response {
    let work = || {
        sync();
        let reg = lock(registry());
        json!({
            "servers": reg.entries.values().map(|e| e.describe()).collect::<Vec<_>>(),
            "config_file": crate::hub::mcp_file().display().to_string(),
            "config_error": reg.config_error,
        })
    };
    match blocking(work).await {
        Ok(v) => Json(v).into_response(),
        Err(r) => r,
    }
}

/// `POST /api/studio/mcp/servers` with `{name, command, args, env}`: adds or updates a server (and starts it).
pub(in crate::server) async fn upsert(Json(body): Json<Value>) -> Response {
    match blocking(move || upsert_server(&body)).await {
        Ok(Ok(())) => Json(json!({"status": "saved"})).into_response(),
        Ok(Err(e)) => api_error(StatusCode::BAD_REQUEST, &e),
        Err(r) => r,
    }
}

/// `DELETE /api/studio/mcp/servers` with `{name}`.
pub(in crate::server) async fn remove(Json(body): Json<Value>) -> Response {
    let name = name_of(&body);
    match blocking(move || remove_server(&name)).await {
        Ok(Ok(())) => Json(json!({"status": "removed"})).into_response(),
        Ok(Err(e)) => api_error(StatusCode::NOT_FOUND, &e),
        Err(r) => r,
    }
}

/// `POST /api/studio/mcp/restart` with `{name}`: stops the server and starts it again, answering with its state.
pub(in crate::server) async fn restart(Json(body): Json<Value>) -> Response {
    let name = name_of(&body);
    let work = move || -> Result<Value, String> {
        let entry = entry_named(&name)?;
        lock(&entry.status).failures = 0;
        let _ = entry.start();
        Ok(entry.describe())
    };
    match blocking(work).await {
        Ok(Ok(v)) => Json(v).into_response(),
        Ok(Err(e)) => api_error(StatusCode::NOT_FOUND, &e),
        Err(r) => r,
    }
}

/// `GET /api/studio/mcp/tools`: `{tools: [{server, name, description, input_schema}]}` of the servers that are ready.
/// Servers that are still starting are waited for (at most 15 s).
pub(in crate::server) async fn tools() -> Response {
    let work = || {
        sync();
        let entries: Vec<Arc<Entry>> = lock(registry()).entries.values().cloned().collect();
        let deadline = Instant::now() + Duration::from_secs(15);
        while Instant::now() < deadline && entries.iter().any(|e| lock(&e.status).state == State::Starting) {
            std::thread::sleep(Duration::from_millis(100));
        }
        let mut tools = Vec::new();
        for entry in &entries {
            let status = lock(&entry.status);
            if status.state != State::Ready {
                continue;
            }
            for tool in &status.tools {
                tools.push(json!({
                    "server": entry.name,
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["input_schema"],
                }));
            }
        }
        json!({"tools": tools})
    };
    match blocking(work).await {
        Ok(v) => Json(v).into_response(),
        Err(r) => r,
    }
}

/// `POST /api/studio/mcp/call` with `{server, name, arguments}`: `{content, is_error}`.
pub(in crate::server) async fn call(Json(body): Json<Value>) -> Response {
    let server = body.get("server").and_then(Value::as_str).unwrap_or_default().to_owned();
    let tool = body.get("name").and_then(Value::as_str).unwrap_or_default().to_owned();
    let arguments = match body.get("arguments") {
        None | Some(Value::Null) => json!({}),
        Some(v) => v.clone(),
    };
    if !arguments.is_object() {
        return api_error(StatusCode::BAD_REQUEST, "arguments must be an object");
    }
    if tool.is_empty() {
        return api_error(StatusCode::BAD_REQUEST, "the tool name is missing");
    }
    let work = move || -> Result<(String, bool), (StatusCode, String)> {
        let entry = entry_named(&server).map_err(|e| (StatusCode::NOT_FOUND, e))?;
        entry.call(&tool, arguments).map_err(|e| (StatusCode::BAD_GATEWAY, e))
    };
    match blocking(work).await {
        Ok(Ok((content, is_error))) => Json(json!({"content": content, "is_error": is_error})).into_response(),
        Ok(Err((status, e))) => api_error(status, &e),
        Err(r) => r,
    }
}

/// `POST /api/studio/mcp/resources` `{server}`: the resources a server publishes.
pub(in crate::server) async fn resources(Json(body): Json<Value>) -> Response {
    let server = body.get("server").and_then(Value::as_str).unwrap_or_default().to_owned();
    let for_work = server.clone();
    let work = move || -> Result<Vec<Value>, (StatusCode, String)> {
        let entry = entry_named(&for_work).map_err(|e| (StatusCode::NOT_FOUND, e))?;
        entry.resources().map_err(|e| (StatusCode::BAD_GATEWAY, e))
    };
    match blocking(work).await {
        Ok(Ok(items)) => Json(json!({"server": server, "resources": items})).into_response(),
        Ok(Err((status, e))) => api_error(status, &e),
        Err(r) => r,
    }
}

/// `POST /api/studio/mcp/resource` `{server, uri}`: the text of one resource.
pub(in crate::server) async fn resource(Json(body): Json<Value>) -> Response {
    let server = body.get("server").and_then(Value::as_str).unwrap_or_default().to_owned();
    let uri = body.get("uri").and_then(Value::as_str).unwrap_or_default().to_owned();
    if uri.is_empty() {
        return api_error(StatusCode::BAD_REQUEST, "the resource uri is missing");
    }
    let for_work = (server.clone(), uri.clone());
    let work = move || -> Result<String, (StatusCode, String)> {
        let entry = entry_named(&for_work.0).map_err(|e| (StatusCode::NOT_FOUND, e))?;
        entry.read_resource(&for_work.1).map_err(|e| (StatusCode::BAD_GATEWAY, e))
    };
    match blocking(work).await {
        Ok(Ok(content)) => Json(json!({"uri": uri, "content": content})).into_response(),
        Ok(Err((status, e))) => api_error(status, &e),
        Err(r) => r,
    }
}

/// The `mcp_resource` tool: list the resources of a server, or read one by uri.
pub(super) fn read_tool(body: &Value) -> Result<String, String> {
    let server = body.get("server").and_then(Value::as_str).unwrap_or("").trim();
    if server.is_empty() {
        return Err("server is missing".to_owned());
    }
    let uri = body.get("uri").and_then(Value::as_str).unwrap_or("").trim();
    let entry = entry_named(server)?;
    if uri.is_empty() {
        let items = entry.resources()?;
        if items.is_empty() {
            return Ok(format!("The MCP server `{server}` publishes no resources."));
        }
        let mut out = format!("{} resource(s) of `{server}`:\n", items.len());
        for item in items.iter().take(100) {
            out.push_str(&format!(
                "- {} ({}){}: {}\n",
                item.get("uri").and_then(Value::as_str).unwrap_or(""),
                item.get("name").and_then(Value::as_str).unwrap_or(""),
                {
                    let kind = item.get("mime_type").and_then(Value::as_str).unwrap_or("");
                    if kind.is_empty() { String::new() } else { format!(" [{kind}]") }
                },
                item.get("description").and_then(Value::as_str).unwrap_or("")
            ));
        }
        return Ok(out);
    }
    let text = entry.read_resource(uri)?;
    let shown: String = text.chars().take(60_000).collect();
    Ok(format!("Resource {uri} of `{server}`:\n\n{shown}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    /// An in-memory server: `handler` sees every line the client sends and answers with lines.
    struct Mock {
        handler: Box<dyn FnMut(Value) -> Vec<String> + Send>,
        inbox: VecDeque<String>,
        closed: bool,
        sent: Vec<Value>,
    }

    impl Mock {
        fn new(handler: impl FnMut(Value) -> Vec<String> + Send + 'static) -> Self {
            Mock { handler: Box::new(handler), inbox: VecDeque::new(), closed: false, sent: Vec::new() }
        }
    }

    impl Transport for Mock {
        fn send(&mut self, line: &str) -> Result<(), String> {
            if self.closed {
                return Err("broken pipe".to_owned());
            }
            let value: Value = serde_json::from_str(line).unwrap();
            self.sent.push(value.clone());
            let replies = (self.handler)(value);
            self.inbox.extend(replies);
            Ok(())
        }
        fn recv(&mut self, _timeout: Duration) -> Recv {
            match self.inbox.pop_front() {
                Some(line) => Recv::Line(line),
                None if self.closed => Recv::Closed,
                None => Recv::Timeout,
            }
        }
    }

    fn reply(request: &Value, result: Value) -> String {
        json!({"jsonrpc": "2.0", "id": request["id"], "result": result}).to_string()
    }

    const SHORT: Duration = Duration::from_millis(50);

    #[test]
    fn initialize_accepts_the_version_of_the_server_and_confirms() {
        let mut client = Client::new(Mock::new(|req| {
            if req["method"] == "initialize" {
                assert_eq!(req["params"]["protocolVersion"], PROTOCOL_VERSION);
                vec![reply(&req, json!({"protocolVersion": "2024-11-05", "capabilities": {}}))]
            } else {
                vec![]
            }
        }));
        client.initialize(SHORT).unwrap();
        assert_eq!(client.protocol, "2024-11-05");
        let methods: Vec<_> = client.transport.sent.iter().map(|m| m["method"].as_str().unwrap().to_owned()).collect();
        assert_eq!(methods, ["initialize", "notifications/initialized"]);
    }

    #[test]
    fn tools_list_follows_the_cursor() {
        let mut client = Client::new(Mock::new(|req| {
            let cursor = req["params"].get("cursor").and_then(Value::as_str).map(str::to_owned);
            let result = match cursor.as_deref() {
                None => json!({"tools": [{"name": "a", "description": "first", "inputSchema": {"type": "object"}}], "nextCursor": "p2"}),
                Some("p2") => json!({"tools": [{"name": "b"}, {"description": "no name"}]}),
                Some(other) => panic!("unexpected cursor {other}"),
            };
            vec![reply(&req, result)]
        }));
        let tools = client.list_tools(SHORT).unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0]["name"], "a");
        assert_eq!(tools[0]["description"], "first");
        assert_eq!(tools[1]["name"], "b");
        assert_eq!(tools[1]["input_schema"], json!({"type": "object"}));
    }

    #[test]
    fn tool_call_joins_text_and_notes_images() {
        let mut client = Client::new(Mock::new(|req| {
            assert_eq!(req["method"], "tools/call");
            assert_eq!(req["params"]["name"], "echo");
            assert_eq!(req["params"]["arguments"]["x"], 1);
            vec![reply(
                &req,
                json!({"content": [
                    {"type": "text", "text": "one"},
                    {"type": "image", "data": "AAAA", "mimeType": "image/png"},
                    {"type": "resource", "resource": {"uri": "file:///a", "text": "body"}},
                    {"type": "text", "text": "two"},
                ], "isError": true}),
            )]
        }));
        let (text, is_error) = client.call_tool("echo", json!({"x": 1}), SHORT).unwrap();
        assert_eq!(text, "one\n[image]\nbody\ntwo");
        assert!(is_error);
    }

    #[test]
    fn structured_content_is_used_when_there_are_no_blocks() {
        let (text, is_error) = result_text(&json!({"structuredContent": {"n": 2}}));
        assert_eq!(text, "{\"n\":2}");
        assert!(!is_error);
    }

    #[test]
    fn rpc_errors_are_reported_with_their_message() {
        let mut client = Client::new(Mock::new(|req| {
            vec![json!({"jsonrpc": "2.0", "id": req["id"], "error": {"code": -32602, "message": "bad arguments"}}).to_string()]
        }));
        let error = client.call_tool("x", json!({}), SHORT).unwrap_err();
        assert_eq!(error, McpError::Rpc("bad arguments (code -32602)".to_owned()));
    }

    #[test]
    fn a_silent_server_times_out_and_the_request_is_cancelled() {
        let mut client = Client::new(Mock::new(|_| vec![]));
        let error = client.call_tool("slow", json!({}), Duration::from_millis(20)).unwrap_err();
        assert_eq!(error, McpError::Timeout);
        assert_eq!(client.transport.sent.last().unwrap()["method"], "notifications/cancelled");
    }

    #[test]
    fn a_closed_connection_is_told_apart_from_a_timeout() {
        let mut mock = Mock::new(|_| vec![]);
        mock.closed = true;
        let mut client = Client::new(mock);
        assert!(matches!(client.call_tool("x", json!({}), SHORT), Err(McpError::Closed(_))));
    }

    #[test]
    fn noise_notifications_and_server_requests_are_skipped_or_answered() {
        let mut client = Client::new(Mock::new(|req| {
            if req["method"] == "tools/call" {
                vec![
                    "starting up...".to_owned(),
                    json!({"jsonrpc": "2.0", "method": "notifications/message", "params": {}}).to_string(),
                    json!({"jsonrpc": "2.0", "id": "s1", "method": "ping"}).to_string(),
                    json!({"jsonrpc": "2.0", "id": "s2", "method": "sampling/createMessage"}).to_string(),
                    json!({"jsonrpc": "2.0", "id": 999, "result": {}}).to_string(),
                    reply(&req, json!({"content": [{"type": "text", "text": "done"}]})),
                ]
            } else {
                vec![]
            }
        }));
        let (text, _) = client.call_tool("t", json!({}), SHORT).unwrap();
        assert_eq!(text, "done");
        let ping = client.transport.sent.iter().find(|m| m["id"] == "s1").unwrap();
        assert!(ping.get("result").is_some());
        let other = client.transport.sent.iter().find(|m| m["id"] == "s2").unwrap();
        assert_eq!(other["error"]["code"], -32601);
    }

    #[test]
    fn config_is_validated() {
        let ok = ServerConfig::from_value(&json!({"command": "npx", "args": ["-y", "pkg"], "env": {"K": "V"}, "timeout_seconds": 5})).unwrap();
        assert_eq!(ok.args, ["-y", "pkg"]);
        assert_eq!(ok.env["K"], "V");
        assert_eq!(ok.timeout, Duration::from_secs(5));
        assert_eq!(ServerConfig::from_value(&json!({"command": "x"})).unwrap().timeout, Duration::from_secs(DEFAULT_CALL_TIMEOUT_SECS));
        assert!(ServerConfig::from_value(&json!({"command": " "})).is_err());
        assert!(ServerConfig::from_value(&json!({"command": "x", "args": [1]})).is_err());
        assert!(ServerConfig::from_value(&json!({"command": "x", "env": {"a": 1}})).is_err());
    }

    #[test]
    fn mcp_json_is_parsed_in_the_common_format() {
        let servers = parse_config(r#"{"mcpServers": {"a": {"command": "x"}, "b": {"command": "y"}}}"#).unwrap();
        assert_eq!(servers.keys().collect::<Vec<_>>(), ["a", "b"]);
        assert!(parse_config("").unwrap().is_empty());
        assert!(parse_config("{").is_err());
        assert!(parse_config(r#"{"mcpServers": []}"#).is_err());
    }

    #[test]
    fn stderr_buffer_stays_bounded_and_backoff_grows() {
        let mut buffer = String::new();
        for _ in 0..1000 {
            push_bounded(&mut buffer, "héllo wörld\n");
        }
        assert!(buffer.len() <= STDERR_LIMIT);
        assert!(buffer.ends_with("wörld\n"));
        assert_eq!(backoff(0), Duration::from_secs(1));
        assert_eq!(backoff(3), Duration::from_secs(8));
        assert_eq!(backoff(30), Duration::from_secs(MAX_BACKOFF_SECS));
    }

    #[test]
    fn server_names_are_restricted() {
        assert!(valid_name("files-1_x.y"));
        assert!(!valid_name(""));
        assert!(!valid_name("a b"));
        assert!(!valid_name("../x"));
    }

    /// A throwaway HTTP server that answers the handshake and one tools/list, so the HTTP transport is tested against
    /// something that speaks the protocol without a network.
    fn fake_mcp_server() -> (String, std::thread::JoinHandle<()>) {
        use std::io::{BufRead, BufReader, Read, Write};
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let handle = std::thread::spawn(move || {
            for _ in 0..3 {
                let Ok((mut stream, _)) = listener.accept() else { return };
                let mut reader = BufReader::new(stream.try_clone().expect("clone"));
                let mut content_length = 0usize;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 {
                        break;
                    }
                    let trimmed = line.trim_end().to_owned();
                    if trimmed.is_empty() {
                        break;
                    }
                    let lower = trimmed.to_ascii_lowercase();
                    if let Some(v) = lower.strip_prefix("content-length:") {
                        content_length = v.trim().parse().unwrap_or(0);
                    }
                }
                let mut body = vec![0u8; content_length];
                let _ = reader.read_exact(&mut body);
                let request: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
                let method = request.get("method").and_then(Value::as_str).unwrap_or("");
                let id = request.get("id").cloned();
                let reply = match (method, id) {
                    ("initialize", Some(id)) => json!({
                        "jsonrpc": "2.0", "id": id,
                        "result": {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "serverInfo": {"name": "fake", "version": "1"}}
                    }),
                    ("tools/list", Some(id)) => json!({
                        "jsonrpc": "2.0", "id": id,
                        "result": {"tools": [{"name": "echo", "description": "Echoes", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}}]}
                    }),
                    _ => Value::Null,
                };
                let (status, payload) = if reply.is_null() { ("202 Accepted", String::new()) } else { ("200 OK", reply.to_string()) };
                let response = format!("HTTP/1.1 {status}\r\ncontent-type: application/json\r\nconnection: close\r\ncontent-length: {}\r\n\r\n{payload}", payload.len());
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });
        (format!("http://{addr}/mcp"), handle)
    }

    #[test]
    fn a_remote_server_is_reached_over_http() {
        let (url, server) = fake_mcp_server();
        let config = ServerConfig::from_value(&json!({"url": url, "headers": {"Authorization": "Bearer ${FLLM_TEST_MCP_TOKEN}"}})).expect("config");
        assert_eq!(config.headers["Authorization"], "Bearer ${FLLM_TEST_MCP_TOKEN}", "the environment reference stays in the file and is expanded per request");
        let mut client = Client::new(HttpTransport::new(&config).expect("transport"));
        client.initialize(Duration::from_secs(5)).expect("initialize");
        assert_eq!(client.protocol, PROTOCOL_VERSION);
        let tools = client.list_tools(Duration::from_secs(5)).expect("tools/list");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "echo");
        server.join().expect("server thread");
    }

    #[test]
    fn the_url_is_checked_and_environment_references_are_expanded() {
        assert!(ServerConfig::from_value(&json!({"url": "ftp://example.com/mcp"})).is_err());
        assert!(ServerConfig::from_value(&json!({})).is_err(), "neither a command nor a url");
        assert!(ServerConfig::from_value(&json!({"command": "npx"})).is_ok());
        let remote = ServerConfig::from_value(&json!({"url": "https://example.com/mcp"})).expect("remote");
        assert_eq!(remote.url.as_deref(), Some("https://example.com/mcp"));
        std::env::set_var("FLLM_TEST_EXPAND", "secret");
        assert_eq!(expand_env("Bearer ${FLLM_TEST_EXPAND}!"), "Bearer secret!");
        assert_eq!(expand_env("no refs"), "no refs");
        assert_eq!(expand_env("${FLLM_MISSING_VAR}"), "");
        assert_eq!(expand_env("${unclosed"), "${unclosed");
    }
}
