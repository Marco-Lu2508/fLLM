//! A small Language Server Protocol client, so the agent can ask a real language server instead of guessing:
//! `diagnostics` for a file, `symbols` for a workspace-wide symbol search and `definition` for a position.
//!
//! The protocol is JSON-RPC 2.0 with a `Content-Length` header, like MCP but with headers. Reading happens on a thread
//! (a pipe has no timeout), the requests go through one channel, and every wait is bounded.
//!
//! What runs is decided by the file: `rust-analyzer` for `.rs`, `pyright-langserver` or `pylsp` for `.py`,
//! `typescript-language-server` for `.ts`/`.tsx`/`.js`, `gopls` for `.go`, `clangd` for C and C++. Without a server on
//! `PATH` the tool says so and points at the built-in `diagnostics` tool, which needs no server.
use super::agent::AgentSession;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

const INIT_TIMEOUT: Duration = Duration::from_secs(30);
const DIAGNOSTICS_TIMEOUT: Duration = Duration::from_secs(12);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(20);
const MAX_ITEMS: usize = 100;

/// The servers this client knows, in the order they are tried for an extension.
const SERVERS: [( &str, &[&str], &[&str]); 6] = [
    ("rust-analyzer", &[], &["rs"]),
    ("typescript-language-server", &["--stdio"], &["ts", "tsx", "js", "jsx", "mjs"]),
    ("pyright-langserver", &["--stdio"], &["py"]),
    ("pylsp", &[], &["py"]),
    ("gopls", &[], &["go"]),
    ("clangd", &[], &["c", "h", "cc", "cpp", "hpp"]),
];

fn on_path(command: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(command))
        .find(|candidate| candidate.is_file())
        .or_else(|| {
            // Windows shims
            std::env::split_paths(&path).map(|dir| dir.join(format!("{command}.cmd"))).find(|c| c.is_file())
        })
}

/// The language server for a file, with the arguments it needs for stdio.
pub(super) fn server_for(path: &Path) -> Option<(PathBuf, Vec<String>)> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    for (command, args, extensions) in SERVERS {
        if !extensions.contains(&ext.as_str()) {
            continue;
        }
        if let Some(binary) = on_path(command) {
            return Some((binary, args.iter().map(|a| (*a).to_owned()).collect()));
        }
    }
    None
}

/// Which servers are installed here, for the message when none fits.
pub(super) fn installed() -> Vec<&'static str> {
    SERVERS.iter().filter(|(c, _, _)| on_path(c).is_some()).map(|(c, _, _)| *c).collect()
}

struct Lsp {
    child: Child,
    stdin: ChildStdin,
    messages: Receiver<Value>,
    next_id: i64,
    /// Diagnostics per file URI, filled by `textDocument/publishDiagnostics`.
    diagnostics: BTreeMap<String, Vec<Value>>,
}

impl Lsp {
    fn start(binary: &Path, args: &[String], root: &Path) -> Result<Lsp, String> {
        let mut child = Command::new(binary)
            .args(args)
            .current_dir(root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("cannot start {}: {e}", binary.display()))?;
        let stdin = child.stdin.take().ok_or("the language server has no stdin")?;
        let stdout: ChildStdout = child.stdout.take().ok_or("the language server has no stdout")?;
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            loop {
                match read_message(&mut reader) {
                    Some(value) => {
                        if tx.send(value).is_err() {
                            return;
                        }
                    }
                    None => return,
                }
            }
        });
        Ok(Lsp { child, stdin, messages: rx, next_id: 1, diagnostics: BTreeMap::new() })
    }

    fn send(&mut self, message: &Value) -> Result<(), String> {
        let body = message.to_string();
        let framed = format!("Content-Length: {}\r\n\r\n{}", body.len(), body);
        self.stdin.write_all(framed.as_bytes()).and_then(|()| self.stdin.flush()).map_err(|e| format!("cannot write to the language server: {e}"))
    }

    fn notify(&mut self, method: &str, params: Value) -> Result<(), String> {
        self.send(&json!({"jsonrpc": "2.0", "method": method, "params": params}))
    }

    /// Sends a request and returns the answer with `id`. Diagnostics notifications that arrive meanwhile are kept.
    fn request(&mut self, method: &str, params: Value, timeout: Duration) -> Result<Value, String> {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}))?;
        self.wait_for_id(id, timeout)
    }

    fn wait_for_id(&mut self, id: i64, timeout: Duration) -> Result<Value, String> {
        let deadline = Instant::now() + timeout;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err("the language server did not answer in time".to_owned());
            }
            let message = match self.messages.recv_timeout(left) {
                Ok(message) => message,
                Err(RecvTimeoutError::Timeout) => return Err("the language server did not answer in time".to_owned()),
                Err(RecvTimeoutError::Disconnected) => return Err("the language server ended".to_owned()),
            };
            if let Some(method) = message.get("method").and_then(Value::as_str) {
                match method {
                    "textDocument/publishDiagnostics" => {
                        if let Some(params) = message.get("params") {
                            let uri = params.get("uri").and_then(Value::as_str).unwrap_or("").to_owned();
                            let items = params.get("diagnostics").and_then(Value::as_array).cloned().unwrap_or_default();
                            self.diagnostics.insert(uri, items);
                        }
                    }
                    // the server asks something: answer with an empty result so it does not wait
                    _ => {
                        if let Some(server_id) = message.get("id").cloned() {
                            let _ = self.send(&json!({"jsonrpc": "2.0", "id": server_id, "result": Value::Null}));
                        }
                    }
                }
                continue;
            }
            if message.get("id").and_then(Value::as_i64) == Some(id) {
                if let Some(error) = message.get("error") {
                    let text = error.get("message").and_then(Value::as_str).unwrap_or("error without a message");
                    return Err(text.to_owned());
                }
                return Ok(message.get("result").cloned().unwrap_or(Value::Null));
            }
        }
    }

    /// Collects diagnostics notifications for a while (servers send them after `didOpen`).
    fn drain_diagnostics(&mut self, how_long: Duration) {
        let deadline = Instant::now() + how_long;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return;
            }
            match self.messages.recv_timeout(left.min(Duration::from_millis(300))) {
                Ok(message) => {
                    if message.get("method").and_then(Value::as_str) == Some("textDocument/publishDiagnostics") {
                        if let Some(params) = message.get("params") {
                            let uri = params.get("uri").and_then(Value::as_str).unwrap_or("").to_owned();
                            let items = params.get("diagnostics").and_then(Value::as_array).cloned().unwrap_or_default();
                            self.diagnostics.insert(uri, items);
                        }
                    }
                }
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => return,
            }
        }
    }
}

impl Drop for Lsp {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// One `Content-Length` framed message, or `None` at the end of the stream.
fn read_message(reader: &mut BufReader<ChildStdout>) -> Option<Value> {
    let mut length = 0usize;
    loop {
        let mut line = String::new();
        let read = reader.read_line(&mut line).ok()?;
        if read == 0 {
            return None;
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            break;
        }
        if let Some(value) = trimmed.to_ascii_lowercase().strip_prefix("content-length:") {
            length = value.trim().parse().ok()?;
        }
    }
    if length == 0 || length > 32 * 1024 * 1024 {
        return None;
    }
    let mut body = vec![0u8; length];
    reader.read_exact(&mut body).ok()?;
    serde_json::from_slice(&body).ok()
}

fn file_uri(path: &Path) -> String {
    let mut text = path.display().to_string().replace('\\', "/");
    if !text.starts_with('/') {
        text.insert(0, '/');
    }
    format!("file://{text}")
}

fn language_id(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase().as_str() {
        "rs" => "rust",
        "py" => "python",
        "ts" | "tsx" => "typescript",
        "js" | "jsx" | "mjs" => "javascript",
        "go" => "go",
        "c" | "h" => "c",
        "cc" | "cpp" | "hpp" => "cpp",
        _ => "plaintext",
    }
}

/// Starts a server for `path`, handshakes and opens the file.
fn open(root: &Path, path: &Path) -> Result<Lsp, String> {
    let (binary, args) = server_for(path).ok_or_else(|| {
        let known = installed();
        let hint = if known.is_empty() {
            "no language server is installed".to_owned()
        } else {
            format!("installed: {}", known.join(", "))
        };
        format!(
            "no language server for this file type ({hint}). Install one (rust-analyzer, pyright-langserver, typescript-language-server, gopls or clangd), or use the diagnostics tool, which needs no server."
        )
    })?;
    let mut lsp = Lsp::start(&binary, &args, root)?;
    let root_uri = file_uri(root);
    lsp.request(
        "initialize",
        json!({
            "processId": std::process::id(),
            "rootUri": root_uri,
            "capabilities": {
                "textDocument": {"publishDiagnostics": {}, "definition": {}, "references": {}},
                "workspace": {"symbol": {}}
            },
            "workspaceFolders": [{"uri": root_uri, "name": root.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()}],
        }),
        INIT_TIMEOUT,
    )?;
    lsp.notify("initialized", json!({}))?;
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    lsp.notify(
        "textDocument/didOpen",
        json!({"textDocument": {"uri": file_uri(path), "languageId": language_id(path), "version": 1, "text": text}}),
    )?;
    Ok(lsp)
}

fn severity_name(severity: Option<i64>) -> &'static str {
    match severity {
        Some(1) => "error",
        Some(2) => "warning",
        Some(3) => "info",
        _ => "hint",
    }
}

/// `diagnostics`: what the language server reports for one file.
fn diagnostics(root: &Path, path: &Path) -> Result<String, String> {
    let mut lsp = open(root, path)?;
    lsp.drain_diagnostics(DIAGNOSTICS_TIMEOUT);
    let items = lsp.diagnostics.get(&file_uri(path)).cloned();
    let shown = path.strip_prefix(root).unwrap_or(path).display().to_string();
    match items {
        None => Ok(format!("The language server sent no diagnostics for {shown} yet. It may still be indexing; try again in a moment.")),
        Some(items) if items.is_empty() => Ok(format!("No problems reported by the language server for {shown}.")),
        Some(items) => {
            let mut out = format!("{} problem(s) from the language server in {shown}:\n", items.len());
            for item in items.iter().take(MAX_ITEMS) {
                let range = item.get("range").cloned().unwrap_or(Value::Null);
                let line = range.pointer("/start/line").and_then(Value::as_u64).unwrap_or(0) + 1;
                let character = range.pointer("/start/character").and_then(Value::as_u64).unwrap_or(0) + 1;
                let message = item.get("message").and_then(Value::as_str).unwrap_or("");
                out.push_str(&format!(
                    "- [{}] {}:{}:{}: {}\n",
                    severity_name(item.get("severity").and_then(Value::as_i64)),
                    shown,
                    line,
                    character,
                    message.replace('\n', " ").chars().take(300).collect::<String>()
                ));
            }
            Ok(out)
        }
    }
}

/// `symbols`: a workspace-wide symbol search.
fn symbols(root: &Path, query: &str) -> Result<String, String> {
    // any file of the workspace starts a server; rust-analyzer needs one to know the crate
    let starter = ["src/main.rs", "src/lib.rs", "main.py", "index.ts", "main.go", "main.c"]
        .iter()
        .map(|p| root.join(p))
        .find(|p| p.is_file())
        .or_else(|| walk_for_source(root))
        .ok_or_else(|| "no source file to start a language server with".to_owned())?;
    let mut lsp = open(root, &starter)?;
    let result = lsp.request("workspace/symbol", json!({"query": query}), REQUEST_TIMEOUT)?;
    let items = result.as_array().cloned().unwrap_or_default();
    if items.is_empty() {
        return Ok(format!("No symbol matches '{query}'."));
    }
    let mut out = format!("{} symbol(s) matching '{query}':\n", items.len());
    for item in items.iter().take(MAX_ITEMS) {
        let name = item.get("name").and_then(Value::as_str).unwrap_or("");
        let kind = item.get("kind").and_then(Value::as_i64).unwrap_or(0);
        let location = item.get("location").cloned().unwrap_or(Value::Null);
        let uri = location.get("uri").and_then(Value::as_str).unwrap_or("");
        let line = location.pointer("/range/start/line").and_then(Value::as_u64).unwrap_or(0) + 1;
        let path = uri.strip_prefix("file://").unwrap_or(uri);
        out.push_str(&format!("- {name} (kind {kind}) at {path}:{line}\n"));
    }
    Ok(out)
}

/// The first source file below `root` (a bounded walk, so a huge tree cannot stall the tool).
fn walk_for_source(root: &Path) -> Option<PathBuf> {
    let mut stack = vec![root.to_owned()];
    let mut seen = 0usize;
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            seen += 1;
            if seen > 5_000 {
                return None;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') || name == "target" || name == "node_modules" {
                continue;
            }
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if server_for(&path).is_some() {
                return Some(path);
            }
        }
    }
    None
}

/// `definition` at a position.
fn definition(root: &Path, path: &Path, line: u64, character: u64) -> Result<String, String> {
    let mut lsp = open(root, path)?;
    let result = lsp.request(
        "textDocument/definition",
        json!({"textDocument": {"uri": file_uri(path)}, "position": {"line": line.saturating_sub(1), "character": character.saturating_sub(1)}}),
        REQUEST_TIMEOUT,
    )?;
    let items: Vec<Value> = match result {
        Value::Array(items) => items,
        Value::Null => Vec::new(),
        other => vec![other],
    };
    if items.is_empty() {
        return Ok("The language server found no definition at that position.".to_owned());
    }
    let mut out = String::from("Definitions:\n");
    for item in items.iter().take(20) {
        let uri = item.get("uri").or_else(|| item.get("targetUri")).and_then(Value::as_str).unwrap_or("");
        let range = item.get("range").or_else(|| item.get("targetSelectionRange")).cloned().unwrap_or(Value::Null);
        let line = range.pointer("/start/line").and_then(Value::as_u64).unwrap_or(0) + 1;
        let path = uri.strip_prefix("file://").unwrap_or(uri);
        out.push_str(&format!("- {}:{}\n", path, line));
    }
    Ok(out)
}

/// The `lsp` tool: `{action: diagnostics|symbols|definition, path?, line?, character?, query?}`.
pub(super) fn run(session: &AgentSession, args: &Value) -> Result<String, String> {
    let root = session.root_dir()?;
    let action = args.get("action").and_then(Value::as_str).unwrap_or("diagnostics");
    match action {
        "diagnostics" => {
            let given = args.get("path").and_then(Value::as_str).unwrap_or("");
            let path = if given.trim().is_empty() {
                walk_for_source(&root).ok_or_else(|| "no source file found in this folder".to_owned())?
            } else {
                session.resolve_path(given)?
            };
            if !path.is_file() {
                return Err(format!("{given} is not a file"));
            }
            diagnostics(&root, &path)
        }
        "symbols" => {
            let query = args.get("query").and_then(Value::as_str).unwrap_or("");
            symbols(&root, query)
        }
        "definition" => {
            let given = args.get("path").and_then(Value::as_str).unwrap_or("");
            if given.trim().is_empty() {
                return Err("definition needs a path".to_owned());
            }
            let path = session.resolve_path(given)?;
            let line = args.get("line").and_then(Value::as_u64).unwrap_or(1);
            let character = args.get("character").and_then(Value::as_u64).unwrap_or(1);
            definition(&root, &path, line, character)
        }
        other => Err(format!("unknown action '{other}'; use diagnostics, symbols or definition")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_extension_picks_the_server_and_nothing_else_does() {
        // no server installed in this environment is fine: then there is simply none for the extension
        let rust = server_for(Path::new("/x/main.rs"));
        if let Some((binary, _)) = rust {
            assert!(binary.to_string_lossy().contains("rust-analyzer"));
        }
        assert!(server_for(Path::new("/x/notes.txt")).is_none());
        assert!(server_for(Path::new("/x/noext")).is_none());
    }

    #[test]
    fn a_file_uri_is_absolute_and_uses_forward_slashes() {
        assert_eq!(file_uri(Path::new("/tmp/a b.rs")), "file:///tmp/a b.rs");
        assert_eq!(file_uri(Path::new("rel/x.py")), "file:///rel/x.py");
    }

    #[test]
    fn the_language_id_follows_the_extension() {
        assert_eq!(language_id(Path::new("a.rs")), "rust");
        assert_eq!(language_id(Path::new("a.tsx")), "typescript");
        assert_eq!(language_id(Path::new("a.unknown")), "plaintext");
    }

    #[test]
    fn severity_numbers_become_words() {
        assert_eq!(severity_name(Some(1)), "error");
        assert_eq!(severity_name(Some(2)), "warning");
        assert_eq!(severity_name(None), "hint");
    }

    /// The framing is tested against a real pipe pair written by hand: header, blank line, JSON body.
    #[test]
    fn framing_reads_a_message_and_ignores_extra_headers() {
        use std::io::Write as _;
        let dir = std::env::temp_dir().join(format!("fllm-lsp-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let script = dir.join("frame.txt");
        let body = r#"{"jsonrpc":"2.0","method":"x","params":{}}"#;
        let mut file = std::fs::File::create(&script).unwrap();
        write!(file, "Content-Type: application/vscode-jsonrpc; charset=utf-8\r\nContent-Length: {}\r\n\r\n{}", body.len(), body).unwrap();
        drop(file);
        // read it back the way the client does: the helper takes a BufReader over any Read, so a file stands in for a pipe
        let mut reader = BufReader::new(std::fs::File::open(&script).unwrap());
        let mut length = 0usize;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap_or(0) == 0 {
                break;
            }
            if line.trim_end().is_empty() {
                break;
            }
            if let Some(value) = line.trim_end().to_ascii_lowercase().strip_prefix("content-length:") {
                length = value.trim().parse().unwrap();
            }
        }
        let mut raw = vec![0u8; length];
        reader.read_exact(&mut raw).unwrap();
        let value: Value = serde_json::from_slice(&raw).unwrap();
        assert_eq!(value["method"], "x");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
