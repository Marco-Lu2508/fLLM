//! A small WebSocket client and a Chrome DevTools Protocol (CDP) connection on top of it, for the in-app browser.
//!
//! Only what CDP needs: a client handshake over plain TCP (the debugging port is on 127.0.0.1), masked text frames
//! out, text frames in (fragments joined, ping answered, close honoured). No extension, no compression, no TLS.
//! The messages of CDP are JSON: a command `{id, method, params, sessionId?}` is answered by `{id, result}` or
//! `{id, error}`, everything else is an event `{method, params, sessionId?}`.
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{tcp::OwnedReadHalf, tcp::OwnedWriteHalf, TcpStream},
    sync::{broadcast, oneshot},
};

/// Largest message accepted from the browser (a full-page screenshot as base64 can be several megabytes).
const MAX_MESSAGE: usize = 64 << 20;
const GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

// ---- SHA-1 (only for the handshake: `Sec-WebSocket-Accept`) ---------------------------------------------------

fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h: [u32; 5] = [0x6745_2301, 0xEFCD_AB89, 0x98BA_DCFE, 0x1032_5476, 0xC3D2_E1F0];
    let mut message = data.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for block in message.chunks(64) {
        let mut w = [0u32; 80];
        for (i, word) in block.chunks(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let [mut a, mut b, mut c, mut d, mut e] = h;
        for (i, wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | (!b & d), 0x5A82_7999),
                20..=39 => (b ^ c ^ d, 0x6ED9_EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1B_BCDC),
                _ => (b ^ c ^ d, 0xCA62_C1D6u32),
            };
            let temp = a.rotate_left(5).wrapping_add(f).wrapping_add(e).wrapping_add(k).wrapping_add(*wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }
        for (slot, value) in h.iter_mut().zip([a, b, c, d, e]) {
            *slot = slot.wrapping_add(value);
        }
    }
    let mut out = [0u8; 20];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

fn base64(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// The `Sec-WebSocket-Accept` value a server must answer for a client key.
pub(super) fn accept_key(key: &str) -> String {
    base64(&sha1(format!("{key}{GUID}").as_bytes()))
}

// ---- frames ---------------------------------------------------------------------------------------------------

pub(super) const OP_CONT: u8 = 0;
pub(super) const OP_TEXT: u8 = 1;
pub(super) const OP_BINARY: u8 = 2;
pub(super) const OP_CLOSE: u8 = 8;
pub(super) const OP_PING: u8 = 9;
pub(super) const OP_PONG: u8 = 10;

#[derive(Debug, PartialEq, Eq)]
pub(super) struct Frame {
    pub fin: bool,
    pub opcode: u8,
    pub payload: Vec<u8>,
}

/// One frame as bytes. `mask` is `Some` for a frame a client sends (required by the protocol) and `None` for a server.
pub(super) fn encode_frame(opcode: u8, payload: &[u8], mask: Option<[u8; 4]>) -> Vec<u8> {
    let mut out = vec![0x80 | opcode];
    let flag = if mask.is_some() { 0x80 } else { 0 };
    match payload.len() {
        n if n < 126 => out.push(flag | n as u8),
        n if n <= 0xFFFF => {
            out.push(flag | 126);
            out.extend_from_slice(&(n as u16).to_be_bytes());
        }
        n => {
            out.push(flag | 127);
            out.extend_from_slice(&(n as u64).to_be_bytes());
        }
    }
    match mask {
        Some(key) => {
            out.extend_from_slice(&key);
            out.extend(payload.iter().enumerate().map(|(i, b)| b ^ key[i % 4]));
        }
        None => out.extend_from_slice(payload),
    }
    out
}

/// Takes one complete frame off the front of `buffer`; `Ok(None)` when more bytes are needed.
pub(super) fn parse_frame(buffer: &mut Vec<u8>) -> Result<Option<Frame>, String> {
    if buffer.len() < 2 {
        return Ok(None);
    }
    let fin = buffer[0] & 0x80 != 0;
    if buffer[0] & 0x70 != 0 {
        return Err("a frame uses an extension that was not negotiated".into());
    }
    let opcode = buffer[0] & 0x0F;
    let masked = buffer[1] & 0x80 != 0;
    let (length, mut at) = match buffer[1] & 0x7F {
        126 => {
            if buffer.len() < 4 {
                return Ok(None);
            }
            (u16::from_be_bytes([buffer[2], buffer[3]]) as usize, 4)
        }
        127 => {
            if buffer.len() < 10 {
                return Ok(None);
            }
            let mut raw = [0u8; 8];
            raw.copy_from_slice(&buffer[2..10]);
            (usize::try_from(u64::from_be_bytes(raw)).unwrap_or(usize::MAX), 10)
        }
        n => (n as usize, 2),
    };
    if length > MAX_MESSAGE {
        return Err("a message from the browser is too large".into());
    }
    let mut key = [0u8; 4];
    if masked {
        if buffer.len() < at + 4 {
            return Ok(None);
        }
        key.copy_from_slice(&buffer[at..at + 4]);
        at += 4;
    }
    if buffer.len() < at + length {
        return Ok(None);
    }
    let mut payload: Vec<u8> = buffer[at..at + length].to_vec();
    if masked {
        for (i, b) in payload.iter_mut().enumerate() {
            *b ^= key[i % 4];
        }
    }
    buffer.drain(..at + length);
    Ok(Some(Frame { fin, opcode, payload }))
}

// ---- the connection -------------------------------------------------------------------------------------------

/// An event of the browser (a message without `id`).
#[derive(Clone, Debug)]
pub(super) struct Event {
    pub method: String,
    pub params: Value,
    pub session: Option<String>,
}

type Pending = Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value, String>>>>>;

/// A CDP connection: commands with answers, and a stream of events.
pub(super) struct Cdp {
    writer: Arc<tokio::sync::Mutex<OwnedWriteHalf>>,
    next_id: AtomicU64,
    pending: Pending,
    events: broadcast::Sender<Event>,
    closed: Arc<AtomicBool>,
}

fn mask_key() -> [u8; 4] {
    let mut key = [0u8; 4];
    let _ = getrandom::fill(&mut key);
    key
}

impl Cdp {
    /// Connects to `ws://host:port/path` (plain TCP) and starts the reader.
    pub(super) async fn connect(host: &str, port: u16, path: &str) -> Result<Cdp, String> {
        let mut stream = tokio::time::timeout(Duration::from_secs(5), TcpStream::connect((host, port)))
            .await
            .map_err(|_| "the browser did not accept the connection in time".to_owned())?
            .map_err(|e| format!("cannot connect to the browser: {e}"))?;
        let mut raw = [0u8; 16];
        let _ = getrandom::fill(&mut raw);
        let key = base64(&raw);
        let request = format!(
            "GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        );
        stream.write_all(request.as_bytes()).await.map_err(|e| format!("cannot talk to the browser: {e}"))?;
        // the answer ends with an empty line; bytes behind it already belong to the first frames
        let mut head: Vec<u8> = Vec::new();
        let mut chunk = [0u8; 4096];
        let body_at = loop {
            if let Some(at) = head.windows(4).position(|w| w == b"\r\n\r\n") {
                break at + 4;
            }
            if head.len() > 16 * 1024 {
                return Err("the browser answered the handshake with too much data".into());
            }
            let n = tokio::time::timeout(Duration::from_secs(5), stream.read(&mut chunk))
                .await
                .map_err(|_| "the browser did not answer the handshake".to_owned())?
                .map_err(|e| format!("cannot read from the browser: {e}"))?;
            if n == 0 {
                return Err("the browser closed the connection during the handshake".into());
            }
            head.extend_from_slice(&chunk[..n]);
        };
        let text = String::from_utf8_lossy(&head[..body_at]).into_owned();
        if !text.lines().next().is_some_and(|l| l.contains(" 101")) {
            return Err(format!("the browser refused the WebSocket upgrade: {}", text.lines().next().unwrap_or("")));
        }
        let accepted = text
            .lines()
            .find_map(|l| l.split_once(':').filter(|(n, _)| n.eq_ignore_ascii_case("sec-websocket-accept")).map(|(_, v)| v.trim().to_owned()));
        if accepted.as_deref() != Some(accept_key(&key).as_str()) {
            return Err("the browser answered the WebSocket handshake with a wrong key".into());
        }
        let leftover = head[body_at..].to_vec();
        let (read, write) = stream.into_split();
        let writer = Arc::new(tokio::sync::Mutex::new(write));
        let pending: Pending = Arc::default();
        let (events, _) = broadcast::channel(2048);
        let closed = Arc::new(AtomicBool::new(false));
        tokio::spawn(read_loop(read, leftover, writer.clone(), pending.clone(), events.clone(), closed.clone()));
        Ok(Cdp { writer, next_id: AtomicU64::new(1), pending, events, closed })
    }

    pub(super) fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Relaxed)
    }

    pub(super) fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.events.subscribe()
    }

    /// Sends one command and waits for its answer.
    pub(super) async fn call(&self, method: &str, params: Value, session: Option<&str>, timeout: Duration) -> Result<Value, String> {
        if self.is_closed() {
            return Err("the browser connection is closed".into());
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().unwrap_or_else(|e| e.into_inner()).insert(id, tx);
        let mut message = json!({"id": id, "method": method, "params": params});
        if let Some(session) = session {
            message["sessionId"] = json!(session);
        }
        let frame = encode_frame(OP_TEXT, message.to_string().as_bytes(), Some(mask_key()));
        let sent = self.writer.lock().await.write_all(&frame).await;
        if let Err(e) = sent {
            self.pending.lock().unwrap_or_else(|e| e.into_inner()).remove(&id);
            return Err(format!("cannot send to the browser: {e}"));
        }
        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(answer)) => answer.map_err(|e| format!("{method}: {e}")),
            Ok(Err(_)) => Err("the browser connection closed".into()),
            Err(_) => {
                self.pending.lock().unwrap_or_else(|e| e.into_inner()).remove(&id);
                Err(format!("{method}: the browser did not answer within {} s", timeout.as_secs()))
            }
        }
    }

    pub(super) async fn close(&self) {
        let frame = encode_frame(OP_CLOSE, &[], Some(mask_key()));
        let _ = self.writer.lock().await.write_all(&frame).await;
        self.closed.store(true, Ordering::Relaxed);
    }
}

async fn read_loop(
    mut read: OwnedReadHalf,
    leftover: Vec<u8>,
    writer: Arc<tokio::sync::Mutex<OwnedWriteHalf>>,
    pending: Pending,
    events: broadcast::Sender<Event>,
    closed: Arc<AtomicBool>,
) {
    let mut buffer = leftover;
    let mut message: Vec<u8> = Vec::new();
    let mut chunk = vec![0u8; 64 * 1024];
    'outer: loop {
        loop {
            match parse_frame(&mut buffer) {
                Ok(Some(frame)) => match frame.opcode {
                    OP_PING => {
                        let pong = encode_frame(OP_PONG, &frame.payload, Some(mask_key()));
                        let _ = writer.lock().await.write_all(&pong).await;
                    }
                    OP_CLOSE => break 'outer,
                    OP_TEXT | OP_BINARY | OP_CONT => {
                        message.extend_from_slice(&frame.payload);
                        if message.len() > MAX_MESSAGE {
                            break 'outer;
                        }
                        if frame.fin {
                            dispatch(&message, &pending, &events);
                            message.clear();
                        }
                    }
                    _ => {}
                },
                Ok(None) => break,
                Err(_) => break 'outer,
            }
        }
        match read.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => buffer.extend_from_slice(&chunk[..n]),
        }
    }
    closed.store(true, Ordering::Relaxed);
    // everybody who still waits for an answer learns that none will come
    let waiting: Vec<_> = pending.lock().unwrap_or_else(|e| e.into_inner()).drain().collect();
    for (_, tx) in waiting {
        let _ = tx.send(Err("the browser connection closed".into()));
    }
}

fn dispatch(message: &[u8], pending: &Pending, events: &broadcast::Sender<Event>) {
    let Ok(value) = serde_json::from_slice::<Value>(message) else { return };
    if let Some(id) = value.get("id").and_then(Value::as_u64) {
        let tx = pending.lock().unwrap_or_else(|e| e.into_inner()).remove(&id);
        if let Some(tx) = tx {
            let answer = match value.get("error") {
                Some(error) => Err(error.get("message").and_then(Value::as_str).unwrap_or("the browser reported an error").to_owned()),
                None => Ok(value.get("result").cloned().unwrap_or(Value::Null)),
            };
            let _ = tx.send(answer);
        }
    } else if let Some(method) = value.get("method").and_then(Value::as_str) {
        let _ = events.send(Event {
            method: method.to_owned(),
            params: value.get("params").cloned().unwrap_or(Value::Null),
            session: value.get("sessionId").and_then(Value::as_str).map(str::to_owned),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn sha1_matches_the_published_vectors() {
        assert_eq!(hex(&sha1(b"abc")), "a9993e364706816aba3e25717850c26c9cd0d89d");
        assert_eq!(hex(&sha1(b"")), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
        assert_eq!(hex(&sha1(b"The quick brown fox jumps over the lazy dog")), "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12");
        // longer than one block
        assert_eq!(hex(&sha1(&[b'a'; 1000])), "291e9a6c66994949b57ba5e650361e98fc36b1ba");
    }

    #[test]
    fn the_accept_key_of_rfc_6455_is_reproduced() {
        assert_eq!(accept_key("dGhlIHNhbXBsZSBub25jZQ=="), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
    }

    #[test]
    fn frames_round_trip_with_and_without_a_mask_at_every_length_class() {
        for len in [0usize, 5, 125, 126, 300, 65535, 65536, 70000] {
            let payload: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            for mask in [None, Some([1, 2, 3, 4])] {
                let mut bytes = encode_frame(OP_TEXT, &payload, mask);
                let frame = parse_frame(&mut bytes).unwrap().unwrap();
                assert_eq!(frame, Frame { fin: true, opcode: OP_TEXT, payload: payload.clone() }, "length {len}");
                assert!(bytes.is_empty());
            }
        }
    }

    #[test]
    fn a_partial_frame_waits_for_more_bytes_and_two_frames_come_out_in_order() {
        let mut bytes = encode_frame(OP_TEXT, b"hello", None);
        bytes.extend(encode_frame(OP_PING, b"x", None));
        let whole = bytes.clone();
        let mut partial = whole[..3].to_vec();
        assert_eq!(parse_frame(&mut partial).unwrap(), None);
        assert_eq!(parse_frame(&mut bytes).unwrap().unwrap().payload, b"hello");
        assert_eq!(parse_frame(&mut bytes).unwrap().unwrap().opcode, OP_PING);
        assert_eq!(parse_frame(&mut bytes).unwrap(), None);
        // reserved bits mean an extension nobody agreed to
        let mut bad = vec![0xC1, 0x00];
        assert!(parse_frame(&mut bad).is_err());
    }

    /// A fake browser: accepts one connection, does the server side of the handshake, answers `Target.getTargets`
    /// with a message split in two fragments, sends an event first, pings once and checks that client frames are masked.
    async fn fake_browser() -> (u16, tokio::task::JoinHandle<bool>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let task = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut head = Vec::new();
            let mut byte = [0u8; 1];
            while !head.ends_with(b"\r\n\r\n") {
                socket.read_exact(&mut byte).await.unwrap();
                head.push(byte[0]);
            }
            let text = String::from_utf8_lossy(&head).into_owned();
            assert!(text.starts_with("GET /devtools/browser/abc HTTP/1.1"));
            let key = text.lines().find_map(|l| l.strip_prefix("Sec-WebSocket-Key: ")).unwrap().to_owned();
            let reply = format!("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {}\r\n\r\n", accept_key(&key));
            socket.write_all(reply.as_bytes()).await.unwrap();
            socket.write_all(&encode_frame(OP_TEXT, br#"{"method":"Target.targetCreated","params":{"n":1},"sessionId":"S1"}"#, None)).await.unwrap();
            socket.write_all(&encode_frame(OP_PING, b"are you there", None)).await.unwrap();
            let mut buffer = Vec::new();
            let mut saw_masked = false;
            let mut saw_pong = false;
            let mut chunk = [0u8; 4096];
            loop {
                let n = socket.read(&mut chunk).await.unwrap();
                if n == 0 {
                    break;
                }
                buffer.extend_from_slice(&chunk[..n]);
                while !buffer.is_empty() {
                    saw_masked |= buffer.len() > 1 && buffer[1] & 0x80 != 0;
                    let Some(frame) = parse_frame(&mut buffer).unwrap() else { break };
                    if frame.opcode == OP_PONG {
                        saw_pong = frame.payload == b"are you there";
                    }
                    if frame.opcode == OP_TEXT {
                        let request: Value = serde_json::from_slice(&frame.payload).unwrap();
                        let id = request["id"].as_u64().unwrap();
                        let answer = if request["method"] == "Boom" {
                            json!({"id": id, "error": {"code": -32000, "message": "no such thing"}})
                        } else {
                            json!({"id": id, "result": {"echo": request["method"], "session": request["sessionId"]}})
                        }
                        .to_string();
                        let (a, b) = answer.as_bytes().split_at(answer.len() / 2);
                        let mut first = encode_frame(OP_TEXT, a, None);
                        first[0] &= 0x7F; // not final
                        socket.write_all(&first).await.unwrap();
                        socket.write_all(&encode_frame(OP_CONT, b, None)).await.unwrap();
                    }
                    if frame.opcode == OP_CLOSE {
                        return saw_masked && saw_pong;
                    }
                }
            }
            saw_masked && saw_pong
        });
        (port, task)
    }

    #[tokio::test]
    async fn the_client_does_the_handshake_matches_answers_to_commands_and_delivers_events() {
        let (port, server) = fake_browser().await;
        let cdp = Cdp::connect("127.0.0.1", port, "/devtools/browser/abc").await.unwrap();
        let mut events = cdp.subscribe();
        let first = cdp.call("Target.getTargets", json!({}), Some("S9"), Duration::from_secs(5)).await.unwrap();
        assert_eq!(first["echo"], "Target.getTargets");
        assert_eq!(first["session"], "S9", "the session id travels with the command");
        let error = cdp.call("Boom", json!({}), None, Duration::from_secs(5)).await.unwrap_err();
        assert!(error.contains("no such thing") && error.contains("Boom"), "{error}");
        let event = events.recv().await.unwrap();
        assert_eq!((event.method.as_str(), event.session.as_deref()), ("Target.targetCreated", Some("S1")));
        assert_eq!(event.params["n"], 1);
        cdp.close().await;
        assert!(server.await.unwrap(), "client frames must be masked and a ping must be answered with a pong");
    }

    #[tokio::test]
    async fn a_wrong_accept_key_and_a_refused_upgrade_are_errors() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            for answer in ["HTTP/1.1 101 Switching Protocols\r\nSec-WebSocket-Accept: nope\r\n\r\n", "HTTP/1.1 403 Forbidden\r\n\r\n"] {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut chunk = [0u8; 2048];
                let _ = socket.read(&mut chunk).await;
                socket.write_all(answer.as_bytes()).await.unwrap();
            }
        });
        let wrong = Cdp::connect("127.0.0.1", port, "/x").await.err().unwrap();
        assert!(wrong.contains("wrong key"), "{wrong}");
        let refused = Cdp::connect("127.0.0.1", port, "/x").await.err().unwrap();
        assert!(refused.contains("refused"), "{refused}");
    }

    #[tokio::test]
    async fn waiting_commands_fail_when_the_browser_goes_away() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut head = Vec::new();
            let mut byte = [0u8; 1];
            while !head.ends_with(b"\r\n\r\n") {
                socket.read_exact(&mut byte).await.unwrap();
                head.push(byte[0]);
            }
            let text = String::from_utf8_lossy(&head).into_owned();
            let key = text.lines().find_map(|l| l.strip_prefix("Sec-WebSocket-Key: ")).unwrap().to_owned();
            socket
                .write_all(format!("HTTP/1.1 101 x\r\nSec-WebSocket-Accept: {}\r\n\r\n", accept_key(&key)).as_bytes())
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(100)).await;
            // the socket drops here
        });
        let cdp = Cdp::connect("127.0.0.1", port, "/x").await.unwrap();
        let error = cdp.call("Never", json!({}), None, Duration::from_secs(5)).await.unwrap_err();
        assert!(error.contains("closed"), "{error}");
        assert!(cdp.is_closed());
    }
}
