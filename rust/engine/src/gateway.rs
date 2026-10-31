//! Messaging gateway: talk to the coding agent from Telegram or Matrix (`fllm-rs gateway telegram|matrix`).
//!
//! The gateway polls the messenger (Telegram `getUpdates`, Matrix `/sync`), runs every text message of an allowed chat or room
//! through the same agent loop as `fllm-rs agent` (against the server named by `--url`, so tools, sandbox and hooks are those of
//! the Studio) and sends the answer back. Each chat has its own session (`agent-sessions/tg-<chat>.json`, `mx-<room>.json`),
//! `/new` starts a fresh one.
//!
//! Safety rules, all enforced here and tested:
//! * **Allowlist or nothing.** `--chat` is required; messages from any other chat or room are dropped without an answer and
//!   without a model call, so the bot cannot be used by whoever finds its name.
//! * **Permissions as in `agent`**: reading is always allowed, edits, commands, network and memory only with the matching
//!   `--allow-*` flag. A chat message cannot widen them.
//! * **Tokens come from the environment** (`FLLM_TELEGRAM_TOKEN`, `FLLM_MATRIX_TOKEN`), never from the command line, and never
//!   appear in an error.
//! * **No loops.** The Matrix gateway ignores its own messages and answers with `m.notice`, which bots must not react to.
//!
//! The position in the message stream (Telegram offset, Matrix `next_batch`) is kept in `gateway-<kind>.cursor` next to the
//! sessions, so a restart or a second `--once` run does not answer the same messages again. The first Matrix start only records the
//! position: what was written before is not answered.
//!
//! Each messenger implements the small [`Chat`] trait; the loop, the allowlist and the sessions do not know which one it is.
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde_json::{json, Value};

use crate::agent_cli::{self, Backend, LoopConfig, Status};

pub const HELP: &str = "usage: fllm-rs gateway telegram --chat ID[,ID...] [--url http://127.0.0.1:8000] [--model ID] [--api-key KEY]
       fllm-rs gateway matrix --homeserver URL --chat '!room:server'[,...] [same options]
                  [--folder DIR] [--profile NAME] [--allow-edit] [--allow-command] [--allow-network] [--allow-memory] [--yes]
                  [--max-steps N] [--max-tokens N] [--system TEXT] [--once]
Answers messages with the coding agent of a running fllm-rs server on this machine.
Tokens are read from the environment, never from the command line: FLLM_TELEGRAM_TOKEN (a bot from @BotFather) or
FLLM_MATRIX_TOKEN (the access token of the bot's Matrix account; FLLM_MATRIX_HOMESERVER may replace --homeserver).
--chat is required: only these chats (Telegram: numbers, a group has a minus sign; Matrix: room ids like !abc:server) are
answered, every other message is ignored. A Matrix invite to a listed room is accepted. Matrix rooms must not be encrypted:
the gateway cannot read encrypted messages. Permissions work as for `fllm-rs agent`: reading is always allowed, the rest only
with the matching --allow-* flag. In the chat: /new starts a fresh conversation, /help shows the commands.
--once handles the messages that are waiting and exits (for cron); without it the gateway runs until stopped. The position in
the message stream is kept in gateway-<kind>.cursor, so nothing is answered twice. The first Matrix start only records it.
FLLM_TELEGRAM_API changes the Bot API address (default https://api.telegram.org).";

const TELEGRAM_LIMIT: usize = 4000;
const MATRIX_LIMIT: usize = 8000;
const POLL_SECONDS: u64 = 25;
const CHAT_HELP: &str = "I run the fLLM coding agent on this server. Send a task as a message.\n/new - start a new conversation\n/help - this text";
const TELEGRAM_TOKEN_ENV: &str = "FLLM_TELEGRAM_TOKEN";
const MATRIX_TOKEN_ENV: &str = "FLLM_MATRIX_TOKEN";
const MATRIX_HOMESERVER_ENV: &str = "FLLM_MATRIX_HOMESERVER";
/// Only text messages (and, to count them, encrypted ones); no presence, no state, no ephemeral events.
const MATRIX_FILTER: &str = r#"{"presence":{"types":[]},"account_data":{"types":[]},"room":{"ephemeral":{"types":[]},"account_data":{"types":[]},"state":{"types":[]},"timeline":{"types":["m.room.message","m.room.encrypted"],"limit":50}}}"#;

/// A text message of an allowed or unknown chat, in the form every messenger is reduced to.
#[derive(Debug, Clone, PartialEq)]
pub struct Incoming {
    /// Telegram: the chat id; Matrix: the room id.
    pub key: String,
    pub text: String,
}

pub trait Chat {
    /// The messages after `cursor` (empty at the very first start) and the cursor to continue from; blocks up to the long-poll time.
    fn updates(&mut self, cursor: &str) -> Result<(Vec<Incoming>, String), String>;
    fn send(&mut self, key: &str, text: &str) -> Result<(), String>;
    /// The name of the session of a chat; a valid slug for [`agent_cli::valid_session_name`].
    fn session_name(&self, key: &str) -> String;
    /// The longest text one message may have.
    fn max_len(&self) -> usize;
}

pub struct GatewayConfig {
    pub allowed: BTreeSet<String>,
    pub loop_cfg: LoopConfig,
    pub folder: String,
    pub system_extra: Option<String>,
    pub sessions_dir: PathBuf,
    /// The persona (`SOUL.md`) of the profile, if any.
    pub soul: Option<String>,
}

#[derive(Debug, PartialEq)]
pub enum Handled {
    /// Not on the list, or empty.
    Ignored,
    Command,
    Answered,
    Failed,
}

/// Cuts a long answer into pieces a messenger accepts, at line breaks where possible, never inside a character.
pub fn split_message(text: &str, max: usize) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    for line in text.split_inclusive('\n') {
        let mut line = line;
        while current.chars().count() + line.chars().count() > max {
            if current.is_empty() {
                let cut: usize = line.char_indices().nth(max).map_or(line.len(), |(i, _)| i);
                parts.push(line[..cut].to_owned());
                line = &line[cut..];
            } else {
                parts.push(std::mem::take(&mut current));
            }
        }
        current.push_str(line);
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

fn reply(chat: &mut dyn Chat, key: &str, text: &str) {
    for part in split_message(text, chat.max_len()) {
        if let Err(e) = chat.send(key, &part) {
            eprintln!("[gateway] cannot send to {key}: {e}");
            return;
        }
    }
}

/// Handles one message. The allowlist is checked first: a stranger costs neither a model call nor an answer.
pub fn handle_update(incoming: &Incoming, cfg: &GatewayConfig, backend: &mut dyn Backend, chat: &mut dyn Chat) -> Handled {
    if !cfg.allowed.contains(&incoming.key) {
        return Handled::Ignored;
    }
    let text = incoming.text.trim();
    if text.is_empty() {
        return Handled::Ignored;
    }
    let key = incoming.key.as_str();
    let name = chat.session_name(key);
    match text.split_whitespace().next().unwrap_or_default().split('@').next().unwrap_or_default() {
        "/start" | "/help" => {
            reply(chat, key, CHAT_HELP);
            return Handled::Command;
        }
        "/new" => {
            let result = agent_cli::save_session_in(&cfg.sessions_dir, &name, &[]);
            reply(chat, key, &match result {
                Ok(()) => "New conversation started.".to_owned(),
                Err(e) => format!("Error: {e}"),
            });
            return Handled::Command;
        }
        _ => {}
    }
    let mut history = match agent_cli::load_session_in(&cfg.sessions_dir, &name) {
        Ok(h) => agent_cli::trim_history(h, 120),
        Err(e) => {
            reply(chat, key, &format!("Error: {e}"));
            return Handled::Failed;
        }
    };
    history.push(json!({"role": "user", "content": text}));
    let system = agent_cli::full_system(backend, &cfg.folder, &cfg.loop_cfg.allow, cfg.system_extra.as_deref(), cfg.soul.as_deref());
    match agent_cli::run_loop(backend, &cfg.loop_cfg, &system, history, &mut |_| {}) {
        Ok(outcome) => {
            let _ = agent_cli::save_session_in(&cfg.sessions_dir, &name, &outcome.messages);
            let mut answer = outcome.answer.trim_end().to_owned();
            if outcome.status == Status::OutOfSteps {
                answer.push_str("\n\n(stopped at the step limit)");
            }
            reply(chat, key, &answer);
            Handled::Answered
        }
        Err(e) => {
            reply(chat, key, &format!("Error: {e}"));
            Handled::Failed
        }
    }
}

/// One poll round: fetch, handle every message, return the cursor to continue from.
pub fn poll_once(cursor: &str, cfg: &GatewayConfig, backend: &mut dyn Backend, chat: &mut dyn Chat) -> Result<String, String> {
    let (messages, next) = chat.updates(cursor)?;
    for message in &messages {
        handle_update(message, cfg, backend, chat);
    }
    Ok(next)
}

pub fn cursor_file(base: &Path, kind: &str) -> PathBuf {
    base.join(format!("gateway-{kind}.cursor"))
}

pub fn load_cursor(file: &Path) -> String {
    std::fs::read_to_string(file).map(|t| t.trim().to_owned()).unwrap_or_default()
}

pub fn save_cursor(file: &Path, cursor: &str) -> Result<(), String> {
    if let Some(dir) = file.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let tmp = file.with_extension("cursor.tmp");
    std::fs::write(&tmp, cursor).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, file).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------------------------------------------
// Telegram
// ---------------------------------------------------------------------------------------------------------------

/// The text message of an update, if it is one.
pub fn incoming_from_update(update: &Value) -> Option<Incoming> {
    let message = &update["message"];
    let id = message["chat"]["id"].as_i64()?;
    let text = message["text"].as_str()?.trim();
    (!text.is_empty()).then(|| Incoming { key: id.to_string(), text: text.to_owned() })
}

/// The messages of a `getUpdates` answer and the next offset. The offset moves past every update, also past those that are
/// no text message, so nothing is fetched again.
pub fn telegram_batch(updates: &[Value], offset: i64) -> (Vec<Incoming>, i64) {
    let mut next = offset;
    let mut out = Vec::new();
    for update in updates {
        if let Some(id) = update["update_id"].as_i64() {
            next = next.max(id + 1);
        }
        if let Some(incoming) = incoming_from_update(update) {
            out.push(incoming);
        }
    }
    (out, next)
}

/// The session name of a Telegram chat: `tg-123`, and `tg-n100456` for a group (a session name is a slug, so the minus sign becomes `n`).
pub fn telegram_session(chat_id: i64) -> String {
    if chat_id < 0 { format!("tg-n{}", chat_id.unsigned_abs()) } else { format!("tg-{chat_id}") }
}

pub struct TelegramHttp {
    client: reqwest::blocking::Client,
    base: String,
    token: String,
    poll_seconds: u64,
}

impl TelegramHttp {
    pub fn new(api: &str, token: &str) -> Result<Self, String> {
        if token.trim().is_empty() || token.chars().any(|c| c.is_whitespace() || c == '/') {
            return Err(format!("{TELEGRAM_TOKEN_ENV} is not set or is not a bot token (create a bot with @BotFather)"));
        }
        let client = reqwest::blocking::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(POLL_SECONDS + 15))
            .redirect(reqwest::redirect::Policy::none())
            .user_agent("fllm-gateway/0.1")
            .build()
            .map_err(|e| e.without_url().to_string())?;
        Ok(Self { client, base: api.trim_end_matches('/').to_owned(), token: token.to_owned(), poll_seconds: POLL_SECONDS })
    }

    /// How long `getUpdates` may wait; 0 for `--once`, so an empty queue returns at once.
    pub fn with_poll_seconds(mut self, seconds: u64) -> Self {
        self.poll_seconds = seconds;
        self
    }

    /// The token sits in the URL, so the URL is stripped from every error.
    fn call(&self, method: &str, body: &Value) -> Result<Value, String> {
        let url = format!("{}/bot{}/{method}", self.base, self.token);
        let response = self.client.post(url).json(body).send().map_err(|e| e.without_url().to_string())?;
        let status = response.status();
        let parsed: Value = response.json().map_err(|e| e.without_url().to_string())?;
        if !status.is_success() || parsed["ok"] != json!(true) {
            let why = parsed["description"].as_str().unwrap_or("no description");
            return Err(format!("Telegram {method}: HTTP {status}: {why}"));
        }
        Ok(parsed["result"].clone())
    }
}

impl Chat for TelegramHttp {
    fn updates(&mut self, cursor: &str) -> Result<(Vec<Incoming>, String), String> {
        let offset = cursor.parse::<i64>().unwrap_or(0);
        let result = self.call("getUpdates", &json!({"offset": offset, "timeout": self.poll_seconds, "allowed_updates": ["message"]}))?;
        let updates = result.as_array().cloned().unwrap_or_default();
        let (messages, next) = telegram_batch(&updates, offset);
        Ok((messages, next.to_string()))
    }

    fn send(&mut self, key: &str, text: &str) -> Result<(), String> {
        let chat_id = key.parse::<i64>().map_err(|_| format!("'{key}' is not a Telegram chat id"))?;
        self.call("sendMessage", &json!({"chat_id": chat_id, "text": text, "disable_web_page_preview": true})).map(|_| ())
    }

    fn session_name(&self, key: &str) -> String {
        telegram_session(key.parse::<i64>().unwrap_or(0))
    }

    fn max_len(&self) -> usize {
        TELEGRAM_LIMIT
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Matrix
// ---------------------------------------------------------------------------------------------------------------

#[derive(Debug, Default, PartialEq)]
pub struct SyncBatch {
    pub events: Vec<Incoming>,
    pub invites: Vec<String>,
    /// Encrypted messages that could not be read.
    pub encrypted: usize,
    pub next: String,
}

/// Reads a `/sync` answer. Messages of the bot itself, notices (what bots send), edits and everything that is no plain text are left out.
pub fn parse_sync(sync: &Value, own_user: &str) -> SyncBatch {
    let mut batch = SyncBatch { next: sync["next_batch"].as_str().unwrap_or_default().to_owned(), ..SyncBatch::default() };
    if let Some(rooms) = sync["rooms"]["join"].as_object() {
        for (room, data) in rooms {
            for event in data["timeline"]["events"].as_array().into_iter().flatten() {
                match event["type"].as_str() {
                    Some("m.room.encrypted") if event["sender"].as_str() != Some(own_user) => batch.encrypted += 1,
                    Some("m.room.message") => {
                        let content = &event["content"];
                        let edit = content["m.relates_to"]["rel_type"].as_str() == Some("m.replace");
                        let (Some("m.text"), Some(body)) = (content["msgtype"].as_str(), content["body"].as_str()) else { continue };
                        if event["sender"].as_str() == Some(own_user) || edit || body.trim().is_empty() {
                            continue;
                        }
                        batch.events.push(Incoming { key: room.clone(), text: body.to_owned() });
                    }
                    _ => {}
                }
            }
        }
    }
    if let Some(invites) = sync["rooms"]["invite"].as_object() {
        batch.invites = invites.keys().cloned().collect();
    }
    batch
}

/// FNV-1a over the bytes: stable between runs and builds, which a session file name needs.
fn fnv1a(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3))
}

/// The session name of a Matrix room: `mx-` + a readable start of the room id + a hash of the whole id.
pub fn matrix_session(room: &str) -> String {
    let readable: String = room.chars().filter(|c| c.is_ascii_alphanumeric()).take(16).collect::<String>().to_ascii_lowercase();
    format!("mx-{readable}-{:016x}", fnv1a(room))
}

pub struct MatrixHttp {
    client: reqwest::blocking::Client,
    base: reqwest::Url,
    token: String,
    own_user: String,
    allowed: BTreeSet<String>,
    poll_ms: u64,
}

static TXN: AtomicU64 = AtomicU64::new(0);

impl MatrixHttp {
    /// Checks the token with `whoami` (the user id is needed to ignore the bot's own messages).
    pub fn connect(homeserver: &str, token: &str, allowed: BTreeSet<String>) -> Result<Self, String> {
        if token.trim().is_empty() || token.chars().any(char::is_whitespace) {
            return Err(format!("{MATRIX_TOKEN_ENV} is not set or is not an access token"));
        }
        let base = reqwest::Url::parse(homeserver).map_err(|e| format!("--homeserver is not an address: {e}"))?;
        if !matches!(base.scheme(), "http" | "https") || base.host_str().is_none() {
            return Err("--homeserver must be an http:// or https:// address".into());
        }
        let client = reqwest::blocking::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(POLL_SECONDS + 20))
            .redirect(reqwest::redirect::Policy::none())
            .user_agent("fllm-gateway/0.1")
            .build()
            .map_err(|e| e.without_url().to_string())?;
        let mut me = Self { client, base, token: token.to_owned(), own_user: String::new(), allowed, poll_ms: POLL_SECONDS * 1000 };
        let who = me.request(reqwest::Method::GET, &["account", "whoami"], &[], None)?;
        me.own_user = who["user_id"].as_str().ok_or("the homeserver did not say who the token belongs to")?.to_owned();
        Ok(me)
    }

    pub fn with_poll_seconds(mut self, seconds: u64) -> Self {
        self.poll_ms = seconds * 1000;
        self
    }

    pub fn own_user(&self) -> &str {
        &self.own_user
    }

    fn url(&self, segments: &[&str], query: &[(&str, String)]) -> Result<reqwest::Url, String> {
        let mut url = self.base.clone();
        {
            let mut path = url.path_segments_mut().map_err(|()| "the homeserver address cannot hold a path".to_owned())?;
            path.pop_if_empty().extend(["_matrix", "client", "v3"]).extend(segments);
        }
        if !query.is_empty() {
            let mut pairs = url.query_pairs_mut();
            for (name, value) in query {
                pairs.append_pair(name, value);
            }
        }
        Ok(url)
    }

    fn request(&self, method: reqwest::Method, segments: &[&str], query: &[(&str, String)], body: Option<&Value>) -> Result<Value, String> {
        let url = self.url(segments, query)?;
        let mut request = self.client.request(method, url).bearer_auth(&self.token);
        if let Some(body) = body {
            request = request.json(body);
        }
        let response = request.send().map_err(|e| format!("cannot reach the homeserver: {}", e.without_url()))?;
        let status = response.status();
        let text = response.text().map_err(|e| e.without_url().to_string())?;
        let parsed: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
        if !status.is_success() {
            let code = parsed["errcode"].as_str().unwrap_or("-");
            let why = parsed["error"].as_str().unwrap_or("no description");
            return Err(format!("Matrix {}: HTTP {status}: {code}: {why}", segments.join("/")));
        }
        Ok(parsed)
    }
}

impl Chat for MatrixHttp {
    fn updates(&mut self, cursor: &str) -> Result<(Vec<Incoming>, String), String> {
        let baseline = cursor.is_empty();
        let mut query = vec![("timeout", if baseline { 0 } else { self.poll_ms }.to_string()), ("filter", MATRIX_FILTER.to_owned())];
        if !baseline {
            query.push(("since", cursor.to_owned()));
        }
        let sync = self.request(reqwest::Method::GET, &["sync"], &query, None)?;
        let batch = parse_sync(&sync, &self.own_user);
        for room in batch.invites.iter().filter(|r| self.allowed.contains(*r)) {
            if let Err(e) = self.request(reqwest::Method::POST, &["join", room], &[], Some(&json!({}))) {
                eprintln!("[gateway] cannot join {room}: {e}");
            }
        }
        if batch.encrypted > 0 {
            eprintln!("[gateway] {} encrypted message(s) ignored: the gateway cannot read encrypted rooms", batch.encrypted);
        }
        let next = if batch.next.is_empty() { cursor.to_owned() } else { batch.next };
        // the very first sync only records the position: old messages are not answered
        Ok((if baseline { Vec::new() } else { batch.events }, next))
    }

    fn send(&mut self, key: &str, text: &str) -> Result<(), String> {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        let txn = format!("fllm-{nanos}-{}", TXN.fetch_add(1, Ordering::Relaxed));
        // m.notice: what bots send, so that no other bot answers it
        self.request(reqwest::Method::PUT, &["rooms", key, "send", "m.room.message", &txn], &[], Some(&json!({"msgtype": "m.notice", "body": text}))).map(|_| ())
    }

    fn session_name(&self, key: &str) -> String {
        matrix_session(key)
    }

    fn max_len(&self) -> usize {
        MATRIX_LIMIT
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------------------------------------------

pub struct Options {
    pub chats: BTreeSet<String>,
    pub homeserver: Option<String>,
    pub agent: Vec<String>,
    pub once: bool,
}

fn valid_room(room: &str) -> bool {
    room.starts_with('!') && room.len() <= 255 && room.split_once(':').is_some_and(|(id, server)| id.len() > 1 && !server.is_empty()) && !room.chars().any(char::is_whitespace)
}

/// Splits the gateway arguments: `--chat`, `--homeserver` and `--once` are the gateway's own, everything else is an agent option.
pub fn parse_args(kind: &str, args: &[String]) -> Result<Options, String> {
    let mut chats = BTreeSet::new();
    let mut homeserver = None;
    let mut agent = Vec::new();
    let mut once = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--chat" => {
                i += 1;
                let list = args.get(i).ok_or("missing value for --chat")?;
                for item in list.split(',').map(str::trim) {
                    if kind == "matrix" {
                        if !valid_room(item) {
                            return Err(format!("'{item}' is not a Matrix room id (like !abc:server; an alias such as #room:server is not accepted)"));
                        }
                        chats.insert(item.to_owned());
                    } else {
                        let id = item.parse::<i64>().map_err(|_| format!("'{item}' is not a chat id (a number; a group has a minus sign)"))?;
                        chats.insert(id.to_string());
                    }
                }
            }
            "--homeserver" => {
                i += 1;
                homeserver = Some(args.get(i).ok_or("missing value for --homeserver")?.clone());
            }
            "--once" => once = true,
            "-p" | "--prompt" | "--session" | "--trajectory" | "--json" | "--ask" => return Err(format!("{} does not apply to the gateway", args[i])),
            other => {
                agent.push(other.to_owned());
                // options that take a value
                if matches!(other, "--url" | "--model" | "--api-key" | "--folder" | "--cwd" | "--system" | "--max-steps" | "--max-tokens" | "--temperature" | "--profile") {
                    i += 1;
                    agent.push(args.get(i).ok_or_else(|| format!("missing value for {other}"))?.clone());
                }
            }
        }
        i += 1;
    }
    if chats.is_empty() {
        return Err("--chat is required: name the chats that may talk to the agent (without a list the bot would answer everyone)".into());
    }
    if kind == "matrix" {
        homeserver = homeserver.or_else(|| std::env::var(MATRIX_HOMESERVER_ENV).ok().filter(|h| !h.is_empty()));
        if homeserver.is_none() {
            return Err(format!("--homeserver (or {MATRIX_HOMESERVER_ENV}) is required for Matrix"));
        }
    } else if homeserver.is_some() {
        return Err("--homeserver only applies to Matrix".into());
    }
    Ok(Options { chats, homeserver, agent, once })
}

pub fn run(args: &[String]) -> Result<i32, String> {
    if args.iter().any(|a| a == "--help" || a == "-h") || args.is_empty() {
        println!("{HELP}");
        return Ok(0);
    }
    let kind = args[0].as_str();
    if !matches!(kind, "telegram" | "matrix") {
        return Err(format!("unknown gateway '{kind}' (telegram, matrix)"));
    }
    let options = parse_args(kind, &args[1..])?;
    // reuse the option parser of `agent` for url, model, permissions and limits (a placeholder task satisfies it)
    let mut agent_args = options.agent.clone();
    agent_args.extend(["-p".to_owned(), "gateway".to_owned()]);
    let parsed = agent_cli::parse_args(&agent_args)?;
    let poll = if options.once { 0 } else { POLL_SECONDS };
    let mut chat: Box<dyn Chat> = if kind == "telegram" {
        let token = std::env::var(TELEGRAM_TOKEN_ENV).unwrap_or_default();
        let api = std::env::var("FLLM_TELEGRAM_API").unwrap_or_else(|_| "https://api.telegram.org".to_owned());
        Box::new(TelegramHttp::new(&api, &token)?.with_poll_seconds(poll))
    } else {
        let token = std::env::var(MATRIX_TOKEN_ENV).unwrap_or_default();
        let homeserver = options.homeserver.clone().unwrap_or_default();
        Box::new(MatrixHttp::connect(&homeserver, &token, options.chats.clone())?.with_poll_seconds(poll))
    };

    let folder = match &parsed.folder {
        Some(f) => f.clone(),
        None => std::env::current_dir().map_err(|e| e.to_string())?,
    };
    let folder = folder.canonicalize().map_err(|e| format!("the folder {} cannot be used: {e}", folder.display()))?;
    let mut http = agent_cli::HttpBackend::new(&parsed.url, parsed.api_key.clone())?;
    let base = agent_cli::profile_base(parsed.profile.as_deref())?;
    let (served, context) = http.model_info()?;
    let model = parsed.model.clone().unwrap_or(served);
    http.set_folder(&folder.display().to_string())?;
    if let Ok(tools) = http.tools() {
        if let Some(warning) = agent_cli::context_warning(tools.iter().map(|t| t.to_string().len()).sum(), context) {
            eprintln!("[warning] {warning}");
        }
    }
    let cfg = GatewayConfig {
        allowed: options.chats,
        loop_cfg: LoopConfig { model: Some(model), allow: parsed.allow.clone(), rules: crate::agent_rules::load_hierarchy(&agent_cli::studio_folder(), &folder)?, max_steps: parsed.max_steps, max_tokens: agent_cli::effective_max_tokens(parsed.max_tokens, context), temperature: parsed.temperature },
        folder: folder.display().to_string(),
        system_extra: parsed.system.clone(),
        sessions_dir: base.clone(),
        soul: agent_cli::load_soul(&base),
    };
    eprintln!("[gateway] {kind}: {} chat(s) allowed, folder {}, permissions {:?}", cfg.allowed.len(), cfg.folder, cfg.loop_cfg.allow);
    let file = cursor_file(&base, kind);
    let mut cursor = load_cursor(&file);
    loop {
        match poll_once(&cursor, &cfg, &mut http, &mut *chat) {
            Ok(next) => {
                if next != cursor {
                    if let Err(e) = save_cursor(&file, &next) {
                        eprintln!("[gateway] cannot save the position ({e}); after a restart messages may be answered twice");
                    }
                    cursor = next;
                }
            }
            Err(e) => {
                eprintln!("[gateway] {e}");
                if options.once {
                    return Ok(1);
                }
                std::thread::sleep(Duration::from_secs(5));
            }
        }
        if options.once {
            http.end();
            return Ok(0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::io::{Read, Write};

    struct FakeChat {
        incoming: VecDeque<(Vec<Incoming>, String)>,
        sent: Vec<(String, String)>,
        cursors: Vec<String>,
    }

    impl FakeChat {
        fn new() -> Self {
            Self { incoming: VecDeque::new(), sent: vec![], cursors: vec![] }
        }
    }

    impl Chat for FakeChat {
        fn updates(&mut self, cursor: &str) -> Result<(Vec<Incoming>, String), String> {
            self.cursors.push(cursor.to_owned());
            Ok(self.incoming.pop_front().unwrap_or((vec![], cursor.to_owned())))
        }
        fn send(&mut self, key: &str, text: &str) -> Result<(), String> {
            self.sent.push((key.to_owned(), text.to_owned()));
            Ok(())
        }
        fn session_name(&self, key: &str) -> String {
            telegram_session(key.parse::<i64>().unwrap_or(0))
        }
        fn max_len(&self) -> usize {
            4000
        }
    }

    struct Model {
        replies: VecDeque<Value>,
        bodies: Vec<Value>,
    }

    impl Backend for Model {
        fn tools(&mut self) -> Result<Vec<Value>, String> {
            Ok(vec![json!({"type": "function", "function": {"name": "read_file", "description": "d", "parameters": {"type": "object", "properties": {}}}, "permission": "read"})])
        }
        fn complete(&mut self, body: Value) -> Result<Value, String> {
            self.bodies.push(body);
            self.replies.pop_front().ok_or_else(|| "no more replies".to_owned())
        }
        fn run_tool(&mut self, name: &str, _args: &Value) -> Result<String, String> {
            Ok(format!("ran {name}"))
        }
    }

    fn answer(t: &str) -> Value {
        json!({"choices": [{"message": {"role": "assistant", "content": t}}]})
    }

    fn msg(chat: i64, text: &str) -> Incoming {
        Incoming { key: chat.to_string(), text: text.to_owned() }
    }

    fn config(dir: &Path, allowed: &[i64]) -> GatewayConfig {
        GatewayConfig {
            allowed: allowed.iter().map(|c| c.to_string()).collect(),
            loop_cfg: LoopConfig { model: Some("m".into()), allow: BTreeSet::from(["read".to_owned()]), rules: Default::default(), max_steps: 4, max_tokens: 50, temperature: 0.0 },
            folder: "/work".into(),
            system_extra: None,
            sessions_dir: dir.to_path_buf(),
            soul: None,
        }
    }

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fllm-gw-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn s(items: &[&str]) -> Vec<String> {
        items.iter().map(|x| (*x).to_owned()).collect()
    }

    #[test]
    fn only_allowed_chats_are_answered_and_the_model_is_not_even_asked_for_others() {
        let dir = tmp("allow");
        let cfg = config(&dir, &[42]);
        let mut model = Model { replies: VecDeque::from([answer("hello 42")]), bodies: vec![] };
        let mut chat = FakeChat::new();
        assert_eq!(handle_update(&msg(99, "do things"), &cfg, &mut model, &mut chat), Handled::Ignored);
        assert!(model.bodies.is_empty() && chat.sent.is_empty(), "a stranger gets no answer and costs no model call");
        assert_eq!(handle_update(&msg(42, "hi"), &cfg, &mut model, &mut chat), Handled::Answered);
        assert_eq!(chat.sent, vec![("42".to_owned(), "hello 42".to_owned())]);
        assert_eq!(handle_update(&msg(42, "   "), &cfg, &mut model, &mut chat), Handled::Ignored);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn telegram_updates_become_messages_and_odd_ones_are_skipped() {
        let text = |id: i64, chat: i64, t: &str| json!({"update_id": id, "message": {"chat": {"id": chat}, "text": t}});
        assert_eq!(incoming_from_update(&text(1, 5, " hi ")), Some(msg(5, "hi")));
        for odd in [json!({}), json!({"update_id": 1}), json!({"update_id": 2, "message": {"chat": {"id": 1}}}), text(3, 1, "   "), json!({"update_id": 4, "message": {"chat": {"id": "x"}, "text": "a"}}), json!({"update_id": 5, "edited_message": {"chat": {"id": 1}, "text": "a"}})] {
            assert_eq!(incoming_from_update(&odd), None, "{odd}");
        }
    }

    #[test]
    fn the_telegram_offset_moves_past_every_update_even_those_that_are_no_text() {
        let updates = vec![json!({"update_id": 10, "message": {"chat": {"id": 3}, "text": "x"}}), json!({"update_id": 11, "message": {"chat": {"id": 3}}}), json!({"update_id": 7})];
        let (messages, next) = telegram_batch(&updates, 0);
        assert_eq!((messages, next), (vec![msg(3, "x")], 12), "a photo or an odd update must not be fetched again");
        assert_eq!(telegram_batch(&[], 12), (vec![], 12));
    }

    #[test]
    fn a_chat_keeps_its_conversation_and_new_clears_it() {
        let dir = tmp("session");
        let cfg = config(&dir, &[7]);
        let mut model = Model { replies: VecDeque::from([answer("one"), answer("two"), answer("three")]), bodies: vec![] };
        let mut chat = FakeChat::new();
        handle_update(&msg(7, "first"), &cfg, &mut model, &mut chat);
        handle_update(&msg(7, "second"), &cfg, &mut model, &mut chat);
        let users = |b: &Value| b["messages"].as_array().unwrap().iter().filter(|m| m["role"] == "user").map(|m| m["content"].as_str().unwrap().to_owned()).collect::<Vec<_>>();
        assert_eq!(users(&model.bodies[1]), vec!["first", "second"], "the second message sees the first");
        assert_eq!(handle_update(&msg(7, "/new"), &cfg, &mut model, &mut chat), Handled::Command);
        handle_update(&msg(7, "third"), &cfg, &mut model, &mut chat);
        assert_eq!(users(&model.bodies[2]), vec!["third"], "/new starts clean");
        assert!(chat.sent.iter().any(|(_, t)| t.contains("New conversation")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn session_names_are_valid_slugs_and_chats_do_not_share_them() {
        assert_eq!(telegram_session(5), "tg-5");
        assert_eq!(telegram_session(-100123), "tg-n100123");
        assert!(agent_cli::valid_session_name(&telegram_session(-100123)) && agent_cli::valid_session_name(&telegram_session(0)));
        let (a, b) = (matrix_session("!abc:matrix.example"), matrix_session("!abd:matrix.example"));
        assert!(agent_cli::valid_session_name(&a), "{a}");
        assert_ne!(a, b);
        assert_eq!(a, matrix_session("!abc:matrix.example"), "stable between calls");
        assert!(agent_cli::valid_session_name(&matrix_session("!ÄÖÜ:s")) && agent_cli::valid_session_name(&matrix_session(&format!("!{}:s", "x".repeat(300)))));
        let dir = tmp("sep");
        let cfg = config(&dir, &[1, -2]);
        let mut model = Model { replies: VecDeque::from([answer("a"), answer("b")]), bodies: vec![] };
        let mut chat = FakeChat::new();
        handle_update(&msg(1, "private"), &cfg, &mut model, &mut chat);
        handle_update(&msg(-2, "group"), &cfg, &mut model, &mut chat);
        let seen: Vec<String> = model.bodies[1]["messages"].as_array().unwrap().iter().filter(|m| m["role"] == "user").map(|m| m["content"].as_str().unwrap().to_owned()).collect();
        assert_eq!(seen, vec!["group"], "the group does not see the private chat");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_model_error_is_reported_in_the_chat_and_help_works() {
        let dir = tmp("err");
        let cfg = config(&dir, &[3]);
        let mut model = Model { replies: VecDeque::new(), bodies: vec![] };
        let mut chat = FakeChat::new();
        assert_eq!(handle_update(&msg(3, "go"), &cfg, &mut model, &mut chat), Handled::Failed);
        assert!(chat.sent[0].1.starts_with("Error:"));
        assert_eq!(handle_update(&msg(3, "/help@my_bot"), &cfg, &mut model, &mut chat), Handled::Command);
        assert!(chat.sent[1].1.contains("/new"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_persona_and_the_project_instructions_reach_the_model() {
        struct Instr(Model);
        impl Backend for Instr {
            fn tools(&mut self) -> Result<Vec<Value>, String> { self.0.tools() }
            fn complete(&mut self, body: Value) -> Result<Value, String> { self.0.complete(body) }
            fn run_tool(&mut self, n: &str, a: &Value) -> Result<String, String> { self.0.run_tool(n, a) }
            fn instructions(&mut self) -> Result<Value, String> { Ok(json!({"files": [{"file": "AGENTS.md", "text": "Run make test."}]})) }
        }
        let dir = tmp("soul");
        let mut cfg = config(&dir, &[8]);
        cfg.soul = Some("Be direct.".into());
        let mut model = Instr(Model { replies: VecDeque::from([answer("ok")]), bodies: vec![] });
        let mut chat = FakeChat::new();
        handle_update(&msg(8, "hi"), &cfg, &mut model, &mut chat);
        let system = model.0.bodies[0]["messages"][0]["content"].as_str().unwrap().to_owned();
        assert!(system.contains("# Persona\nBe direct.") && system.contains("Run make test."), "{system}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn poll_once_hands_over_the_cursor_and_every_message() {
        let dir = tmp("poll");
        let cfg = config(&dir, &[3]);
        let mut model = Model { replies: VecDeque::new(), bodies: vec![] };
        let mut chat = FakeChat::new();
        chat.incoming.push_back((vec![msg(3, "x"), msg(99, "y")], "12".to_owned()));
        let next = poll_once("0", &cfg, &mut model, &mut chat).unwrap();
        assert_eq!(next, "12", "a failing message must not be fetched again");
        assert_eq!(poll_once(&next, &cfg, &mut model, &mut chat).unwrap(), "12");
        assert_eq!(chat.cursors, vec!["0", "12"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_cursor_survives_a_restart() {
        let dir = tmp("cursor");
        let file = cursor_file(&dir, "telegram");
        assert_eq!(load_cursor(&file), "", "no file: start at the beginning");
        save_cursor(&file, "s72_1_2").unwrap();
        assert_eq!(load_cursor(&file), "s72_1_2");
        save_cursor(&file, "s73").unwrap();
        assert_eq!(load_cursor(&file), "s73");
        assert_ne!(cursor_file(&dir, "matrix"), file, "each messenger has its own position");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn long_answers_are_split_without_losing_text() {
        let text: String = (0..300).map(|i| format!("line {i} ääää\n")).collect();
        let parts = split_message(&text, 500);
        assert!(parts.len() > 1 && parts.iter().all(|p| p.chars().count() <= 500));
        assert_eq!(parts.concat(), text, "joining the pieces gives the original");
        let one_long = "ü".repeat(1200);
        let parts = split_message(&one_long, 500);
        assert_eq!(parts.iter().map(|p| p.chars().count()).collect::<Vec<_>>(), vec![500, 500, 200]);
        assert_eq!(parts.concat(), one_long);
        assert!(split_message("", 10).is_empty());
    }

    #[test]
    fn arguments_need_a_chat_list_and_keep_the_agent_options() {
        assert!(parse_args("telegram", &s(&[])).is_err(), "no allowlist, no bot");
        assert!(parse_args("telegram", &s(&["--chat", "abc"])).is_err());
        assert!(parse_args("telegram", &s(&["--chat", "1", "-p", "x"])).is_err());
        assert!(parse_args("telegram", &s(&["--chat", "1", "--ask"])).is_err(), "nobody can answer a question in a chat");
        assert!(parse_args("telegram", &s(&["--chat", "1", "--homeserver", "https://h"])).is_err());
        let o = parse_args("telegram", &s(&["--chat", "5, -100200", "--allow-edit", "--max-steps", "9", "--once"])).unwrap();
        assert_eq!(o.chats, BTreeSet::from(["5".to_owned(), "-100200".to_owned()]));
        assert!(o.once);
        let parsed = agent_cli::parse_args(&[o.agent.clone(), s(&["-p", "g"])].concat()).unwrap();
        assert!(parsed.allow.contains("edit") && !parsed.allow.contains("command"));
        assert_eq!(parsed.max_steps, 9);
    }

    #[test]
    fn matrix_arguments_need_room_ids_and_a_homeserver() {
        let ok = parse_args("matrix", &s(&["--chat", "!abc:matrix.example, !def:other.example", "--homeserver", "https://matrix.example", "--profile", "work"])).unwrap();
        assert_eq!(ok.chats.len(), 2);
        assert_eq!(ok.homeserver.as_deref(), Some("https://matrix.example"));
        assert!(ok.agent.contains(&"--profile".to_owned()), "agent options pass through");
        for bad in ["#room:server", "abc:server", "!abc", "!:server", "!abc:", "!a b:server", "42"] {
            assert!(parse_args("matrix", &s(&["--chat", bad, "--homeserver", "https://h"])).is_err(), "{bad} is no room id");
        }
        if std::env::var_os(MATRIX_HOMESERVER_ENV).is_none() {
            assert!(parse_args("matrix", &s(&["--chat", "!a:b"])).is_err(), "no homeserver, no connection");
        }
    }

    // ---- Matrix: pure parsing ----

    fn event(kind: &str, sender: &str, content: Value) -> Value {
        json!({"type": kind, "sender": sender, "event_id": "$e", "content": content})
    }

    #[test]
    fn a_sync_answer_yields_the_text_messages_of_others_in_order() {
        let sync = json!({"next_batch": "s2", "rooms": {
            "join": {"!room:srv": {"timeline": {"events": [
                event("m.room.message", "@marco:srv", json!({"msgtype": "m.text", "body": "first"})),
                event("m.room.message", "@bot:srv", json!({"msgtype": "m.text", "body": "my own words"})),
                event("m.room.message", "@other:srv", json!({"msgtype": "m.notice", "body": "a bot said this"})),
                event("m.room.message", "@marco:srv", json!({"msgtype": "m.image", "body": "cat.png"})),
                event("m.room.message", "@marco:srv", json!({"msgtype": "m.text", "body": "* first, edited", "m.relates_to": {"rel_type": "m.replace", "event_id": "$x"}})),
                event("m.room.message", "@marco:srv", json!({"msgtype": "m.text", "body": "  "})),
                event("m.room.encrypted", "@marco:srv", json!({"algorithm": "m.megolm.v1.aes-sha2"})),
                event("m.room.encrypted", "@bot:srv", json!({})),
                event("m.room.message", "@marco:srv", json!({"msgtype": "m.text", "body": "second"})),
                event("m.room.topic", "@marco:srv", json!({"topic": "x"})),
            ]}}},
            "invite": {"!invited:srv": {}}}});
        let batch = parse_sync(&sync, "@bot:srv");
        assert_eq!(batch.events, vec![Incoming { key: "!room:srv".into(), text: "first".into() }, Incoming { key: "!room:srv".into(), text: "second".into() }]);
        assert_eq!((batch.next.as_str(), batch.encrypted, batch.invites.clone()), ("s2", 1, vec!["!invited:srv".to_owned()]));
        assert_eq!(parse_sync(&json!({}), "@bot:srv"), SyncBatch::default());
    }

    // ---- HTTP stubs ----

    fn read_request(stream: &mut std::net::TcpStream) -> String {
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
        String::from_utf8_lossy(&got).into_owned()
    }

    /// What the stub received on one connection: `(request line, headers, body)`.
    type Seen = (String, String, String);

    /// Serves the given answers in order, one connection each; returns what it received as `(request line, headers, body)`.
    fn stub(answers: Vec<(u16, Value)>) -> (String, std::thread::JoinHandle<Vec<Seen>>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let handle = std::thread::spawn(move || {
            let mut seen = Vec::new();
            for (status, payload) in answers {
                let (mut stream, _) = listener.accept().unwrap();
                let request = read_request(&mut stream);
                let (head, body) = request.split_once("\r\n\r\n").unwrap_or((&request, ""));
                let (line, headers) = head.split_once("\r\n").unwrap_or((head, ""));
                let payload = payload.to_string();
                stream.write_all(format!("HTTP/1.1 {status} X\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{payload}", payload.len()).as_bytes()).unwrap();
                seen.push((line.to_owned(), headers.to_ascii_lowercase(), body.to_owned()));
            }
            seen
        });
        (url, handle)
    }

    #[test]
    fn a_telegram_message_travels_through_the_http_client_the_loop_and_back() {
        let dir = tmp("http");
        let (api, server) = stub(vec![
            (200, json!({"ok": true, "result": [{"update_id": 5, "message": {"chat": {"id": 42}, "text": "hello bot"}}]})),
            (200, json!({"ok": true, "result": {"message_id": 1}})),
        ]);
        let mut chat = TelegramHttp::new(&api, "123:ABC").unwrap().with_poll_seconds(0);
        let cfg = config(&dir, &[42]);
        let mut model = Model { replies: VecDeque::from([answer("pong")]), bodies: vec![] };
        let next = poll_once("0", &cfg, &mut model, &mut chat).unwrap();
        assert_eq!(next, "6");
        let seen = server.join().unwrap();
        assert!(seen[0].0.contains("/bot123:ABC/getUpdates") && seen[1].0.contains("/bot123:ABC/sendMessage"), "{seen:?}");
        let ask: Value = serde_json::from_str(&seen[0].2).unwrap();
        assert_eq!((ask["offset"].as_i64(), ask["timeout"].as_u64()), (Some(0), Some(0)), "--once does not wait");
        let sent: Value = serde_json::from_str(&seen[1].2).unwrap();
        assert_eq!((sent["chat_id"].as_i64(), sent["text"].as_str()), (Some(42), Some("pong")));
        assert!(!seen[1].2.contains("ABC"), "the token is not in the message body");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_bad_telegram_token_is_refused_and_errors_never_contain_it() {
        assert!(TelegramHttp::new("http://x", "").is_err());
        assert!(TelegramHttp::new("http://x", "a/b").is_err());
        let mut chat = TelegramHttp::new("http://127.0.0.1:1", "999:SECRET-TOKEN").unwrap();
        let err = chat.updates("0").unwrap_err();
        assert!(!err.contains("SECRET-TOKEN"), "{err}");
        let err = chat.send("1", "x").unwrap_err();
        assert!(!err.contains("SECRET-TOKEN"), "{err}");
    }

    #[test]
    fn matrix_connects_syncs_answers_and_does_not_answer_the_first_sync() {
        let dir = tmp("matrix");
        let room = "!room:srv";
        let (hs, server) = stub(vec![
            (200, json!({"user_id": "@bot:srv"})),
            // first sync: a message that was written long ago must not be answered
            (200, json!({"next_batch": "s1", "rooms": {"join": {room: {"timeline": {"events": [event("m.room.message", "@marco:srv", json!({"msgtype": "m.text", "body": "old"}))]}}}, "invite": {"!mine:srv": {}, "!stranger:srv": {}}}})),
            (200, json!({"room_id": "!mine:srv"})),
            // second sync: a new message
            (200, json!({"next_batch": "s2", "rooms": {"join": {room: {"timeline": {"events": [event("m.room.message", "@marco:srv", json!({"msgtype": "m.text", "body": "ping"}))]}}}}})),
            (200, json!({"event_id": "$sent"})),
        ]);
        let allowed = BTreeSet::from([room.to_owned(), "!mine:srv".to_owned()]);
        let mut chat = MatrixHttp::connect(&hs, "syt_SECRET", allowed.clone()).unwrap().with_poll_seconds(0);
        assert_eq!(chat.own_user(), "@bot:srv");
        let mut cfg = config(&dir, &[]);
        cfg.allowed = allowed;
        let mut model = Model { replies: VecDeque::from([answer("pong")]), bodies: vec![] };
        let first = poll_once("", &cfg, &mut model, &mut chat).unwrap();
        assert_eq!(first, "s1");
        assert!(model.bodies.is_empty(), "the old message is not answered");
        let second = poll_once(&first, &cfg, &mut model, &mut chat).unwrap();
        assert_eq!(second, "s2");
        let seen = server.join().unwrap();
        assert!(seen[0].0.starts_with("GET /_matrix/client/v3/account/whoami"), "{:?}", seen[0]);
        assert!(seen[0].1.contains("authorization: bearer syt_secret"), "the token goes in the header: {}", seen[0].1);
        assert!(seen[1].0.contains("/sync?") && seen[1].0.contains("timeout=0") && !seen[1].0.contains("since="), "{}", seen[1].0);
        assert!(seen[2].0.starts_with("POST /_matrix/client/v3/join/") && seen[2].0.contains("mine"), "only the listed invite is accepted: {}", seen[2].0);
        assert!(seen[3].0.contains("since=s1"), "{}", seen[3].0);
        assert!(seen[4].0.starts_with("PUT /_matrix/client/v3/rooms/") && seen[4].0.contains("/send/m.room.message/fllm-"), "{}", seen[4].0);
        let sent: Value = serde_json::from_str(&seen[4].2).unwrap();
        assert_eq!((sent["msgtype"].as_str(), sent["body"].as_str()), (Some("m.notice"), Some("pong")), "bots answer with m.notice");
        assert!(!seen.iter().any(|(line, _, _)| line.contains("SECRET") || line.contains("secret")), "the token is never in a URL");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn matrix_refuses_bad_input_and_errors_never_contain_the_token() {
        let none = BTreeSet::new();
        assert!(MatrixHttp::connect("https://h", "", none.clone()).is_err());
        assert!(MatrixHttp::connect("https://h", "a b", none.clone()).is_err());
        assert!(MatrixHttp::connect("ftp://h", "tok", none.clone()).is_err());
        assert!(MatrixHttp::connect("not an address", "tok", none.clone()).is_err());
        let (hs, server) = stub(vec![(401, json!({"errcode": "M_UNKNOWN_TOKEN", "error": "Invalid access token"}))]);
        let err = MatrixHttp::connect(&hs, "syt_SECRET", none.clone()).err().unwrap();
        server.join().unwrap();
        assert!(err.contains("M_UNKNOWN_TOKEN") && !err.contains("SECRET"), "{err}");
        let err = MatrixHttp::connect("http://127.0.0.1:1", "syt_SECRET", none).err().unwrap();
        assert!(!err.contains("SECRET"), "{err}");
    }
}
