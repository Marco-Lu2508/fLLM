//! Discord and Slack for `fllm-rs gateway`. Both use plain HTTPS polling with a bot token (no websocket), so the gateway needs
//! no inbound port and no extra library:
//!
//! * **Discord**: `GET /channels/{id}/messages?after=` per allowed channel, `POST /channels/{id}/messages`. Reading message text in
//!   a server channel needs the *Message Content* privileged intent in the bot's settings; direct messages work without it.
//! * **Slack**: `conversations.history` per allowed channel (`oldest` = the last seen timestamp) and `chat.postMessage`. The bot
//!   needs the history scope of the channel type (`channels:history`, `groups:history`, `im:history`) and `chat:write`. Slack
//!   limits `conversations.history` to one call a minute and 15 messages for apps distributed outside a workspace (since 2025);
//!   an app created for one workspace is not limited that way.
//!
//! Both keep one position per channel in the cursor string (`channel:last-id,channel:last-id`). The first poll of a channel only
//! records its newest message: what was written before the gateway started is not answered.
use crate::gateway::{Chat, Incoming};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

pub const DISCORD_TOKEN_ENV: &str = "FLLM_DISCORD_TOKEN";
pub const SLACK_TOKEN_ENV: &str = "FLLM_SLACK_TOKEN";
const DISCORD_LIMIT: usize = 1900;
const SLACK_LIMIT: usize = 3500;

/// `a:1,b:2` as a map (a channel id never contains `:` or `,`).
pub fn parse_cursor(cursor: &str) -> BTreeMap<String, String> {
    cursor.split(',').filter_map(|p| p.split_once(':')).filter(|(k, v)| !k.is_empty() && !v.is_empty()).map(|(k, v)| (k.to_owned(), v.to_owned())).collect()
}

pub fn render_cursor(map: &BTreeMap<String, String>) -> String {
    map.iter().map(|(k, v)| format!("{k}:{v}")).collect::<Vec<_>>().join(",")
}

fn client() -> Result<reqwest::blocking::Client, String> {
    reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::none())
        .user_agent("DiscordBot (https://github.com/Marco-Lu2508/fLLM, 0.1) fllm-gateway")
        .build()
        .map_err(|e| e.without_url().to_string())
}

fn is_snowflake(id: &str) -> bool {
    !id.is_empty() && id.len() <= 24 && id.bytes().all(|b| b.is_ascii_digit())
}

pub fn valid_discord_channel(id: &str) -> bool {
    is_snowflake(id)
}

pub fn valid_slack_channel(id: &str) -> bool {
    id.len() >= 3 && id.len() <= 24 && matches!(id.as_bytes()[0], b'C' | b'G' | b'D') && id.bytes().all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
}

fn snowflake_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    (a.len(), a).cmp(&(b.len(), b))
}

// ---------------------------------------------------------------------------------------------------------------
// Discord
// ---------------------------------------------------------------------------------------------------------------

/// The text messages of a channel answer, oldest first, and the newest id seen. Messages of bots (the gateway itself included) and
/// every message type except a plain message or a reply are left out; the id moves past all of them.
pub fn discord_batch(channel: &str, messages: &[Value], own_id: &str) -> (Vec<Incoming>, Option<String>) {
    let mut sorted: Vec<&Value> = messages.iter().filter(|m| m["id"].as_str().is_some()).collect();
    sorted.sort_by(|a, b| snowflake_cmp(a["id"].as_str().unwrap_or(""), b["id"].as_str().unwrap_or("")));
    let mut out = Vec::new();
    for m in &sorted {
        let kind = m["type"].as_u64().unwrap_or(0);
        let author = m["author"]["id"].as_str().unwrap_or("");
        let text = m["content"].as_str().unwrap_or("").trim();
        if (kind == 0 || kind == 19) && !text.is_empty() && author != own_id && m["author"]["bot"].as_bool() != Some(true) {
            out.push(Incoming { key: channel.to_owned(), text: text.to_owned(), sender: author.to_owned() });
        }
    }
    (out, sorted.last().and_then(|m| m["id"].as_str()).map(str::to_owned))
}

pub struct DiscordHttp {
    client: reqwest::blocking::Client,
    base: String,
    token: String,
    own_id: String,
    channels: BTreeSet<String>,
    pause_ms: u64,
}

impl DiscordHttp {
    pub fn connect(api: &str, token: &str, channels: BTreeSet<String>) -> Result<Self, String> {
        if token.trim().is_empty() || token.chars().any(char::is_whitespace) {
            return Err(format!("{DISCORD_TOKEN_ENV} is not set or is not a bot token (Developer Portal > Bot > Reset Token)"));
        }
        let mut me = Self { client: client()?, base: api.trim_end_matches('/').to_owned(), token: token.to_owned(), own_id: String::new(), channels, pause_ms: 3000 };
        let who = me.call(reqwest::Method::GET, "/users/@me", None)?;
        me.own_id = who["id"].as_str().ok_or("Discord did not say who the token belongs to")?.to_owned();
        Ok(me)
    }

    /// Time between two polls (Discord has no long poll); 0 for `--once`.
    pub fn with_poll_seconds(mut self, seconds: u64) -> Self {
        self.pause_ms = if seconds == 0 { 0 } else { 3000 };
        self
    }

    fn call(&self, method: reqwest::Method, path: &str, body: Option<&Value>) -> Result<Value, String> {
        let mut request = self.client.request(method, format!("{}{path}", self.base)).header("Authorization", format!("Bot {}", self.token));
        if let Some(body) = body {
            request = request.json(body);
        }
        let response = request.send().map_err(|e| format!("cannot reach Discord: {}", e.without_url()))?;
        let status = response.status();
        if status.as_u16() == 429 {
            let wait = response.headers().get("retry-after").and_then(|v| v.to_str().ok()).and_then(|v| v.parse::<f64>().ok()).unwrap_or(2.0).min(30.0);
            std::thread::sleep(Duration::from_secs_f64(wait));
            return Err(format!("Discord asked to slow down ({wait} s); the next poll continues"));
        }
        let text = response.text().map_err(|e| e.without_url().to_string())?;
        let parsed: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
        if !status.is_success() {
            return Err(format!("Discord {path}: HTTP {status}: {}", parsed["message"].as_str().unwrap_or("no description")));
        }
        Ok(parsed)
    }
}

impl Chat for DiscordHttp {
    fn updates(&mut self, cursor: &str) -> Result<(Vec<Incoming>, String), String> {
        if self.pause_ms > 0 && !cursor.is_empty() {
            std::thread::sleep(Duration::from_millis(self.pause_ms));
        }
        let mut positions = parse_cursor(cursor);
        let mut out = Vec::new();
        for channel in self.channels.clone() {
            match positions.get(&channel).cloned() {
                None => {
                    // first contact: remember the newest message, answer nothing
                    let newest = self.call(reqwest::Method::GET, &format!("/channels/{channel}/messages?limit=1"), None)?;
                    let id = newest.as_array().and_then(|a| a.first()).and_then(|m| m["id"].as_str()).unwrap_or("0");
                    positions.insert(channel, id.to_owned());
                }
                Some(after) => {
                    let list = self.call(reqwest::Method::GET, &format!("/channels/{channel}/messages?limit=50&after={after}"), None)?;
                    let (messages, newest) = discord_batch(&channel, list.as_array().map_or(&[][..], Vec::as_slice), &self.own_id);
                    out.extend(messages);
                    if let Some(id) = newest {
                        positions.insert(channel, id);
                    }
                }
            }
        }
        Ok((out, render_cursor(&positions)))
    }

    fn send(&mut self, key: &str, text: &str) -> Result<(), String> {
        if !is_snowflake(key) {
            return Err(format!("'{key}' is not a Discord channel id"));
        }
        // no mention in an answer may ping anybody
        self.call(reqwest::Method::POST, &format!("/channels/{key}/messages"), Some(&json!({"content": text, "allowed_mentions": {"parse": []}}))).map(|_| ())
    }

    fn session_name(&self, key: &str) -> String {
        format!("dc-{}", key.chars().filter(char::is_ascii_digit).take(24).collect::<String>())
    }

    fn max_len(&self) -> usize {
        DISCORD_LIMIT
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Slack
// ---------------------------------------------------------------------------------------------------------------

/// Plain messages of a `conversations.history` answer, oldest first, and the newest timestamp seen. Bot messages, the gateway's own
/// and every message with a subtype (joins, edits, file shares ...) are left out.
pub fn slack_batch(channel: &str, messages: &[Value], own_user: &str) -> (Vec<Incoming>, Option<String>) {
    let mut sorted: Vec<&Value> = messages.iter().filter(|m| m["ts"].as_str().is_some()).collect();
    sorted.sort_by(|a, b| a["ts"].as_str().unwrap_or("").parse::<f64>().unwrap_or(0.0).total_cmp(&b["ts"].as_str().unwrap_or("").parse::<f64>().unwrap_or(0.0)));
    let mut out = Vec::new();
    for m in &sorted {
        let user = m["user"].as_str().unwrap_or("");
        let text = m["text"].as_str().unwrap_or("").trim();
        if m["subtype"].is_null() && m["bot_id"].is_null() && !user.is_empty() && user != own_user && !text.is_empty() {
            out.push(Incoming { key: channel.to_owned(), text: text.to_owned(), sender: user.to_owned() });
        }
    }
    (out, sorted.last().and_then(|m| m["ts"].as_str()).map(str::to_owned))
}

pub struct SlackHttp {
    client: reqwest::blocking::Client,
    base: String,
    token: String,
    own_user: String,
    channels: BTreeSet<String>,
    pause_ms: u64,
}

impl SlackHttp {
    pub fn connect(api: &str, token: &str, channels: BTreeSet<String>) -> Result<Self, String> {
        if token.trim().is_empty() || token.chars().any(char::is_whitespace) {
            return Err(format!("{SLACK_TOKEN_ENV} is not set or is not a bot token (xoxb-..., OAuth & Permissions)"));
        }
        let mut me = Self { client: client()?, base: api.trim_end_matches('/').to_owned(), token: token.to_owned(), own_user: String::new(), channels, pause_ms: 5000 };
        let who = me.call("auth.test", &json!({}))?;
        me.own_user = who["user_id"].as_str().ok_or("Slack did not say who the token belongs to")?.to_owned();
        Ok(me)
    }

    pub fn with_poll_seconds(mut self, seconds: u64) -> Self {
        self.pause_ms = if seconds == 0 { 0 } else { 5000 };
        self
    }

    fn call(&self, method: &str, body: &Value) -> Result<Value, String> {
        let response = self
            .client
            .post(format!("{}/{method}", self.base))
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Content-Type", "application/json; charset=utf-8")
            .json(body)
            .send()
            .map_err(|e| format!("cannot reach Slack: {}", e.without_url()))?;
        let status = response.status();
        if status.as_u16() == 429 {
            let wait = response.headers().get("retry-after").and_then(|v| v.to_str().ok()).and_then(|v| v.parse::<f64>().ok()).unwrap_or(5.0).min(60.0);
            std::thread::sleep(Duration::from_secs_f64(wait));
            return Err(format!("Slack asked to slow down ({wait} s); the next poll continues"));
        }
        let parsed: Value = response.json().map_err(|e| e.without_url().to_string())?;
        if !status.is_success() || parsed["ok"] != json!(true) {
            return Err(format!("Slack {method}: {}", parsed["error"].as_str().unwrap_or("no description")));
        }
        Ok(parsed)
    }
}

impl Chat for SlackHttp {
    fn updates(&mut self, cursor: &str) -> Result<(Vec<Incoming>, String), String> {
        if self.pause_ms > 0 && !cursor.is_empty() {
            std::thread::sleep(Duration::from_millis(self.pause_ms));
        }
        let mut positions = parse_cursor(cursor);
        let mut out = Vec::new();
        for channel in self.channels.clone() {
            match positions.get(&channel).cloned() {
                None => {
                    let newest = self.call("conversations.history", &json!({"channel": channel, "limit": 1}))?;
                    let ts = newest["messages"].as_array().and_then(|a| a.first()).and_then(|m| m["ts"].as_str()).unwrap_or("0");
                    positions.insert(channel, ts.to_owned());
                }
                Some(oldest) => {
                    let list = self.call("conversations.history", &json!({"channel": channel, "oldest": oldest, "inclusive": false, "limit": 50}))?;
                    let (messages, newest) = slack_batch(&channel, list["messages"].as_array().map_or(&[][..], Vec::as_slice), &self.own_user);
                    out.extend(messages);
                    if let Some(ts) = newest {
                        positions.insert(channel, ts);
                    }
                }
            }
        }
        Ok((out, render_cursor(&positions)))
    }

    fn send(&mut self, key: &str, text: &str) -> Result<(), String> {
        if !valid_slack_channel(key) {
            return Err(format!("'{key}' is not a Slack channel id"));
        }
        self.call("chat.postMessage", &json!({"channel": key, "text": text, "mrkdwn": false, "unfurl_links": false, "unfurl_media": false, "link_names": false})).map(|_| ())
    }

    fn session_name(&self, key: &str) -> String {
        format!("sl-{}", key.to_ascii_lowercase())
    }

    fn max_len(&self) -> usize {
        SLACK_LIMIT
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    #[test]
    fn cursors_round_trip_and_ignore_garbage() {
        let map = parse_cursor("111:900,222:5,bad,:x,333:");
        assert_eq!(map.len(), 2);
        assert_eq!(render_cursor(&map), "111:900,222:5");
        assert!(parse_cursor("").is_empty());
    }

    #[test]
    fn channel_ids_are_checked() {
        assert!(valid_discord_channel("1234567890123456789") && !valid_discord_channel("12a") && !valid_discord_channel(""));
        assert!(valid_slack_channel("C024BE91L") && valid_slack_channel("D01ABC") && !valid_slack_channel("#general") && !valid_slack_channel("c024"));
    }

    #[test]
    fn discord_messages_come_oldest_first_without_bots_and_system_messages() {
        let list = vec![
            json!({"id": "1002", "type": 0, "content": "second", "author": {"id": "u1"}}),
            json!({"id": "1001", "type": 0, "content": "first", "author": {"id": "u1"}}),
            json!({"id": "1003", "type": 0, "content": "mine", "author": {"id": "me", "bot": true}}),
            json!({"id": "1004", "type": 7, "content": "joined", "author": {"id": "u2"}}),
            json!({"id": "1005", "type": 19, "content": "a reply", "author": {"id": "u2"}}),
            json!({"id": "1006", "type": 0, "content": "", "author": {"id": "u2"}}),
            json!({"id": "99", "type": 0, "content": "old", "author": {"id": "u3"}}),
        ];
        let (msgs, newest) = discord_batch("chan", &list, "me");
        assert_eq!(msgs.iter().map(|m| m.text.as_str()).collect::<Vec<_>>(), vec!["old", "first", "second", "a reply"]);
        assert_eq!(msgs[1].sender, "u1");
        assert_eq!(newest.as_deref(), Some("1006"), "ids are compared as numbers, not text");
    }

    #[test]
    fn slack_messages_skip_subtypes_bots_and_the_gateway() {
        let list = vec![
            json!({"ts": "1700000002.000100", "user": "U1", "text": "later"}),
            json!({"ts": "1700000001.000100", "user": "U1", "text": "earlier"}),
            json!({"ts": "1700000003.000100", "user": "UBOT", "text": "mine"}),
            json!({"ts": "1700000004.000100", "user": "U2", "subtype": "channel_join", "text": "joined"}),
            json!({"ts": "1700000005.000100", "bot_id": "B1", "text": "another bot"}),
        ];
        let (msgs, newest) = slack_batch("C01ABC", &list, "UBOT");
        assert_eq!(msgs.iter().map(|m| m.text.as_str()).collect::<Vec<_>>(), vec!["earlier", "later"]);
        assert_eq!(newest.as_deref(), Some("1700000005.000100"));
    }

    fn read_request(stream: &mut std::net::TcpStream) -> String {
        let mut buf = Vec::new();
        let mut chunk = [0u8; 2048];
        loop {
            let n = stream.read(&mut chunk).unwrap();
            buf.extend_from_slice(&chunk[..n]);
            let text = String::from_utf8_lossy(&buf).to_string();
            if let Some((head, body)) = text.split_once("\r\n\r\n") {
                let len = head.lines().find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap_or(0))).unwrap_or(0);
                if body.len() >= len {
                    return text;
                }
            }
            if n == 0 {
                return text;
            }
        }
    }

    fn stub(answers: Vec<Value>) -> (String, std::thread::JoinHandle<Vec<(String, String)>>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let handle = std::thread::spawn(move || {
            let mut seen = Vec::new();
            for payload in answers {
                let (mut stream, _) = listener.accept().unwrap();
                let request = read_request(&mut stream);
                let (head, body) = request.split_once("\r\n\r\n").unwrap_or((&request, ""));
                seen.push((head.to_ascii_lowercase(), body.to_owned()));
                let payload = payload.to_string();
                stream.write_all(format!("HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{payload}", payload.len()).as_bytes()).unwrap();
            }
            seen
        });
        (url, handle)
    }

    #[test]
    fn discord_records_the_position_first_then_answers_new_messages_only() {
        let (api, server) = stub(vec![
            json!({"id": "me"}),                                                                    // /users/@me
            json!([{"id": "500", "type": 0, "content": "old message", "author": {"id": "u1"}}]),     // first contact: limit=1
            json!([{"id": "501", "type": 0, "content": "run it", "author": {"id": "u1"}}]),          // second poll
            json!({"id": "502"}),                                                                    // send
        ]);
        let mut chat = DiscordHttp::connect(&api, "TOKEN123", BTreeSet::from(["777".to_owned()])).unwrap().with_poll_seconds(0);
        let (first, cursor) = chat.updates("").unwrap();
        assert!(first.is_empty());
        assert_eq!(cursor, "777:500");
        let (second, cursor) = chat.updates(&cursor).unwrap();
        assert_eq!(second, vec![Incoming { key: "777".into(), text: "run it".into(), sender: "u1".into() }]);
        assert_eq!(cursor, "777:501");
        chat.send("777", "done @everyone").unwrap();
        let seen = server.join().unwrap();
        assert!(seen[0].0.contains("get /users/@me") && seen[0].0.contains("authorization: bot token123"));
        assert!(seen[1].0.contains("limit=1") && seen[2].0.contains("after=500"));
        let sent: Value = serde_json::from_str(&seen[3].1).unwrap();
        assert_eq!(sent["allowed_mentions"]["parse"], json!([]), "no answer can ping anybody");
        assert!(!seen[3].1.contains("TOKEN123"));
    }

    #[test]
    fn slack_polls_with_oldest_and_posts_plain_text() {
        let (api, server) = stub(vec![
            json!({"ok": true, "user_id": "UBOT"}),
            json!({"ok": true, "messages": [{"ts": "10.5", "user": "U1", "text": "before"}]}),
            json!({"ok": true, "messages": [{"ts": "11.5", "user": "U1", "text": "please build"}, {"ts": "12.5", "user": "UBOT", "text": "ok"}]}),
            json!({"ok": true}),
        ]);
        let mut chat = SlackHttp::connect(&api, "xoxb-test", BTreeSet::from(["C01ABC".to_owned()])).unwrap().with_poll_seconds(0);
        let (none, cursor) = chat.updates("").unwrap();
        assert!(none.is_empty() && cursor == "C01ABC:10.5");
        let (msgs, cursor) = chat.updates(&cursor).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!((msgs[0].text.as_str(), msgs[0].sender.as_str()), ("please build", "U1"));
        assert_eq!(cursor, "C01ABC:12.5", "the gateway's own message moves the position too");
        chat.send("C01ABC", "built").unwrap();
        let seen = server.join().unwrap();
        assert!(seen[2].1.contains("\"oldest\":\"10.5\""));
        assert!(seen[3].1.contains("\"mrkdwn\":false") && seen[0].0.contains("authorization: bearer xoxb-test"));
        assert!(chat.send("general", "x").is_err());
    }

    #[test]
    fn an_error_answer_names_the_reason_without_the_token() {
        let (api, server) = stub(vec![json!({"ok": true, "user_id": "U"}), json!({"ok": false, "error": "not_in_channel"})]);
        let mut chat = SlackHttp::connect(&api, "xoxb-secret", BTreeSet::from(["C01ABC".to_owned()])).unwrap().with_poll_seconds(0);
        let e = chat.updates("").unwrap_err();
        assert!(e.contains("not_in_channel") && !e.contains("secret"));
        server.join().unwrap();
        assert!(SlackHttp::connect("http://127.0.0.1:1", "has space", BTreeSet::new()).is_err());
        assert!(DiscordHttp::connect("http://127.0.0.1:1", "", BTreeSet::new()).is_err());
    }
}
