//! Row 77 against a small server of our own that answers like `/api/studio/agents`: the list is
//! read with the folder in the query, `create_agent` posts what the page posts and answers in the
//! page's words, and a refusal of the route comes back as its message (`Error 400` without one).

use fllm_app::code::{create_agent_file, fetch_agents};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

type Log = Arc<Mutex<Vec<(String, String, Value)>>>;

async fn serve() -> (String, Log) {
    let log: Log = Arc::default();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let seen = log.clone();
    tokio::spawn(async move {
        loop {
            let (mut sock, _) = listener.accept().await.unwrap();
            let seen = seen.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 16384];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                let text = String::from_utf8_lossy(&buf[..n]).to_string();
                let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
                let mut first = head.lines().next().unwrap_or("").split_whitespace();
                let (method, path) = (first.next().unwrap_or("").to_string(), first.next().unwrap_or("").to_string());
                let body: Value = serde_json::from_str(body).unwrap_or(Value::Null);
                seen.lock().unwrap().push((method.clone(), path.clone(), body.clone()));
                let (status, answer) = match (method.as_str(), body["name"].as_str().unwrap_or("")) {
                    ("GET", _) => ("200 OK", json!({ "agents": [{ "name": "code-reviewer", "description": "d", "tools": ["read_file"], "model": "", "body": "b", "max_steps": 4, "scope": "workspace" }] }).to_string()),
                    ("POST", "taken") => ("400 Bad Request", json!({ "error": { "message": "an agent with this name exists" } }).to_string()),
                    ("POST", "bare") => ("500 Internal Server Error", String::new()),
                    _ => ("200 OK", json!({ "ok": true }).to_string()),
                };
                let _ = sock
                    .write_all(format!("HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{answer}", answer.len()).as_bytes())
                    .await;
            });
        }
    });
    (format!("http://{addr}"), log)
}

#[tokio::test]
async fn the_list_is_asked_for_with_the_folder() {
    let (base, log) = serve().await;
    let agents = fetch_agents(base.clone(), "/home/marco/my project".into()).await.unwrap();
    assert_eq!(agents.len(), 1);
    assert_eq!((agents[0].name.as_str(), agents[0].tools.clone(), agents[0].max_steps), ("code-reviewer", vec!["read_file".to_string()], Some(4.0)));
    fetch_agents(base, String::new()).await.unwrap();
    let seen = log.lock().unwrap().clone();
    assert_eq!(seen[0].1, "/api/studio/agents?workspace=%2Fhome%2Fmarco%2Fmy%20project", "the folder as encodeURIComponent writes it");
    assert_eq!(seen[1].1, "/api/studio/agents", "no folder, no query");
}

#[tokio::test]
async fn an_agent_is_posted_as_the_page_posts_it() {
    let (base, log) = serve().await;
    let said = create_agent_file(
        base.clone(),
        "/work".into(),
        json!({ "name": "  fixer ", "description": "Fixes.", "instructions": "You fix.", "tools": ["read_file", "edit_file"], "model": "big" }),
    )
    .await
    .unwrap();
    assert_eq!(said, "The agent \"fixer\" was created (for this folder). Start it with the task tool: agent \"fixer\".");
    let said = create_agent_file(base.clone(), "/work".into(), json!({ "name": "all", "scope": "user", "tools": "nope", "description": 5 })).await.unwrap();
    assert_eq!(said, "The agent \"all\" was created (for every folder). Start it with the task tool: agent \"all\".");
    let seen = log.lock().unwrap().clone();
    assert_eq!(seen[0].0, "POST");
    assert_eq!(
        seen[0].2,
        json!({ "name": "fixer", "description": "Fixes.", "tools": ["read_file", "edit_file"], "model": "big", "body": "You fix.", "scope": "workspace", "workspace": "/work" })
    );
    assert_eq!(seen[1].2["tools"], json!([]), "a tools list that is no list is an empty one");
    assert_eq!(seen[1].2["scope"], "user");
    assert_eq!(seen[1].2["description"], 5, "what the model wrote goes on as it is");
    assert_eq!(seen[1].2["model"], "", "and what it left out is an empty text");
}

#[tokio::test]
async fn a_refusal_comes_back_in_the_words_of_the_route() {
    let (base, _log) = serve().await;
    assert_eq!(create_agent_file(base.clone(), "/w".into(), json!({ "name": "taken" })).await.unwrap_err(), "an agent with this name exists");
    assert_eq!(create_agent_file(base, "/w".into(), json!({ "name": "bare" })).await.unwrap_err(), "Error 500");
}
