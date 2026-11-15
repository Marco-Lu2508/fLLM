//! Row 88 against a small server of our own that answers like `/api/studio/learn/…`: what the
//! window posts after a run, for a thumb and for the switches, and how it reads the answers.

use fllm_app::learn::*;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

type Log = Arc<Mutex<Vec<(String, String, Value)>>>;

async fn serve(status: Value) -> (String, Log) {
    let log: Log = Arc::default();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let seen = log.clone();
    tokio::spawn(async move {
        loop {
            let (mut sock, _) = listener.accept().await.unwrap();
            let seen = seen.clone();
            let status = status.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 65536];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                let text = String::from_utf8_lossy(&buf[..n]).to_string();
                let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
                let mut first = head.lines().next().unwrap_or("").split_whitespace();
                let (method, path) = (first.next().unwrap_or("").to_string(), first.next().unwrap_or("").to_string());
                let body: Value = serde_json::from_str(body).unwrap_or(Value::Null);
                seen.lock().unwrap().push((method.clone(), path.clone(), body));
                let (code, answer) = match (method.as_str(), path.split('?').next().unwrap_or("")) {
                    ("GET", "/api/studio/learn") => ("200 OK", status.to_string()),
                    ("POST", "/api/studio/learn/after-run") => ("200 OK", json!({ "queued": true }).to_string()),
                    ("POST", "/api/studio/learn/feedback") => ("200 OK", json!({ "noted": 2, "notes": ["Two failures in a row."] }).to_string()),
                    ("POST", "/api/studio/learn/settings") => ("200 OK", json!({ "ok": true }).to_string()),
                    _ => ("404 Not Found", json!({ "error": { "message": "no such route" } }).to_string()),
                };
                let _ = sock
                    .write_all(format!("HTTP/1.1 {code}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{answer}", answer.len()).as_bytes())
                    .await;
            });
        }
    });
    (format!("http://{addr}"), log)
}

#[tokio::test]
async fn the_status_is_read_with_the_folder() {
    let (base, log) = serve(json!({ "enabled": false, "ask_first": true, "pending": 2, "running": false, "last_run": { "t": 5.5, "summary": "wrote a skill", "lines": ["a"] } })).await;
    let st = fetch_status(base, "/home/marco/my project".into()).await.expect("a status");
    assert_eq!(st.enabled, Some(false));
    assert!(st.ask_first);
    assert_eq!(st.pending, Some(2.0));
    assert_eq!(st.last_run.unwrap().summary, "wrote a skill");
    assert_eq!(log.lock().unwrap()[0].1, "/api/studio/learn?workspace=%2Fhome%2Fmarco%2Fmy%20project");
    // no server, no status
    assert!(fetch_status("http://127.0.0.1:9".into(), String::new()).await.is_none());
}

#[tokio::test]
async fn the_notice_is_taken_once() {
    let (base, log) = serve(json!({ "notice_pending": true, "notice": "fLLM learns from your sessions." })).await;
    assert_eq!(fetch_notice(base, "/w".into()).await, "fLLM learns from your sessions.");
    let seen = log.lock().unwrap().clone();
    assert_eq!(seen[1].0, "POST");
    assert_eq!(seen[1].1, "/api/studio/learn/settings");
    assert_eq!(seen[1].2, json!({ "notice": true }), "the server is told it was shown");
    let (base, log) = serve(json!({ "notice_pending": false, "notice": "old" })).await;
    assert_eq!(fetch_notice(base, "/w".into()).await, "");
    assert_eq!(log.lock().unwrap().len(), 1, "nothing is posted for a notice that was shown");
}

#[tokio::test]
async fn the_switches_are_posted_as_the_page_posts_them() {
    let (base, log) = serve(json!({})).await;
    set_enabled(base.clone(), true, "/w".into()).await.unwrap();
    set_project(base.clone(), "/w".into(), Some(false)).await.unwrap();
    set_project(base.clone(), "/w".into(), None).await.unwrap();
    let seen = log.lock().unwrap().clone();
    assert_eq!(seen[0].2, json!({ "enabled": true, "workspace": "/w", "project": null }), "the global switch clears the folder's own setting");
    assert_eq!(seen[1].2, json!({ "workspace": "/w", "project": false }));
    assert_eq!(seen[2].2, json!({ "workspace": "/w", "project": null }));
}

#[tokio::test]
async fn a_run_and_a_thumb_go_to_the_server() {
    let (base, log) = serve(json!({})).await;
    let messages = vec![json!({ "role": "user", "content": "deploy" })];
    let body = after_run_body("s1", "/w", true, &["deploy".to_string()], messages);
    assert!(after_run(base.clone(), body).await);
    let last = Last { session: "s1".into(), workspace: "/w".into(), skills: vec!["deploy".into()] };
    let (noted, notes) = feedback(base.clone(), feedback_body("down", &last)).await.unwrap();
    assert_eq!((noted, notes), (2, vec!["Two failures in a row.".to_string()]));
    let seen = log.lock().unwrap().clone();
    assert_eq!(seen[0].2, json!({ "session": "s1", "workspace": "/w", "outcome": "failed", "skills": ["deploy"], "messages": [{ "role": "user", "content": "deploy" }] }));
    assert_eq!(seen[1].2, json!({ "value": "down", "skills": ["deploy"], "session": "s1", "workspace": "/w" }));
    assert_eq!(after_run_body("", "/w", false, &[], vec![])["session"], "studio", "a conversation without a name is called studio");
    assert_eq!(after_run_body("s", "/w", false, &[], vec![])["outcome"], "done");
    // a server that refuses: the run is not queued and a thumb says why
    assert!(!after_run("http://127.0.0.1:9".into(), json!({})).await);
    assert!(feedback("http://127.0.0.1:9".into(), json!({})).await.is_err());
}
