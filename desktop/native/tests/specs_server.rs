//! Row 80 against a small server of our own that answers like `/api/studio/agent/specs`: the
//! list is read, an action is posted as the page posts it, a refusal comes back with the words
//! of the page, and `/spec run` / `/spec NAME …` find, reload and ask in the page's order.

use fllm_app::code::{spec_new, spec_run};
use fllm_app::specs::{fetch_specs, spec_act, tick_body};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// What the server was asked: `(method, path, body)` in order.
type Log = Arc<Mutex<Vec<(String, String, Value)>>>;

async fn respond(sock: &mut tokio::net::TcpStream, status: &str, body: &str) {
    let _ = sock
        .write_all(format!("HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}", body.len()).as_bytes())
        .await;
}

fn list() -> Value {
    json!({ "specs": [
        { "name": "export", "requirements": true, "design": true, "tasks_file": true, "total": 2, "done": 1,
          "tasks": [{ "index": 1, "text": "a", "done": true, "depth": 0 }, { "index": 2, "text": "b", "done": false, "depth": 1 }] },
        { "name": "search", "requirements": true, "design": false, "tasks_file": false, "total": 1, "done": 1,
          "tasks": [{ "index": 1, "text": "c", "done": true, "depth": 0 }] }
    ] })
}

/// `fresh` is what the list route answers; the actions answer by their `action`.
async fn serve(fresh: Value, list_status: &'static str) -> (String, Log) {
    let log: Log = Arc::default();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let seen = log.clone();
    tokio::spawn(async move {
        loop {
            let (mut sock, _) = listener.accept().await.unwrap();
            let seen = seen.clone();
            let fresh = fresh.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 16384];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                let text = String::from_utf8_lossy(&buf[..n]).to_string();
                let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
                let mut first = head.lines().next().unwrap_or("").split_whitespace();
                let (method, path) = (first.next().unwrap_or("").to_string(), first.next().unwrap_or("").to_string());
                let body: Value = serde_json::from_str(body).unwrap_or(Value::Null);
                seen.lock().unwrap().push((method.clone(), path.clone(), body.clone()));
                match (method.as_str(), path.as_str(), body["action"].as_str().unwrap_or("")) {
                    ("GET", "/api/studio/agent/specs", _) => respond(&mut sock, list_status, &fresh.to_string()).await,
                    ("POST", "/api/studio/agent/specs", "tick") => respond(&mut sock, "200 OK", &fresh.to_string()).await,
                    ("POST", "/api/studio/agent/specs", "task_prompt") if body["name"] == "refused" => {
                        respond(&mut sock, "400 Bad Request", r#"{"error":{"message":"the task has no text"}}"#).await
                    }
                    ("POST", "/api/studio/agent/specs", "task_prompt") if body["name"] == "bare" => respond(&mut sock, "500 Internal Server Error", "").await,
                    ("POST", "/api/studio/agent/specs", "task_prompt") => {
                        respond(&mut sock, "200 OK", &json!({ "prompt": format!("Do task {} of {}.", body["index"], body["name"].as_str().unwrap_or("")) }).to_string()).await
                    }
                    ("POST", "/api/studio/agent/specs", "new_prompt") => {
                        respond(&mut sock, "200 OK", &json!({ "prompt": format!("Write requirements for {}.", body["name"].as_str().unwrap_or("")), "name": body["name"] }).to_string()).await
                    }
                    _ => respond(&mut sock, "404 Not Found", "{}").await,
                }
            });
        }
    });
    (format!("http://{addr}"), log)
}

fn paths(log: &Log) -> Vec<(String, String)> {
    log.lock().unwrap().iter().map(|(m, p, _)| (m.clone(), p.clone())).collect()
}

/// The list is read with every field the side draws; a server that cannot give it gives an
/// error, which the window turns into an empty list (`code.specs = []` of `loadSpecs`).
#[tokio::test]
async fn the_list_is_read_with_everything_the_side_draws() {
    let (base, log) = serve(list(), "200 OK").await;
    let specs = fetch_specs(base).await.unwrap();
    assert_eq!(paths(&log), vec![("GET".to_string(), "/api/studio/agent/specs".to_string())]);
    assert_eq!(specs.len(), 2);
    assert_eq!((specs[0].name.as_str(), specs[0].done, specs[0].total), ("export", 1, 2));
    assert!(specs[0].requirements && specs[0].design && specs[0].tasks_file);
    assert!(specs[1].requirements && !specs[1].design && !specs[1].tasks_file, "the chips follow the files that exist");
    assert_eq!(specs[0].tasks[1].depth, 1, "the depth of a nested task");
    assert!(!specs[0].tasks[1].done);

    let (base, _log) = serve(list(), "404 Not Found").await;
    assert!(fetch_specs(base).await.is_err(), "a route that is not there answers with an error");
}

/// An action is posted with the body the page builds and answers with what the server sent;
/// a refusal carries its message, one without a message the page's own words, and no server
/// at all says so.
#[tokio::test]
async fn an_action_is_posted_like_the_page_posts_it() {
    let (base, log) = serve(list(), "200 OK").await;
    let answer = spec_act(base.clone(), tick_body("tick", "export", 2)).await.unwrap();
    assert_eq!(answer["specs"].as_array().map(Vec::len), Some(2));
    let seen = log.lock().unwrap().clone();
    assert_eq!(seen[0].0, "POST");
    assert_eq!(seen[0].2, json!({ "action": "tick", "name": "export", "index": 2 }), "action, name and the number of the task");

    let refused = spec_act(base.clone(), json!({ "action": "task_prompt", "name": "refused", "index": 1 })).await.unwrap_err();
    assert_eq!(refused, "the task has no text", "the message of the body");
    let bare = spec_act(base, json!({ "action": "task_prompt", "name": "bare", "index": 1 })).await.unwrap_err();
    assert_eq!(bare, "The spec action failed.", "the page's own words when the body has none");
    let gone = spec_act("http://127.0.0.1:9".into(), tick_body("untick", "export", 1)).await.unwrap_err();
    assert!(gone.starts_with("no server at"), "a request that did not arrive says so: {gone}");
}

/// `/spec run`: the spec the window holds runs without a read; a missing one is read for first
/// and only then declared missing; the task asked for goes through whatever it is, and a spec
/// with nothing left says so before it asks the server for anything.
#[tokio::test]
async fn a_run_finds_reloads_and_asks_in_the_pages_order() {
    let current = fllm_app::specs::parse_specs(&list());

    // the spec is in the list the window holds: one request, the prompt of the first open task
    let (base, log) = serve(list(), "200 OK").await;
    let (prompt, display) = spec_run(base, current.clone(), "export".into(), None).await.unwrap();
    assert_eq!((prompt.as_str(), display.as_str()), ("Do task 2 of export.", "/spec run export 2"));
    assert_eq!(paths(&log), vec![("POST".to_string(), "/api/studio/agent/specs".to_string())], "no reload for a spec that is there");

    // a number that was asked for wins, and goes out as it was written
    let (base, log) = serve(list(), "200 OK").await;
    let (_, display) = spec_run(base, current.clone(), "export".into(), Some(7.0)).await.unwrap();
    assert_eq!(display, "/spec run export 7");
    assert_eq!(log.lock().unwrap()[0].2["index"], json!(7), "a whole number goes out as an integer");

    // the window has nothing: the list is read, the spec is found, the run goes on
    let (base, log) = serve(list(), "200 OK").await;
    let (prompt, _) = spec_run(base, Vec::new(), "export".into(), None).await.unwrap();
    assert_eq!(prompt, "Do task 2 of export.");
    assert_eq!(paths(&log).iter().map(|(m, _)| m.as_str()).collect::<Vec<_>>(), ["GET", "POST"], "reload first, then the task");

    // a name that stays missing after the reload
    let (base, log) = serve(list(), "200 OK").await;
    let missing = spec_run(base, Vec::new(), "nope".into(), None).await.unwrap_err();
    assert_eq!(missing, "There is no spec \"nope\". /spec lists them.");
    assert_eq!(paths(&log).len(), 1, "one read and nothing else");

    // a reload that fails is a list of nothing (`loadSpecs` catches into `[]`)
    let (base, _log) = serve(list(), "500 Internal Server Error").await;
    let broken = spec_run(base, Vec::new(), "export".into(), None).await.unwrap_err();
    assert_eq!(broken, "There is no spec \"export\". /spec lists them.");

    // every task done: said without asking the server for a prompt
    let (base, log) = serve(list(), "200 OK").await;
    let done = spec_run(base, current.clone(), "search".into(), None).await.unwrap_err();
    assert_eq!(done, "Every task of this spec is done.");
    assert!(log.lock().unwrap().is_empty(), "nothing was asked");

    // the server refuses the prompt: its message is what the list says
    let (base, _log) = serve(list(), "200 OK").await;
    let mut held = current;
    held.push(fllm_app::specs::Spec { name: "refused".into(), tasks: vec![fllm_app::specs::SpecTask { index: 1, ..Default::default() }], total: 1, ..Default::default() });
    let refused = spec_run(base, held, "refused".into(), None).await.unwrap_err();
    assert_eq!(refused, "the task has no text");
}

/// `/spec NAME what you want` asks for the prompt of a new spec and runs it with the words as
/// they were typed.
#[tokio::test]
async fn a_new_spec_asks_for_its_prompt_and_runs_with_the_words_typed() {
    let (base, log) = serve(list(), "200 OK").await;
    let (prompt, display) = spec_new(base, "csv".into(), "Export it".into(), "/spec csv   Export it".into()).await.unwrap();
    assert_eq!(prompt, "Write requirements for csv.");
    assert_eq!(display, "/spec csv   Export it", "the display keeps the spaces of the argument");
    assert_eq!(log.lock().unwrap()[0].2, json!({ "action": "new_prompt", "name": "csv", "description": "Export it" }));
}
