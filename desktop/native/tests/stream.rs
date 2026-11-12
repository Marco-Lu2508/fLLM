//! The streaming client against a small server of our own that speaks the same SSE as an fLLM server.

use fllm_app::api::{chat_stream, fetch_model, StreamEvent, WireMessage};
use futures::StreamExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Serves one request: `/v1/models` as JSON, `/v1/chat/completions` as the given SSE body (cut into chunks at awkward places).
async fn serve(sse: Vec<Vec<u8>>, status: u16) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let (mut sock, _) = listener.accept().await.unwrap();
            let sse = sse.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 16384];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                let head = String::from_utf8_lossy(&buf[..n]).to_string();
                if head.starts_with("GET /v1/models") {
                    let body = r#"{"object":"list","data":[{"id":"tiny","max_model_len":2048}]}"#;
                    let _ = sock.write_all(format!("HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}", body.len()).as_bytes()).await;
                } else if status != 200 {
                    let body = r#"{"error":{"message":"context length exceeded"}}"#;
                    let _ = sock.write_all(format!("HTTP/1.1 {status} Bad\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}", body.len()).as_bytes()).await;
                } else {
                    let _ = sock.write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n").await;
                    for part in sse {
                        let _ = sock.write_all(format!("{:x}\r\n", part.len()).as_bytes()).await;
                        let _ = sock.write_all(&part).await;
                        let _ = sock.write_all(b"\r\n").await;
                        let _ = sock.flush().await;
                        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                    }
                    let _ = sock.write_all(b"0\r\n\r\n").await;
                }
            });
        }
    });
    format!("http://{addr}")
}

fn msgs() -> Vec<WireMessage> {
    vec![WireMessage { role: "user".into(), content: "Hi".into() }]
}

#[tokio::test]
async fn the_model_comes_from_v1_models() {
    let base = serve(vec![], 200).await;
    let m = fetch_model(&base).await.unwrap();
    assert_eq!((m.id.as_str(), m.max_context), ("tiny", 2048));
}

#[tokio::test]
async fn no_server_is_an_error_with_the_address_in_it() {
    let err = fetch_model("http://127.0.0.1:9").await.unwrap_err();
    assert!(err.contains("127.0.0.1:9"), "{err}");
}

#[tokio::test]
async fn a_stream_cut_in_the_middle_of_a_line_and_of_a_character_gives_the_whole_text() {
    let text = "data: {\"choices\":[{\"delta\":{\"content\":\"Grüße\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\" Welt\"}}]}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3}}\n\ndata: [DONE]\n\n".as_bytes().to_vec();
    // cut inside "ü" (two bytes) and inside a JSON line
    let cut_a = text.iter().position(|&b| b == 0xc3).unwrap() + 1; // between the two bytes of the first ü
    let cut_b = cut_a + 40;
    let parts = vec![text[..cut_a].to_vec(), text[cut_a..cut_b].to_vec(), text[cut_b..].to_vec()];
    let base = serve(parts, 200).await;
    let events: Vec<StreamEvent> = chat_stream(base, "tiny".into(), msgs()).collect().await;
    let answer: String = events.iter().filter_map(|e| if let StreamEvent::Delta(s) = e { Some(s.as_str()) } else { None }).collect();
    assert_eq!(answer, "Grüße Welt");
    assert!(events.contains(&StreamEvent::Usage { prompt: 5, completion: 3 }));
    assert_eq!(events.last(), Some(&StreamEvent::Done));
}

#[tokio::test]
async fn a_server_error_becomes_one_error_event_with_its_message() {
    let base = serve(vec![], 400).await;
    let events: Vec<StreamEvent> = chat_stream(base, "tiny".into(), msgs()).collect().await;
    assert_eq!(events, vec![StreamEvent::Error("context length exceeded".into())]);
}

#[tokio::test]
async fn a_connection_that_ends_without_done_still_ends_the_stream() {
    let base = serve(vec![b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n".to_vec()], 200).await;
    let events: Vec<StreamEvent> = chat_stream(base, "tiny".into(), msgs()).collect().await;
    assert_eq!(events, vec![StreamEvent::Delta("Hi".into()), StreamEvent::Done]);
}

#[tokio::test]
async fn an_unreachable_server_is_an_error_event_and_the_stream_ends() {
    let events: Vec<StreamEvent> = chat_stream("http://127.0.0.1:9".into(), "tiny".into(), msgs()).collect().await;
    assert_eq!(events.len(), 1);
    assert!(matches!(&events[0], StreamEvent::Error(e) if e.contains("could not be reached")), "{events:?}");
}
