//! Talks to a real fLLM server with the window's own client: lists the model, streams a short answer and stops after N events.
//!   FLLM_URL=http://127.0.0.1:8080 cargo run --example probe -- "Say hi in three words" 12
use fllm_app::api::{chat_stream, fetch_model, StreamEvent, WireMessage};
use futures::StreamExt;

#[tokio::main]
async fn main() {
    let base = std::env::var("FLLM_URL").unwrap_or_else(|_| "http://127.0.0.1:8080".into());
    let mut args = std::env::args().skip(1);
    let prompt = args.next().unwrap_or_else(|| "Say hi in three words.".into());
    let limit: usize = args.next().and_then(|n| n.parse().ok()).unwrap_or(12);
    let model = fetch_model(&base).await.expect("a server is running");
    println!("model: {} (context {})", model.id, model.max_context);
    let mut events = Box::pin(chat_stream(base, model.id, vec![WireMessage { role: "user".into(), content: prompt }], Default::default()));
    let mut seen = 0;
    while let Some(ev) = events.next().await {
        println!("{ev:?}");
        seen += 1;
        if seen >= limit || matches!(ev, StreamEvent::Done | StreamEvent::Error(_)) {
            break;
        }
    }
    // dropping the stream closes the connection: the server stops generating
}
