//! One turn of Code mode against a real server: opens a folder, gets the tools, asks the model something that needs a tool and prints the
//! stream events (the tool call parts) and, if the model called a tool, what the tool answered.
//! `FLLM_URL=http://127.0.0.1:8080 cargo run --example code_probe -- /path/to/folder "Read a.txt and tell me what is in it"`
use fllm_app::api::{chat_stream_with_tools, StreamEvent, WireMessage};
use fllm_app::code;
use futures::StreamExt;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let base = std::env::var("FLLM_URL").unwrap_or_else(|_| "http://127.0.0.1:8080".into());
    let mut args = std::env::args().skip(1);
    let folder = args.next().expect("a folder");
    let task = args.next().unwrap_or_else(|| "List the files in the folder.".into());
    let w = code::open_workspace(base.clone(), folder).await.expect("workspace");
    println!("workspace: {w:?}");
    let (tools, for_model) = code::fetch_tools(base.clone()).await.expect("tools");
    println!("{} tools; first: {:?}", tools.len(), tools.first());
    let model = fllm_app::api::fetch_model(&base).await.expect("model").id;
    println!("{} tools in the list for the model", for_model.as_array().map_or(0, |a| a.len()));
    let wire = vec![WireMessage::new("system", "You are a coding agent. Use the tools."), WireMessage::new("user", task)];
    let mut events = Box::pin(chat_stream_with_tools(base.clone(), model, wire, Default::default(), Some(for_model)));
    let mut calls: Vec<(String, String, String)> = Vec::new();
    while let Some(ev) = events.next().await {
        match ev {
            StreamEvent::ToolCall { index, id, name, arguments } => {
                while calls.len() <= index {
                    calls.push(Default::default());
                }
                if let Some(i) = id { calls[index].0 = i; }
                if let Some(n) = name { calls[index].1 = n; }
                calls[index].2.push_str(&arguments);
            }
            StreamEvent::Delta(s) => print!("{s}"),
            StreamEvent::Done => break,
            StreamEvent::Error(e) => { println!("\nerror: {e}"); break; }
            _ => {}
        }
    }
    println!("\ncalls: {calls:?}");
    for (_, name, args) in calls {
        let v: serde_json::Value = serde_json::from_str(&args).unwrap_or_default();
        println!("{name} {v} -> {:?}", code::run_tool(base.clone(), name.clone(), v.clone()).await);
    }
}
