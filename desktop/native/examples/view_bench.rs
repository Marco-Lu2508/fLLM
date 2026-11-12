//! How long the window takes for a long chat: opening it (parsing the markdown of every message), building the view tree once, and laying
//! out and drawing it with the software renderer (the same work the window does per frame, without the GPU).
//! `cargo run --release --example view_bench -- 50 200 800`
use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message};
use fllm_app::store::{ChatRecord, Record, Role, Store};
use iced_test::Simulator;
use std::time::Instant;

const ANSWER: &str = "**GGUF** is a single-file format for model weights: a header with metadata, then the tensors, ready to be memory-mapped.\n\n```python\nwith open(\"model.gguf\", \"rb\") as f:\n    magic = f.read(4)\nprint(magic)  # b'GGUF'\n```\n\n1. Pick a size that fits your memory.\n2. Prefer `Q4_K_M` first.\n3. Move up only if answers get worse.\n\n> Rule of thumb: the file size plus the KV cache has to fit in memory.";

fn main() {
    let sizes: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    for n in if sizes.is_empty() { vec![50, 200, 800] } else { sizes } {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chats.json");
        let messages: Vec<Record> = (0..n)
            .map(|i| Record { role: if i % 2 == 0 { Role::User } else { Role::Assistant }, content: if i % 2 == 0 { format!("Question {i}: what is GGUF and which size should I pick?") } else { ANSWER.into() }, reasoning: String::new(), stats: None })
            .collect();
        Store { chats: vec![ChatRecord { id: 0, title: "long".into(), messages }], next_id: 1, ..Default::default() }.save(&path).unwrap();
        let (mut app, _t) = App::with("http://127.0.0.1:9".into(), path);
        let _ = app.update(Message::Connected(Ok(ModelInfo { id: "m".into(), max_context: 8192 })));
        let t = Instant::now();
        let _ = app.update(Message::OpenChat(0));
        let open = t.elapsed();
        let t = Instant::now();
        let view = app.view();
        let build = t.elapsed();
        let t = Instant::now();
        let mut ui = Simulator::with_size(iced::Settings::default(), iced::Size::new(1280.0, 900.0), view);
        let layout = t.elapsed();
        let t = Instant::now();
        let _ = ui.snapshot(&app.theme()).unwrap();
        let draw = t.elapsed();
        println!("{n:>5} messages: open {open:>9.1?} | build view {build:>9.1?} | layout {layout:>9.1?} | draw (software) {draw:>9.1?}");
    }
}
