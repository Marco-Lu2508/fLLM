//! How long the window takes for a long chat: opening it (parsing the markdown of every message), building the view tree once, and laying
//! out and drawing it with the software renderer (the same work the window does per frame, without the GPU).
//! `cargo run --release --example view_bench -- 50 200 800` (the number of messages in the chat).
//! Then the cost of one token of a stream: an answer of 1500 pieces arrives in a chat of 0 and of 200 messages; each piece costs the update
//! (markdown) and the view with its layout.
use fllm_app::api::ModelInfo;
use fllm_app::api::StreamEvent;
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
            .map(|i| Record { role: if i % 2 == 0 { Role::User } else { Role::Assistant }, content: if i % 2 == 0 { format!("Question {i}: what is GGUF and which size should I pick?") } else { ANSWER.into() }, reasoning: String::new(), stats: None, passages: None })
            .collect();
        Store { chats: vec![ChatRecord::new(0, "long", messages)], next_id: 1, ..Default::default() }.save(&path).unwrap();
        let (mut app, _t) = App::with("http://127.0.0.1:9".into(), path);
        let _ = app.update(Message::Connected(Ok(ModelInfo::new("m", 8192))));
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
    stream_cost();
}

/// The pieces of an answer with code, lists and quotes, about four characters each.
fn pieces(n: usize) -> Vec<String> {
    let text = ANSWER.repeat(n / 20 + 2);
    let chars: Vec<char> = text.chars().collect();
    chars.chunks(4).take(n).map(|c| c.iter().collect()).collect()
}

fn stream_cost() {
    use iced::widget::text_editor::{Action, Edit};
    for prior in [0usize, 200] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chats.json");
        let messages: Vec<Record> = (0..prior)
            .map(|i| Record { role: if i % 2 == 0 { Role::User } else { Role::Assistant }, content: if i % 2 == 0 { format!("Question {i}") } else { ANSWER.into() }, reasoning: String::new(), stats: None, passages: None })
            .collect();
        let chats = if prior == 0 { vec![] } else { vec![ChatRecord::new(0, "long", messages)] };
        Store { chats, next_id: 1, ..Default::default() }.save(&path).unwrap();
        let (mut app, _t) = App::with("http://127.0.0.1:9".into(), path);
        let _ = app.update(Message::Connected(Ok(ModelInfo::new("m", 8192))));
        if prior > 0 {
            let _ = app.update(Message::OpenChat(0));
        }
        let _ = app.update(Message::Edit(Action::Edit(Edit::Paste(std::sync::Arc::new("Explain GGUF".to_string())))));
        let _ = app.update(Message::Send);
        let id = app.open_chat().unwrap();
        let mut update = Vec::new();
        let mut frame = Vec::new();
        for piece in pieces(1500) {
            let t = Instant::now();
            let _ = app.update(Message::Stream(id, StreamEvent::Delta(piece)));
            update.push(t.elapsed());
            let t = Instant::now();
            let ui = Simulator::with_size(iced::Settings::default(), iced::Size::new(1280.0, 900.0), app.view());
            frame.push(t.elapsed());
            drop(ui);
        }
        let stat = |v: &mut Vec<std::time::Duration>, from: usize, to: usize| {
            let mut w: Vec<_> = v[from..to].to_vec();
            w.sort();
            format!("median {:>8.1?}  p95 {:>8.1?}  max {:>8.1?}", w[w.len() / 2], w[w.len() * 95 / 100], w[w.len() - 1])
        };
        println!("stream in a chat of {prior:>3} messages, per piece, first 100 pieces: update {} | view+layout {}", stat(&mut update, 0, 100), stat(&mut frame, 0, 100));
        println!("{:>56}last 100 pieces:  update {} | view+layout {}", "", stat(&mut update, 1400, 1500), stat(&mut frame, 1400, 1500));
    }
}
