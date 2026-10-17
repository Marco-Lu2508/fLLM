//! `cargo run --release -p fllm-checkpoint --example tokenizer_json -- <checkpoint dir>` prints the `tokenizer.json`
//! the converter derives from `tekken.json`, a tiktoken rank file or a SentencePiece model.
fn main() {
    let dir = std::env::args().nth(1).expect("usage: tokenizer_json <checkpoint dir>");
    match fllm_checkpoint::tokenizer_json_from_dir(std::path::Path::new(&dir)) {
        Ok(json) => println!("{json}"),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
