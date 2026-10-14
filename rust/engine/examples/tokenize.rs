//! Print the token ids fLLM produces for each JSON-encoded string line of stdin:
//! `tokenize MODEL.gguf < lines.jsonl`.
use fllm_engine::tokenizer::TokenizerBundle;
use fllm_gguf_probe::MappedGguf;
use std::io::BufRead;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("GGUF path is required")?;
    let tokenizer = TokenizerBundle::from_gguf(&MappedGguf::open(path)?)?;
    for line in std::io::stdin().lock().lines() {
        let text: String = serde_json::from_str(&line?)?;
        println!(
            "{}",
            serde_json::to_string(&tokenizer.encode_without_bos(&text)?)?
        );
    }
    Ok(())
}
