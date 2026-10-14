//! Transport for independent checkpoint conversion/logit reference checks.
use fllm_engine::{rust_decoder::RustDecoder, tokenizer::TokenizerBundle, GgufModel};
use serde_json::{json, Value};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let input: Value =
        serde_json::from_slice(&fs::read(args.get(1).ok_or("model.gguf reference.json")?)?)?;
    let ids = input
        .get("ids")
        .unwrap_or(&input["prompt_ids"])
        .as_array()
        .ok_or("ids array required")?
        .iter()
        .map(|v| v.as_i64().ok_or("invalid token ID"))
        .collect::<Result<Vec<_>, _>>()?;
    if ids.is_empty() {
        return Err("empty reference token sequence".into());
    }
    let mut model = GgufModel::load(args.first().ok_or("model required")?, 128, 1, 2)?;
    model.set_kv_kind(1);
    let decoder = RustDecoder::load(&model, "cpu")?;
    let table = (0..model.table_stride())
        .map(|n| n as i64)
        .collect::<Vec<_>>();
    let mut key = vec![0; model.key_buffer_len()?];
    let mut value = vec![0; model.value_buffer_len()?];
    let rows = (0..ids.len()).collect::<Vec<_>>();
    let lengths = (1..=ids.len()).map(|n| n as i64).collect::<Vec<_>>();
    let logits = decoder.forward(
        &ids,
        &lengths,
        &table.repeat(ids.len()),
        &mut key,
        &mut value,
        &rows,
        false,
    )?;
    let tokenizer = TokenizerBundle::from_gguf(model.mapping())?;
    let text = input["text"].as_str().unwrap_or("hello");
    println!(
        "{}",
        json!({"logits":logits,"encoded":tokenizer.encode(text)?,"vocab_size":model.vocab_size()})
    );
    Ok(())
}
