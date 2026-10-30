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
    // FLLM_VERIFY_KV=f32 keeps the cache in single precision (the default f16 cache alone costs about 1e-3 of relative logit error)
    if std::env::var("FLLM_VERIFY_KV").is_ok_and(|v| v.eq_ignore_ascii_case("f32")) {
        model.set_kv_formats(fllm_engine::kv_format::KvFormat::F32, fllm_engine::kv_format::KvFormat::F32);
    } else {
        model.set_kv_kind(1);
    }
    let decoder = RustDecoder::load(&model, "cpu")?;
    let table = (0..model.table_stride())
        .map(|n| n as i64)
        .collect::<Vec<_>>();
    let mut key = vec![0; model.key_buffer_len()?];
    let mut value = vec![0; model.value_buffer_len()?];
    let rows = (0..ids.len()).collect::<Vec<_>>();
    let lengths = (1..=ids.len()).map(|n| n as i64).collect::<Vec<_>>();
    // FLLM_VERIFY_SPLIT=K runs the first K tokens as one batch and every later token as a batch of its own (cached decoding)
    let split = std::env::var("FLLM_VERIFY_SPLIT").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&k| k >= 1 && k < ids.len());
    let first = split.unwrap_or(ids.len());
    let mut logits = decoder.forward(
        &ids[..first],
        &lengths[..first],
        &table.repeat(first),
        &mut key,
        &mut value,
        &rows[..first],
        false,
    )?;
    for n in first..ids.len() {
        logits.extend(decoder.forward(&ids[n..=n], &lengths[n..=n], &table, &mut key, &mut value, &[0], false)?);
    }
    let tokenizer = TokenizerBundle::from_gguf(model.mapping())?;
    let text = input["text"].as_str().unwrap_or("hello");
    println!(
        "{}",
        json!({"logits":logits,"encoded":tokenizer.encode(text)?,"vocab_size":model.vocab_size()})
    );
    Ok(())
}
