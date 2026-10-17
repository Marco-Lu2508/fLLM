//! Transport for the multimodal logit checks (`scripts/verify_multimodal.py`): runs one prompt with an
//! image through the vision tower stored in the model file and the CPU decoder, and prints the logits
//! of every position as JSON.
use fllm_engine::{kv_format::KvFormat, rust_decoder::RustDecoder, vision, GgufModel};
use serde_json::{json, Value};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let model_path = args.first().ok_or("usage: verify_multimodal model.gguf request.json")?;
    let request: Value = serde_json::from_slice(&fs::read(args.get(1).ok_or("request.json required")?)?)?;
    let ids = request["ids"]
        .as_array()
        .ok_or("ids array required")?
        .iter()
        .map(|v| v.as_i64().ok_or("invalid token ID"))
        .collect::<Result<Vec<_>, _>>()?;
    let mut model = GgufModel::load(model_path, 256, 1, 2)?;
    // a full-precision cache keeps the comparison with the reference free of cache rounding
    model.set_kv_formats(KvFormat::F32, KvFormat::F32);
    let encoder = vision::load_from_model(model.mapping(), 4, "cpu")?.ok_or("the model file has no vision tower")?;
    let media = match (request["image"].as_str(), request["video"].as_array()) {
        (Some(path), _) => Some(encoder.encode_image(&fs::read(path)?)?),
        (None, Some(frames)) => {
            let frames = frames
                .iter()
                .map(|p| Ok(fs::read(p.as_str().ok_or("video frames are file paths")?)?))
                .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
            Some(encoder.encode_video(&frames, request["fps"].as_f64().unwrap_or(2.0) as f32)?)
        }
        _ => None,
    };
    let decoder = RustDecoder::load(&model, "cpu")?;
    let mut overrides = Vec::new();
    let mut grid = Value::Null;
    let mut deepstack = Vec::new();
    if let Some(media) = media {
        grid = json!(media.grid);
        let atomic = |_: &str| Ok(None);
        let mut inputs = vision::pipeline::place_media(&*encoder, &ids, vec![media], &atomic)?;
        // VERIFY_PLAIN_ROPE=1 runs the same prompt with ordinary positions (negative control for M-RoPE)
        if std::env::var_os("VERIFY_PLAIN_ROPE").is_none() {
            decoder.set_row_positions3(inputs.mrope.take().map(|m| m.prompt));
        }
        if let Some(vis) = inputs.visible_lengths {
            decoder.set_row_visible_lengths(Some(vis));
        }
        deepstack = std::mem::take(&mut inputs.deepstack);
        overrides = inputs.overrides;
    }
    // VERIFY_NO_DEEPSTACK=1 drops the DeepStack features (negative control)
    if !deepstack.is_empty() && std::env::var_os("VERIFY_NO_DEEPSTACK").is_none() {
        decoder.set_row_deepstack(Some(deepstack));
    }
    let table = (0..model.table_stride()).map(|n| n as i64).collect::<Vec<_>>();
    let mut key = vec![0; model.key_buffer_len()?];
    let mut value = vec![0; model.value_buffer_len()?];
    let rows = (0..ids.len()).collect::<Vec<_>>();
    let lengths = (1..=ids.len()).map(|n| n as i64).collect::<Vec<_>>();
    let logits = decoder.forward_with_embeddings(
        &ids,
        &lengths,
        &table.repeat(ids.len()),
        &mut key,
        &mut value,
        &rows,
        false,
        &overrides,
    )?;
    println!("{}", json!({"logits": logits, "grid": grid, "vocab_size": model.vocab_size()}));
    Ok(())
}
