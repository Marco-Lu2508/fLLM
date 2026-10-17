//! Gemma 4 and Gemma 3n on the CPU against tiny transformers checkpoints.
//!
//! `rust/engine/tests/fixtures/{gemma4,gemma3n}-tiny` hold a float32 checkpoint (config.json, model.safetensors) and the
//! logits transformers computes for ten fixed tokens (`golden.json`); `scripts/make_gemma_fixtures.py` regenerates them.
//! Every test converts the checkpoint with the importer, loads the GGUF and runs the engine with an exact f32 KV cache,
//! so the numbers can be compared with the reference to about 1e-5.
use crate::{config::ModelConfig, kv_format::KvFormat, rust_decoder::RustDecoder, GgufModel};
use fllm_checkpoint::{convert_checkpoint, ConvertOptions, OutputDType};
use fllm_gguf_probe::MappedGguf;
use serde_json::{json, Value};
use std::{fs, path::{Path, PathBuf}};

const VOCAB: usize = 32;

struct Fixture {
    gguf: PathBuf,
    ids: Vec<i64>,
    logits: Vec<f32>,
    dir: PathBuf,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

/// Copy the fixture checkpoint into a scratch directory, add a tokenizer and convert it to a float32 GGUF.
fn fixture(name: &str) -> Fixture {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name);
    let dir = crate::import_tests::scratch(name);
    for file in ["config.json", "model.safetensors"] {
        fs::copy(source.join(file), dir.join(file)).unwrap();
    }
    let vocab: serde_json::Map<String, Value> = (0..VOCAB)
        .map(|i| {
            let piece = match i {
                0 => "<pad>".to_owned(),
                1 => "<eos>".to_owned(),
                2 => "<bos>".to_owned(),
                n => format!("t{n}"),
            };
            (piece, json!(i))
        })
        .collect();
    let tokenizer = json!({"version": "1.0", "added_tokens": [], "model": {"type": "BPE", "vocab": vocab, "merges": []}});
    fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).unwrap();
    fs::write(
        dir.join("tokenizer_config.json"),
        json!({"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>"}).to_string(),
    )
    .unwrap();
    let gguf = dir.join("model.gguf");
    convert_checkpoint(&dir, &gguf, ConvertOptions { output_dtype: OutputDType::F32, overwrite: false, spec_head: None })
        .unwrap_or_else(|e| panic!("conversion of {name} failed: {e}"));
    let golden: Value = serde_json::from_slice(&fs::read(source.join("golden.json")).unwrap()).unwrap();
    Fixture {
        gguf,
        ids: golden["ids"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect(),
        logits: golden["logits"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect(),
        dir,
    }
}

fn load(f: &Fixture) -> (GgufModel, RustDecoder) {
    let mut model = GgufModel::load(f.gguf.to_str().unwrap(), 64, 1, 2).unwrap();
    model.set_kv_formats(KvFormat::F32, KvFormat::F32);
    let decoder = RustDecoder::load(&model, "cpu").unwrap();
    (model, decoder)
}

fn table(model: &GgufModel, rows: usize) -> Vec<i64> {
    let stride = model.table_stride();
    (0..rows).flat_map(|_| 0..stride as i64).collect()
}

fn relative_difference(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let (low, high) = want.iter().fold((f32::MAX, f32::MIN), |(l, h), v| (l.min(*v), h.max(*v)));
    got.iter().zip(want).fold(0f32, |m, (a, b)| m.max((a - b).abs())) / (high - low)
}

/// All positions in one batch: every row attends to the rows before it through the cache written in the same pass.
fn batch_logits(f: &Fixture) -> Vec<f32> {
    let (model, decoder) = load(f);
    let mut key = vec![0u16; model.key_buffer_len().unwrap()];
    let mut value = vec![0u16; model.value_buffer_len().unwrap()];
    let n = f.ids.len();
    let lengths: Vec<i64> = (1..=n as i64).collect();
    let rows: Vec<usize> = (0..n).collect();
    decoder.forward(&f.ids, &lengths, &table(&model, n), &mut key, &mut value, &rows, false).unwrap()
}

#[test]
fn gemma4_matches_transformers() {
    let f = fixture("gemma4-tiny");
    let got = batch_logits(&f);
    let difference = relative_difference(&got, &f.logits);
    assert!(difference < 1e-4, "largest relative logit difference {difference}");
}

#[test]
fn gemma3n_matches_transformers() {
    let f = fixture("gemma3n-tiny");
    let got = batch_logits(&f);
    let difference = relative_difference(&got, &f.logits);
    assert!(difference < 1e-4, "largest relative logit difference {difference}");
}

/// Prompt in one pass, then one token per step through the cache: sliding layers, shared layers and the mixed cache
/// regions must give the same logits as the single batch.
fn incremental_matches_batch(name: &str) {
    let f = fixture(name);
    let (model, decoder) = load(&f);
    let mut key = vec![0u16; model.key_buffer_len().unwrap()];
    let mut value = vec![0u16; model.value_buffer_len().unwrap()];
    let prompt = 4;
    let rows: Vec<usize> = (0..prompt).collect();
    let lengths: Vec<i64> = (1..=prompt as i64).collect();
    let mut got = decoder
        .forward(&f.ids[..prompt], &lengths, &table(&model, prompt), &mut key, &mut value, &rows, false)
        .unwrap();
    for position in prompt..f.ids.len() {
        let step = decoder
            .forward(&f.ids[position..=position], &[position as i64 + 1], &table(&model, 1), &mut key, &mut value, &[0], false)
            .unwrap();
        got.extend(step);
    }
    let difference = relative_difference(&got, &f.logits);
    assert!(difference < 1e-4, "{name}: largest relative logit difference {difference}");
}

#[test]
fn gemma4_decode_through_the_cache_matches_the_batch() {
    incremental_matches_batch("gemma4-tiny");
}

#[test]
fn gemma3n_decode_through_the_cache_matches_the_batch() {
    incremental_matches_batch("gemma3n-tiny");
}

#[test]
fn gemma4_conversion_describes_the_layer_geometry() {
    let f = fixture("gemma4-tiny");
    let mapping = MappedGguf::open(&f.gguf).unwrap();
    let cfg = ModelConfig::from_gguf(&mapping).unwrap();
    assert_eq!(cfg.arch, "gemma4");
    let geoms = cfg.layer_geoms.as_ref().expect("per-layer geometry");
    // sliding layers: 2 heads of width 8 with 2 KV heads; global layers: width 16 with one KV head (K = V)
    let shape = |l: usize| (geoms[l].heads, geoms[l].kv_heads, geoms[l].head_dim, geoms[l].value_dim, geoms[l].kv_source, geoms[l].ff);
    assert_eq!(shape(0), (2, 2, 8, 8, None, 16));
    assert_eq!(shape(2), (2, 1, 16, 16, None, 16));
    // the last two layers read the caches of the last earlier layer of their kind and have a double-wide MLP
    assert_eq!(shape(3), (2, 2, 8, 8, Some(1), 32));
    assert_eq!(shape(4), (2, 1, 16, 16, Some(2), 32));
    assert_eq!((cfg.rope.dims, cfg.rope.local_dims), (16, Some(8)));
    assert!(cfg.value_norm && cfg.attention_scale == Some(1.0));
    assert_eq!(cfg.layer_rope_dims(0), 8);
    assert_eq!(cfg.layer_rope_dims(2), 16);
    let names: Vec<String> = mapping.index().tensors.iter().map(|t| t.name.clone()).collect();
    let has = |n: &str| names.iter().any(|t| t == n);
    assert!(has("blk.2.attn_k.weight") && !has("blk.2.attn_v.weight"), "K = V layers carry no value projection");
    assert!(!has("blk.3.attn_k.weight") && !has("blk.4.attn_v.weight"), "shared layers carry no key / value projection");
    assert!(has("rope_freqs.weight") && has("blk.0.ffn_gate_up_exps.weight") && has("per_layer_token_embd.weight"));
}

#[test]
fn shared_layers_own_no_cache_and_the_pools_follow_each_layers_geometry() {
    let f = fixture("gemma4-tiny");
    let (model, _) = load(&f);
    let layout = model.kv_pool_layout();
    assert_eq!(layout.key_words[3], 0);
    assert_eq!(layout.key_words[4], 0);
    // slots x kv_heads x head_dim 16-bit values... as f32 pairs: 2 words each
    let slots = 64usize.div_ceil(16) * 16;
    assert_eq!(layout.key_words[0], slots * 2 * 8 * 2);
    assert_eq!(layout.key_words[2], slots * 16 * 2);
    assert_eq!(layout.key_total, layout.key_words.iter().sum::<usize>());
    assert_eq!(model.key_buffer_len().unwrap(), layout.key_total);
}


#[test]
fn llama_cpp_gemma4_layout_loads_and_initializes() {
    use fllm_checkpoint::{gguf::{write_to, Meta, OutTensor}, safetensors::Payload};
    use std::sync::Arc;

    let scratch = crate::import_tests::scratch("llama_cpp_gemma4_test");
    let gguf_path = scratch.join("gemma4_llamacpp.gguf");

    let meta = vec![
        ("general.architecture".to_string(), Meta::Str("gemma4".to_string())),
        ("general.name".to_string(), Meta::Str("gemma4-tiny".to_string())),
        ("general.alignment".to_string(), Meta::U32(32)),
        ("gemma4.block_count".to_string(), Meta::U32(2)),
        ("gemma4.context_length".to_string(), Meta::U32(64)),
        ("gemma4.embedding_length".to_string(), Meta::U32(16)),
        ("gemma4.feed_forward_length".to_string(), Meta::U32s(vec![32, 32])),
        ("gemma4.attention.head_count".to_string(), Meta::U32s(vec![2, 2])),
        ("gemma4.attention.head_count_kv".to_string(), Meta::U32s(vec![2, 2])),
        ("gemma4.attention.key_length".to_string(), Meta::U32(8)),
        ("gemma4.attention.key_length_swa".to_string(), Meta::U32(8)),
        ("gemma4.attention.value_length".to_string(), Meta::U32(8)),
        ("gemma4.attention.value_length_swa".to_string(), Meta::U32(8)),
        ("gemma4.attention.sliding_window".to_string(), Meta::U32(16)),
        ("gemma4.rope.dimension_count".to_string(), Meta::U32(8)),
        ("gemma4.rope.dimension_count_swa".to_string(), Meta::U32(8)),
        ("gemma4.rope.freq_base".to_string(), Meta::F32(10000.0)),
        ("gemma4.rope.freq_base_swa".to_string(), Meta::F32(10000.0)),
        ("tokenizer.ggml.tokens".to_string(), Meta::Strs(vec!["<pad>".into(), "<s>".into(), "</s>".into(), "hello".into()])),
        ("tokenizer.ggml.token_type".to_string(), Meta::U32s(vec![2, 3, 3, 1])),
        ("tokenizer.ggml.scores".to_string(), Meta::F32s(vec![0.0, 0.0, 0.0, 0.0])),
        ("tokenizer.ggml.bos_token_id".to_string(), Meta::U32(1)),
        ("tokenizer.ggml.eos_token_id".to_string(), Meta::U32(2)),
    ];

    let make_tensor = |name: &str, dims: Vec<u64>| -> OutTensor {
        let count: usize = dims.iter().map(|&d| d as usize).product();
        let bytes = vec![0u8; count * 4];
        let len = bytes.len() as u64;
        OutTensor {
            name: name.to_string(),
            dims,
            source_dtype: "F32".into(),
            dtype: "F32".into(),
            data: Payload::Owned(Arc::new(Vec::new()), 0..0),
            packed: None,
            data_len: len,
            source_for_output: None,
            add: 0.0,
            owned: Some(bytes),
            parts: Vec::new(),
        }
    };

    let tensors = vec![
        make_tensor("token_embd.weight", vec![16, 4]),
        make_tensor("output_norm.weight", vec![16]),
        make_tensor("output.weight", vec![16, 4]),
        make_tensor("blk.0.attn_norm.weight", vec![16]),
        make_tensor("blk.0.attn_q.weight", vec![16, 16]),
        make_tensor("blk.0.attn_k.weight", vec![16, 16]),
        make_tensor("blk.0.attn_v.weight", vec![16, 16]),
        make_tensor("blk.0.attn_output.weight", vec![16, 16]),
        make_tensor("blk.0.ffn_norm.weight", vec![16]),
        make_tensor("blk.0.ffn_gate.weight", vec![16, 32]),
        make_tensor("blk.0.ffn_up.weight", vec![16, 32]),
        make_tensor("blk.0.ffn_down.weight", vec![32, 16]),
        make_tensor("blk.1.attn_norm.weight", vec![16]),
        make_tensor("blk.1.attn_q.weight", vec![16, 16]),
        make_tensor("blk.1.attn_k.weight", vec![16, 16]),
        make_tensor("blk.1.attn_v.weight", vec![16, 16]),
        make_tensor("blk.1.attn_output.weight", vec![16, 16]),
        make_tensor("blk.1.ffn_norm.weight", vec![16]),
        make_tensor("blk.1.ffn_gate.weight", vec![16, 32]),
        make_tensor("blk.1.ffn_up.weight", vec![16, 32]),
        make_tensor("blk.1.ffn_down.weight", vec![32, 16]),
    ];

    let mut file = std::fs::File::create(&gguf_path).unwrap();
    write_to(&mut file, &meta, &tensors).unwrap();
    drop(file);

    let model = GgufModel::load(gguf_path.to_str().unwrap(), 64, 1, 2).unwrap();
    assert_eq!(model.config().arch, "gemma4");
    assert_eq!(model.config().layers, 2);
    let decoder = RustDecoder::load(&model, "cpu").unwrap();
    let mut key = vec![0u16; model.key_buffer_len().unwrap()];
    let mut value = vec![0u16; model.value_buffer_len().unwrap()];
    let stride = model.table_stride();
    let tables: Vec<i64> = (0..stride as i64).collect();
    let out = decoder.forward(&[1], &[1], &tables, &mut key, &mut value, &[0], false).unwrap();
    assert_eq!(out.len(), 4);
    let _ = std::fs::remove_dir_all(&scratch);
}
