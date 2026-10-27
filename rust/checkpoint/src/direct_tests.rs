//! Loading a checkpoint without a GGUF file must give the model the converter would have written: every tensor byte,
//! type, dimension and metadata value is compared against `convert_checkpoint` (and `quantize_gguf` for `--quant`).
use crate::{
    convert_checkpoint, load_checkpoint_as_gguf, load_checkpoint_quantized, quantize_gguf, ConvertOptions, OutputDType, Preset,
};
use fllm_gguf_probe::MappedGguf;
use half::{bf16, f16};
use serde_json::json;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

const HIDDEN: usize = 256;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const INTER: usize = 256;
const LAYERS: usize = 2;
const VOCAB: usize = 48;

fn scratch(tag: &str) -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let p = std::env::temp_dir().join(format!("fllm-direct-{tag}-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).unwrap();
    p
}

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
}

#[derive(Clone, Copy)]
enum Store {
    Bf16,
    F16,
    F32,
}

fn tensor_names() -> Vec<(String, Vec<usize>)> {
    let mut v = vec![("model.embed_tokens.weight".to_owned(), vec![VOCAB, HIDDEN]), ("lm_head.weight".to_owned(), vec![VOCAB, HIDDEN]), ("model.norm.weight".to_owned(), vec![HIDDEN])];
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}");
        v.push((format!("{p}.input_layernorm.weight"), vec![HIDDEN]));
        v.push((format!("{p}.post_attention_layernorm.weight"), vec![HIDDEN]));
        v.push((format!("{p}.self_attn.q_proj.weight"), vec![HEADS * HEAD_DIM, HIDDEN]));
        v.push((format!("{p}.self_attn.k_proj.weight"), vec![KV_HEADS * HEAD_DIM, HIDDEN]));
        v.push((format!("{p}.self_attn.v_proj.weight"), vec![KV_HEADS * HEAD_DIM, HIDDEN]));
        v.push((format!("{p}.self_attn.o_proj.weight"), vec![HIDDEN, HEADS * HEAD_DIM]));
        v.push((format!("{p}.mlp.gate_proj.weight"), vec![INTER, HIDDEN]));
        v.push((format!("{p}.mlp.up_proj.weight"), vec![INTER, HIDDEN]));
        v.push((format!("{p}.mlp.down_proj.weight"), vec![HIDDEN, INTER]));
    }
    v
}

fn write_file(path: &Path, entries: &[(String, &'static str, Vec<usize>, Vec<u8>)]) {
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    for (name, dtype, shape, bytes) in entries {
        header.insert(name.clone(), json!({"dtype": dtype, "shape": shape, "data_offsets": [payload.len(), payload.len() + bytes.len()]}));
        payload.extend_from_slice(bytes);
    }
    let text = serde_json::Value::Object(header).to_string();
    // pad the header to 8 bytes like the reference writer does; an odd header length would test unaligned tensors instead
    let text = format!("{text}{}", " ".repeat((8 - text.len() % 8) % 8));
    let mut out = (text.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(text.as_bytes());
    out.extend_from_slice(&payload);
    fs::write(path, out).unwrap();
}

/// A tiny Llama checkpoint in `store`, as one file or as `shards` files with an index.
fn llama(dir: &Path, store: Store, shards: usize, seed: u64) {
    llama_quantized(dir, store, shards, seed, None);
}

/// A packed-weights layout of the attention and MLP linears (random codes: only the converter's handling is compared).
#[derive(Clone, Copy)]
enum Packed {
    Gptq { bits: usize },
    Awq,
}

const GROUP: usize = 32;

fn llama_quantized(dir: &Path, store: Store, shards: usize, seed: u64, packed: Option<Packed>) {
    let mut cfg = json!({
        "model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": HIDDEN, "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS, "num_hidden_layers": LAYERS, "intermediate_size": INTER, "vocab_size": VOCAB,
        "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "max_position_embeddings": 128, "hidden_act": "silu",
        "tie_word_embeddings": false, "bos_token_id": 1, "eos_token_id": 2
    });
    match packed {
        Some(Packed::Gptq { bits }) => cfg["quantization_config"] = json!({"quant_method": "gptq", "bits": bits, "group_size": GROUP, "desc_act": false, "sym": false}),
        Some(Packed::Awq) => cfg["quantization_config"] = json!({"quant_method": "awq", "bits": 4, "group_size": GROUP, "version": "gemm", "zero_point": true}),
        None => {}
    }
    fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
    let vocab: serde_json::Map<String, serde_json::Value> = (0..VOCAB).map(|i| (format!("t{i}"), json!(i))).collect();
    fs::write(dir.join("tokenizer.json"), json!({"version": "1.0", "added_tokens": [], "model": {"type": "BPE", "vocab": vocab, "merges": []}}).to_string()).unwrap();
    fs::write(dir.join("tokenizer_config.json"), json!({"bos_token": "t1", "eos_token": "t2"}).to_string()).unwrap();
    let mut rng = Rng(seed);
    let mut entries = Vec::new();
    for (name, shape) in tensor_names() {
        let n: usize = shape.iter().product();
        if let (Some(layout), true) = (packed, name.contains("_proj.weight")) {
            let (out, inp) = (shape[0], shape[1]);
            let prefix = name.trim_end_matches(".weight");
            let words = |rng: &mut Rng, count: usize| -> Vec<u8> { (0..count).flat_map(|_| ((rng.unit() * 2e9) as i32).to_le_bytes()).collect() };
            let scales: Vec<u8> = (0..inp / GROUP * out).flat_map(|_| f16::from_f32(0.01 + 0.01 * rng.unit().abs()).to_le_bytes()).collect();
            let (qweight, qzeros) = match layout {
                Packed::Gptq { bits } => {
                    let per = 32 / bits;
                    ((vec![inp / per, out], words(&mut rng, inp / per * out)), (vec![inp / GROUP, out / per], words(&mut rng, inp / GROUP * out / per)))
                }
                Packed::Awq => ((vec![inp, out / 8], words(&mut rng, inp * out / 8)), (vec![inp / GROUP, out / 8], words(&mut rng, inp / GROUP * out / 8))),
            };
            entries.push((format!("{prefix}.qweight"), "I32", qweight.0, qweight.1));
            entries.push((format!("{prefix}.qzeros"), "I32", qzeros.0, qzeros.1));
            entries.push((format!("{prefix}.scales"), "F16", vec![inp / GROUP, out], scales));
            continue;
        }
        let norm = name.contains("norm");
        let values: Vec<f32> = (0..n).map(|_| if norm { 1.0 + 0.1 * rng.unit() } else { 0.5 * rng.unit() }).collect();
        let (dtype, bytes): (&'static str, Vec<u8>) = match store {
            Store::Bf16 => ("BF16", values.iter().flat_map(|v| bf16::from_f32(*v).to_le_bytes()).collect()),
            Store::F16 => ("F16", values.iter().flat_map(|v| f16::from_f32(*v).to_le_bytes()).collect()),
            Store::F32 => ("F32", values.iter().flat_map(|v| v.to_le_bytes()).collect()),
        };
        entries.push((name, dtype, shape, bytes));
    }
    if shards <= 1 {
        write_file(&dir.join("model.safetensors"), &entries);
        return;
    }
    let per = entries.len().div_ceil(shards);
    let mut weight_map = serde_json::Map::new();
    for (i, chunk) in entries.chunks(per).enumerate() {
        let file = format!("model-{:05}-of-{:05}.safetensors", i + 1, shards);
        write_file(&dir.join(&file), chunk);
        for (name, ..) in chunk {
            weight_map.insert(name.clone(), json!(file));
        }
    }
    fs::write(dir.join("model.safetensors.index.json"), json!({"metadata": {}, "weight_map": weight_map}).to_string()).unwrap();
}

/// Every tensor (name, dims, type, bytes) and every metadata value of the two models is equal.
fn assert_same(direct: &MappedGguf, file: &MappedGguf, what: &str) {
    assert_eq!(direct.index().metadata, file.index().metadata, "{what}: metadata");
    assert_eq!(direct.index().tensors.len(), file.index().tensors.len(), "{what}: tensor count");
    for t in &direct.index().tensors {
        let r = file.index().tensors.iter().find(|r| r.name == t.name).unwrap_or_else(|| panic!("{what}: {} missing", t.name));
        assert_eq!(t.dimensions, r.dimensions, "{what}: dims of {}", t.name);
        assert_eq!(t.type_id, r.type_id, "{what}: type of {}", t.name);
        assert_eq!(t.payload_len, r.payload_len, "{what}: size of {}", t.name);
        assert!(direct.tensor_bytes(&t.name).unwrap() == file.tensor_bytes(&t.name).unwrap(), "{what}: bytes of {}", t.name);
    }
}

fn convert_to(dir: &Path, name: &str, dtype: OutputDType) -> PathBuf {
    let out = dir.join(name);
    convert_checkpoint(dir, &out, ConvertOptions { output_dtype: dtype, overwrite: false, spec_head: None }).unwrap();
    out
}

#[test]
fn unquantized_checkpoints_load_identically_to_the_converted_file() {
    for (store, shards, outtype) in [
        (Store::Bf16, 1, OutputDType::Preserve),
        (Store::Bf16, 3, OutputDType::Preserve),
        (Store::F16, 2, OutputDType::Preserve),
        (Store::F32, 1, OutputDType::Preserve),
        (Store::Bf16, 1, OutputDType::F16),
        (Store::F16, 1, OutputDType::F32),
        (Store::F32, 2, OutputDType::F16),
    ] {
        let dir = scratch("plain");
        llama(&dir, store, shards, 7);
        let options = ConvertOptions { output_dtype: outtype, overwrite: false, spec_head: None };
        let direct = load_checkpoint_as_gguf(&dir, options).unwrap();
        let file = MappedGguf::open(convert_to(&dir, "ref.gguf", outtype)).unwrap();
        assert_same(&direct, &file, &format!("{shards} shards, {outtype:?}"));
        let _ = fs::remove_dir_all(dir);
    }
}

#[test]
fn quantizing_while_loading_equals_convert_then_quantize() {
    for (store, shards, preset) in [
        (Store::Bf16, 1, "q8_0"),
        (Store::Bf16, 2, "q4_k_m"),
        (Store::F16, 1, "q4_0"),
        (Store::F32, 1, "q6_k"),
        (Store::Bf16, 1, "q5_k_s"),
        (Store::Bf16, 1, "q2_k"),
        (Store::Bf16, 1, "mxfp4"),
    ] {
        let dir = scratch("quant");
        llama(&dir, store, shards, 11);
        let preset = Preset::parse(preset).unwrap();
        let options = ConvertOptions { output_dtype: OutputDType::F16, overwrite: false, spec_head: None };
        let direct = load_checkpoint_quantized(&dir, options, Some(preset)).unwrap();
        let stage = convert_to(&dir, "stage.gguf", OutputDType::F16);
        let quantized = dir.join("quantized.gguf");
        quantize_gguf(&stage, &quantized, preset).unwrap();
        let file = MappedGguf::open(&quantized).unwrap();
        assert_same(&direct, &file, &format!("{preset:?} from {shards} shards"));
        // the point of the mode: the large weights really are in the smaller type
        let q = direct.index().tensors.iter().find(|t| t.name == "blk.0.ffn_up.weight").unwrap();
        assert_ne!(q.type_id, 1, "{preset:?}: ffn_up stayed half precision");
        let _ = fs::remove_dir_all(dir);
    }
}

#[test]
fn untouched_tensors_stay_slices_of_the_checkpoint() {
    let dir = scratch("zerocopy");
    llama(&dir, Store::Bf16, 1, 3);
    let direct = load_checkpoint_as_gguf(&dir, ConvertOptions::default()).unwrap();
    let embd = direct.extra_mappings().get("token_embd.weight").unwrap();
    let bytes = direct.tensor_bytes("token_embd.weight").unwrap();
    let (m0, m1) = (embd.as_ptr() as usize, embd.as_ptr() as usize + embd.len());
    assert!(bytes.as_ptr() as usize >= m0 && bytes.as_ptr() as usize + bytes.len() <= m1);
    // the rope-permuted attention weights are built once, in the anonymous map shared by all built tensors
    let q = direct.extra_mappings().get("blk.0.attn_q.weight").unwrap();
    assert!(!std::sync::Arc::ptr_eq(embd, q));
    assert!(std::sync::Arc::ptr_eq(q, direct.extra_mappings().get("blk.1.attn_k.weight").unwrap()));
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn packed_gptq_and_awq_checkpoints_load_identically_to_the_converted_file() {
    for (packed, label, shards) in [(Packed::Gptq { bits: 4 }, "gptq4", 1), (Packed::Gptq { bits: 8 }, "gptq8", 2), (Packed::Awq, "awq", 1)] {
        let dir = scratch(label);
        llama_quantized(&dir, Store::F16, shards, 5, Some(packed));
        let options = ConvertOptions::default();
        let direct = load_checkpoint_as_gguf(&dir, options).unwrap();
        let file = MappedGguf::open(convert_to(&dir, "ref.gguf", OutputDType::Preserve)).unwrap();
        assert_same(&direct, &file, label);
        let _ = fs::remove_dir_all(dir);
    }
}
