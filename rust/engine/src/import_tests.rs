//! End-to-end checks of the checkpoint import with tiny synthetic models, CPU only.
//!
//! Every test builds a small Llama-like checkpoint in a quantized layout (NVIDIA ModelOpt NVFP4, compressed-tensors
//! int4 / int8, block-wise FP8), converts it with `fllm_checkpoint::convert_checkpoint` and runs the engine on the
//! result. The reference is the same model as a plain F32 checkpoint whose linear weights were dequantized in this
//! file with the formulas of the quantization libraries, so the importer and the engine are checked against an
//! independent computation.

use crate::{rust_decoder::RustDecoder, GgufModel};
use fllm_checkpoint::{convert_checkpoint, ConvertOptions, OutputDType};
use fllm_gguf_probe::MappedGguf;
use serde_json::{json, Value};
use std::{fs, path::{Path, PathBuf}};

const HIDDEN: usize = 64;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 16;
const INTER: usize = 128;
const LAYERS: usize = 2;
const VOCAB: usize = 64;

pub(crate) fn scratch(label: &str) -> PathBuf {
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!("fllm-import-{label}-{}-{n}", std::process::id()));
    fs::create_dir_all(&p).unwrap();
    p
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        (self.next() >> 11) % n
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

struct Entry {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn f32_entry(name: &str, shape: &[usize], values: &[f32]) -> Entry {
    Entry {
        name: name.into(),
        dtype: "F32",
        shape: shape.to_vec(),
        bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}
fn bf16_entry(name: &str, shape: &[usize], values: &[f32]) -> Entry {
    Entry {
        name: name.into(),
        dtype: "BF16",
        shape: shape.to_vec(),
        bytes: values.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect(),
    }
}

fn write_safetensors(path: &Path, entries: &[Entry]) {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for e in entries {
        header.insert(
            e.name.clone(),
            json!({"dtype": e.dtype, "shape": e.shape, "data_offsets": [offset, offset + e.bytes.len()]}),
        );
        offset += e.bytes.len();
    }
    let header = Value::Object(header).to_string();
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    for e in entries {
        out.extend_from_slice(&e.bytes);
    }
    fs::write(path, out).unwrap();
}

// Independent decoders of the quantization formats (written from the format definitions, not from the importer).
fn ref_e2m1(code: u8) -> f32 {
    let mag = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(code & 7) as usize];
    if code & 8 != 0 {
        -mag
    } else {
        mag
    }
}
fn ref_e4m3(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let (e, m) = (((byte >> 3) & 15) as i32, (byte & 7) as f32);
    sign * if e == 0 { m / 8.0 * 2f32.powi(-6) } else { (1.0 + m / 8.0) * 2f32.powi(e - 7) }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Kind {
    Nvfp4,
    /// compressed-tensors pack-quantized int4 with this group size
    CtInt4(usize),
    /// compressed-tensors pack-quantized int8, per channel
    CtPacked8,
    /// compressed-tensors int-quantized int8, per channel
    CtInt8,
    /// block-wise FP8 (`weight_scale_inv`), square tiles of this size
    Fp8Block(usize),
}

/// A linear layer of the model: the entries of the quantized checkpoint and the dequantized F32 values.
fn quantized_linear(rng: &mut Rng, kind: Kind, prefix: &str, out: usize, inp: usize) -> (Vec<Entry>, Vec<f32>) {
    let mut entries = Vec::new();
    let mut dense = vec![0f32; out * inp];
    match kind {
        Kind::Nvfp4 => {
            let global = 0.01 + 0.02 * (rng.below(100) as f32) / 100.0;
            let codes: Vec<u8> = (0..out * inp).map(|_| rng.below(16) as u8).collect();
            let scales: Vec<u8> = (0..out * inp / 16).map(|_| 0x30 + rng.below(12) as u8).collect();
            for r in 0..out {
                for c in 0..inp {
                    dense[r * inp + c] =
                        ref_e2m1(codes[r * inp + c]) * ref_e4m3(scales[r * inp / 16 + c / 16]) * global;
                }
            }
            // ModelOpt packs the even input index into the low nibble
            let packed: Vec<u8> = codes.chunks(2).map(|p| p[0] | (p[1] << 4)).collect();
            entries.push(Entry { name: format!("{prefix}.weight"), dtype: "U8", shape: vec![out, inp / 2], bytes: packed });
            entries.push(Entry { name: format!("{prefix}.weight_scale"), dtype: "F8_E4M3", shape: vec![out, inp / 16], bytes: scales });
            entries.push(f32_entry(&format!("{prefix}.weight_scale_2"), &[], &[global]));
            entries.push(f32_entry(&format!("{prefix}.input_scale"), &[], &[0.25]));
        }
        Kind::CtInt4(group) => {
            let codes: Vec<u8> = (0..out * inp).map(|_| rng.below(16) as u8).collect();
            let scales: Vec<f32> = (0..out * inp / group).map(|_| half::f16::from_f32(0.004 + 0.002 * rng.below(8) as f32).to_f32()).collect();
            for r in 0..out {
                for c in 0..inp {
                    dense[r * inp + c] = (codes[r * inp + c] as i32 - 8) as f32 * scales[r * (inp / group) + c / group];
                }
            }
            // eight codes per int32, input index i at bit 4 * (i % 8)
            let mut bytes = Vec::new();
            for word in codes.chunks(8) {
                let w = word.iter().enumerate().fold(0u32, |a, (i, &c)| a | ((c as u32) << (4 * i)));
                bytes.extend_from_slice(&w.to_le_bytes());
            }
            entries.push(Entry { name: format!("{prefix}.weight_packed"), dtype: "I32", shape: vec![out, inp / 8], bytes });
            entries.push(Entry {
                name: format!("{prefix}.weight_scale"),
                dtype: "F16",
                shape: vec![out, inp / group],
                bytes: scales.iter().flat_map(|s| half::f16::from_f32(*s).to_bits().to_le_bytes()).collect(),
            });
            let shape: Vec<u8> = [out as i64, inp as i64].iter().flat_map(|v| v.to_le_bytes()).collect();
            entries.push(Entry { name: format!("{prefix}.weight_shape"), dtype: "I64", shape: vec![2], bytes: shape });
        }
        Kind::CtPacked8 | Kind::CtInt8 => {
            let raw: Vec<u8> = (0..out * inp).map(|_| rng.below(256) as u8).collect();
            let scales: Vec<f32> = (0..out).map(|_| half::f16::from_f32(0.0004 + 0.0002 * rng.below(8) as f32).to_f32()).collect();
            let packed8 = matches!(kind, Kind::CtPacked8);
            for r in 0..out {
                for c in 0..inp {
                    let b = raw[r * inp + c];
                    let q = if packed8 { b as i32 - 128 } else { b as i8 as i32 };
                    dense[r * inp + c] = q as f32 * scales[r];
                }
            }
            if packed8 {
                entries.push(Entry { name: format!("{prefix}.weight_packed"), dtype: "I32", shape: vec![out, inp / 4], bytes: raw });
            } else {
                entries.push(Entry { name: format!("{prefix}.weight"), dtype: "I8", shape: vec![out, inp], bytes: raw });
            }
            entries.push(Entry {
                name: format!("{prefix}.weight_scale"),
                dtype: "F16",
                shape: vec![out, 1],
                bytes: scales.iter().flat_map(|s| half::f16::from_f32(*s).to_bits().to_le_bytes()).collect(),
            });
            if packed8 {
                let shape: Vec<u8> = [out as i64, inp as i64].iter().flat_map(|v| v.to_le_bytes()).collect();
                entries.push(Entry { name: format!("{prefix}.weight_shape"), dtype: "I64", shape: vec![2], bytes: shape });
            }
        }
        Kind::Fp8Block(tile) => {
            let bytes: Vec<u8> = (0..out * inp).map(|_| (rng.below(2) as u8) << 7 | (0x28 + rng.below(16) as u8)).collect();
            let tiles = (out / tile) * (inp / tile);
            let scales: Vec<f32> = (0..tiles).map(|_| 0.01 + 0.005 * rng.below(8) as f32).collect();
            for r in 0..out {
                for c in 0..inp {
                    dense[r * inp + c] = ref_e4m3(bytes[r * inp + c]) * scales[(r / tile) * (inp / tile) + c / tile];
                }
            }
            entries.push(Entry { name: format!("{prefix}.weight"), dtype: "F8_E4M3", shape: vec![out, inp], bytes });
            entries.push(f32_entry(&format!("{prefix}.weight_scale_inv"), &[out / tile, inp / tile], &scales));
        }
    }
    (entries, dense)
}

fn quantization_config(kind: Kind) -> Value {
    match kind {
        Kind::Nvfp4 => json!({"quant_method": "modelopt", "quant_algo": "NVFP4", "group_size": 16,
            "kv_cache_quant_algo": "FP8", "exclude_modules": ["lm_head"]}),
        Kind::CtInt4(g) => json!({"quant_method": "compressed-tensors", "format": "pack-quantized", "ignore": ["lm_head"],
            "config_groups": {"group_0": {"targets": ["Linear"], "weights": {"num_bits": 4, "type": "int", "symmetric": true,
            "strategy": "group", "group_size": g}}}}),
        Kind::CtPacked8 => json!({"quant_method": "compressed-tensors", "format": "pack-quantized", "ignore": ["lm_head"],
            "config_groups": {"group_0": {"targets": ["Linear"], "weights": {"num_bits": 8, "type": "int", "symmetric": true,
            "strategy": "channel"}}}}),
        Kind::CtInt8 => json!({"quant_method": "compressed-tensors", "format": "int-quantized", "ignore": ["lm_head"],
            "config_groups": {"group_0": {"targets": ["Linear"], "weights": {"num_bits": 8, "type": "int", "symmetric": true,
            "strategy": "channel"}}}}),
        Kind::Fp8Block(t) => json!({"quant_method": "fp8", "activation_scheme": "dynamic", "fmt": "e4m3", "weight_block_size": [t, t]}),
    }
}

fn write_support_files(dir: &Path, quant: Option<Value>) {
    let mut cfg = json!({
        "model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": HIDDEN,
        "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS, "num_hidden_layers": LAYERS,
        "intermediate_size": INTER, "vocab_size": VOCAB, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
        "max_position_embeddings": 64, "hidden_act": "silu", "tie_word_embeddings": false,
        "bos_token_id": 1, "eos_token_id": 2, "torch_dtype": "bfloat16"
    });
    if let Some(q) = quant {
        cfg["quantization_config"] = q;
    }
    fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
    let vocab: serde_json::Map<String, Value> = (0..VOCAB)
        .map(|i| {
            let piece = match i {
                0 => "<unk>".to_owned(),
                1 => "<s>".to_owned(),
                2 => "</s>".to_owned(),
                n => format!("t{n}"),
            };
            (piece, json!(i))
        })
        .collect();
    let tok = json!({"version": "1.0", "added_tokens": [], "model": {"type": "BPE", "vocab": vocab, "merges": []}});
    fs::write(dir.join("tokenizer.json"), tok.to_string()).unwrap();
    fs::write(
        dir.join("tokenizer_config.json"),
        json!({"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>"}).to_string(),
    )
    .unwrap();
}

/// Build the same random model twice: quantized (`kind`) in `quant_dir`, dequantized F32 in `ref_dir`.
pub(crate) fn build_pair(kind: Kind, seed: u64, quant_dir: &Path, ref_dir: &Path) {
    let mut rng = Rng(seed);
    let mut q_entries = Vec::new();
    let mut r_entries = Vec::new();
    let both = |e: Entry, q: &mut Vec<Entry>, r: &mut Vec<Entry>| {
        r.push(Entry { name: e.name.clone(), dtype: e.dtype, shape: e.shape.clone(), bytes: e.bytes.clone() });
        q.push(e);
    };
    let embed: Vec<f32> = (0..VOCAB * HIDDEN).map(|_| rng.unit()).collect();
    both(bf16_entry("model.embed_tokens.weight", &[VOCAB, HIDDEN], &embed), &mut q_entries, &mut r_entries);
    let head: Vec<f32> = (0..VOCAB * HIDDEN).map(|_| 0.3 * rng.unit()).collect();
    both(bf16_entry("lm_head.weight", &[VOCAB, HIDDEN], &head), &mut q_entries, &mut r_entries);
    let norm: Vec<f32> = (0..HIDDEN).map(|_| 1.0 + 0.1 * rng.unit()).collect();
    both(bf16_entry("model.norm.weight", &[HIDDEN], &norm), &mut q_entries, &mut r_entries);
    for layer in 0..LAYERS {
        for n in ["input_layernorm", "post_attention_layernorm"] {
            let w: Vec<f32> = (0..HIDDEN).map(|_| 1.0 + 0.1 * rng.unit()).collect();
            both(bf16_entry(&format!("model.layers.{layer}.{n}.weight"), &[HIDDEN], &w), &mut q_entries, &mut r_entries);
        }
        let linears = [
            ("self_attn.q_proj", HEADS * HEAD_DIM, HIDDEN),
            ("self_attn.k_proj", KV_HEADS * HEAD_DIM, HIDDEN),
            ("self_attn.v_proj", KV_HEADS * HEAD_DIM, HIDDEN),
            ("self_attn.o_proj", HIDDEN, HEADS * HEAD_DIM),
            ("mlp.gate_proj", INTER, HIDDEN),
            ("mlp.up_proj", INTER, HIDDEN),
            ("mlp.down_proj", HIDDEN, INTER),
        ];
        for (name, out, inp) in linears {
            let prefix = format!("model.layers.{layer}.{name}");
            let (entries, dense) = quantized_linear(&mut rng, kind, &prefix, out, inp);
            q_entries.extend(entries);
            r_entries.push(f32_entry(&format!("{prefix}.weight"), &[out, inp], &dense));
        }
        if let Kind::Nvfp4 = kind {
            // FP8 KV-cache scales of ModelOpt exports: ignored on import
            q_entries.push(f32_entry(&format!("model.layers.{layer}.self_attn.k_proj.k_scale"), &[], &[0.5]));
            q_entries.push(f32_entry(&format!("model.layers.{layer}.self_attn.v_proj.v_scale"), &[], &[0.5]));
        }
    }
    write_support_files(quant_dir, Some(quantization_config(kind)));
    write_safetensors(&quant_dir.join("model.safetensors"), &q_entries);
    write_support_files(ref_dir, None);
    write_safetensors(&ref_dir.join("model.safetensors"), &r_entries);
}

pub(crate) fn convert(dir: &Path, dtype: OutputDType) -> PathBuf {
    let out = dir.join(format!("out-{dtype:?}.gguf"));
    convert_checkpoint(dir, &out, ConvertOptions { output_dtype: dtype, overwrite: false, spec_head: None })
        .unwrap_or_else(|e| panic!("conversion of {} failed: {e}", dir.display()));
    out
}

fn logits(gguf: &Path, tokens: &[i64]) -> Vec<f32> {
    let model = GgufModel::load(gguf.to_str().unwrap(), 64, 1, 2).unwrap();
    let decoder = RustDecoder::load(&model, "cpu").unwrap();
    let mut key = vec![0u16; model.key_buffer_len().unwrap()];
    let mut value = vec![0u16; model.value_buffer_len().unwrap()];
    let stride = model.table_stride();
    let mut tables = vec![0i64; tokens.len() * stride];
    for r in 0..tokens.len() {
        for b in 0..stride {
            tables[r * stride + b] = b as i64;
        }
    }
    let lengths: Vec<i64> = (1..=tokens.len() as i64).collect();
    let rows: Vec<usize> = (0..tokens.len()).collect();
    decoder.forward(tokens, &lengths, &tables, &mut key, &mut value, &rows, false).unwrap()
}

pub(crate) fn tensor_types(gguf: &Path) -> std::collections::BTreeMap<String, u32> {
    let m = MappedGguf::open(gguf).unwrap();
    m.index().tensors.iter().map(|t| (t.name.clone(), t.type_id)).collect()
}

fn assert_close(a: &[f32], b: &[f32], tolerance: f32, what: &str) {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let worst = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale;
    eprintln!("{what}: largest relative logit difference {worst:.2e}");
    assert!(worst <= tolerance, "{what}: largest logit difference {worst} (relative to {scale}) exceeds {tolerance}");
    assert!(b.iter().any(|v| v.abs() > 1e-3), "{what}: reference logits are all zero");
}

const PROMPT: [i64; 6] = [1, 5, 9, 3, 17, 40];

/// Convert both models and compare the logits of the engine on the CPU.
fn check(kind: Kind, seed: u64, preserve_types: &[(&str, u32)], tolerance: f32) {
    let quant = scratch("q");
    let reference = scratch("r");
    build_pair(kind, seed, &quant, &reference);
    let q_gguf = convert(&quant, OutputDType::Preserve);
    let r_gguf = convert(&reference, OutputDType::Preserve);
    let types = tensor_types(&q_gguf);
    for (name, id) in preserve_types {
        assert_eq!(types.get(*name), Some(id), "{kind:?}: type of {name}");
    }
    let got = logits(&q_gguf, &PROMPT);
    let want = logits(&r_gguf, &PROMPT);
    assert_close(&got, &want, tolerance, &format!("{kind:?}"));
    // the expanded conversion (`--outtype f16`) must agree as well
    let expanded = convert(&quant, OutputDType::F16);
    let expanded_types = tensor_types(&expanded);
    assert_eq!(expanded_types.get("blk.0.attn_q.weight"), Some(&1), "{kind:?}: f16 expansion");
    assert_close(&logits(&expanded, &PROMPT), &want, tolerance.max(2e-2), &format!("{kind:?} as f16"));
    let _ = fs::remove_dir_all(quant);
    let _ = fs::remove_dir_all(reference);
}

#[test]
fn modelopt_nvfp4_checkpoint_runs_like_its_dequantized_reference() {
    check(
        Kind::Nvfp4,
        0x9e37_79b9_7f4a_7c15,
        &[
            ("blk.0.attn_q.weight", 40),
            ("blk.1.ffn_down.weight", 40),
            ("blk.0.attn_q.scale", 0),
            ("blk.0.attn_q.input_scale", 0),
            ("output.weight", 30),
            ("token_embd.weight", 30),
        ],
        1e-4,
    );
}

#[test]
fn nvfp4_import_keeps_scale_tensors_and_drops_kv_scales() {
    let quant = scratch("names");
    let reference = scratch("names-ref");
    build_pair(Kind::Nvfp4, 77, &quant, &reference);
    let types = tensor_types(&convert(&quant, OutputDType::Preserve));
    for layer in 0..LAYERS {
        for t in ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"] {
            for suffix in ["weight", "scale", "input_scale"] {
                assert!(types.contains_key(&format!("blk.{layer}.{t}.{suffix}")), "blk.{layer}.{t}.{suffix}");
            }
        }
    }
    assert!(!types.keys().any(|n| n.contains("k_scale") || n.contains("v_scale")), "{types:?}");
    let _ = fs::remove_dir_all(quant);
    let _ = fs::remove_dir_all(reference);
}

#[test]
fn hf_quant_config_json_beside_a_plain_config_selects_the_modelopt_importer() {
    let quant = scratch("hfq");
    let reference = scratch("hfq-ref");
    build_pair(Kind::Nvfp4, 5, &quant, &reference);
    // move the quantization description from config.json to hf_quant_config.json
    let mut cfg: Value = serde_json::from_slice(&fs::read(quant.join("config.json")).unwrap()).unwrap();
    let q = cfg.as_object_mut().unwrap().remove("quantization_config").unwrap();
    fs::write(quant.join("config.json"), cfg.to_string()).unwrap();
    fs::write(
        quant.join("hf_quant_config.json"),
        json!({"producer": {"name": "modelopt", "version": "0.33"}, "quantization": {"quant_algo": q["quant_algo"], "group_size": 16,
            "kv_cache_quant_algo": null, "exclude_modules": ["lm_head"]}})
        .to_string(),
    )
    .unwrap();
    let types = tensor_types(&convert(&quant, OutputDType::Preserve));
    assert_eq!(types.get("blk.0.ffn_up.weight"), Some(&40));
    let _ = fs::remove_dir_all(quant);
    let _ = fs::remove_dir_all(reference);
}

#[test]
fn compressed_tensors_int4_groups_become_q4z_and_match_the_reference() {
    check(Kind::CtInt4(32), 11, &[("blk.0.attn_q.weight", 0x8000_0010), ("blk.1.ffn_down.weight", 0x8000_0010), ("output.weight", 30)], 5e-3);
}

#[test]
fn compressed_tensors_int8_become_q8_0_and_match_the_reference() {
    for kind in [Kind::CtInt8, Kind::CtPacked8] {
        // Q8_0 weights meet 8-bit activations on the CPU: that adds the usual ~1 % activation rounding
        check(kind, 13, &[("blk.0.attn_q.weight", 8), ("blk.1.ffn_down.weight", 8)], 3e-2);
    }
}

#[test]
fn block_wise_fp8_expands_to_f16_and_matches_the_reference() {
    check(Kind::Fp8Block(32), 17, &[("blk.0.attn_q.weight", 1), ("blk.1.ffn_down.weight", 1)], 5e-3);
}

#[test]
fn unsupported_quantization_configs_fail_with_actionable_messages() {
    let dir = scratch("unsupported");
    let reference = scratch("unsupported-ref");
    build_pair(Kind::Nvfp4, 3, &dir, &reference);
    let cases = [
        (json!({"quant_method": "modelopt", "quant_algo": "W4A8_AWQ"}), "unsupported ModelOpt quant_algo"),
        (json!({"quant_method": "modelopt", "quant_algo": "NVFP4", "group_size": 32}), "group_size 32"),
        (json!({"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized", "config_groups":
            {"g": {"weights": {"num_bits": 4, "type": "float", "strategy": "tensor_group", "group_size": 8}}}}), "group_size"),
        (json!({"quant_method": "compressed-tensors", "format": "pack-quantized", "config_groups":
            {"g": {"weights": {"num_bits": 4, "type": "int", "symmetric": false, "strategy": "group", "group_size": 32}}}}), "asymmetric"),
        (json!({"quant_method": "compressed-tensors", "format": "pack-quantized", "config_groups":
            {"g": {"weights": {"num_bits": 3, "type": "int", "symmetric": true, "strategy": "group", "group_size": 32}}}}), "3-bit"),
        (json!({"quant_method": "some_new_method"}), "unsupported quantization_config quant_method"),
        (json!({"quant_method": "bitsandbytes"}), "neither load_in_4bit nor load_in_8bit"),
    ];
    for (q, expect) in cases {
        let mut cfg: Value = serde_json::from_slice(&fs::read(dir.join("config.json")).unwrap()).unwrap();
        cfg["quantization_config"] = q;
        fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
        let err = convert_checkpoint(&dir, &dir.join("never.gguf"), ConvertOptions::default()).unwrap_err().to_string();
        assert!(err.contains(expect), "{expect:?} not in {err:?}");
        assert!(!dir.join("never.gguf").exists());
    }
    let _ = fs::remove_dir_all(dir);
    let _ = fs::remove_dir_all(reference);
}

#[test]
fn the_comparison_detects_a_wrong_model() {
    // control: a quantized model against the reference of another random model must be far off
    let (quant, reference, other) = (scratch("c-q"), scratch("c-r"), scratch("c-o"));
    let other_quant = scratch("c-oq");
    build_pair(Kind::Nvfp4, 1, &quant, &reference);
    build_pair(Kind::Nvfp4, 2, &other_quant, &other);
    let got = logits(&convert(&quant, OutputDType::Preserve), &PROMPT);
    let wrong = logits(&convert(&other, OutputDType::Preserve), &PROMPT);
    let scale = wrong.iter().fold(0f32, |m, v| m.max(v.abs()));
    let worst = got.iter().zip(&wrong).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale;
    assert!(worst > 0.2, "different models produce nearly equal logits ({worst})");
    for d in [quant, reference, other, other_quant] {
        let _ = fs::remove_dir_all(d);
    }
}

#[test]
fn direct_safetensors_loader_zero_copy_and_logit_parity() {
    let quant = scratch("direct-q");
    let reference = scratch("direct-r");
    build_pair(Kind::Nvfp4, 42, &quant, &reference);

    // 1. Direct model load without converting to GGUF file
    let direct_model = GgufModel::load(reference.to_str().unwrap(), 64, 1, 2).unwrap();
    let direct_mapping = direct_model.mapping();

    // 2. Reference model converted to GGUF file
    let r_gguf = convert(&reference, OutputDType::Preserve);
    let r_mapped = MappedGguf::open(&r_gguf).unwrap();

    // 3. Verify zero-copy: untouched tensors (like token_embd.weight) are mapped directly from safetensors
    assert!(direct_mapping.extra_mappings().contains_key("token_embd.weight"), "token_embd.weight must be in extra_mappings");
    let embd_bytes = direct_mapping.tensor_bytes("token_embd.weight").unwrap();
    let mmap = direct_mapping.extra_mappings().get("token_embd.weight").unwrap();
    let mmap_start = mmap.as_ptr() as usize;
    let mmap_end = mmap_start + mmap.len();
    let embd_ptr = embd_bytes.as_ptr() as usize;
    assert!(
        embd_ptr >= mmap_start && embd_ptr + embd_bytes.len() <= mmap_end,
        "token_embd.weight ({:#x}) must reside inside safetensors mmap ({:#x}..{:#x})",
        embd_ptr,
        mmap_start,
        mmap_end
    );

    // 4. Verify bit-for-bit parity of every tensor in the model
    assert_eq!(direct_mapping.index().tensors.len(), r_mapped.index().tensors.len());
    for t in direct_mapping.index().tensors.iter() {
        let direct_b = direct_mapping.tensor_bytes(&t.name).unwrap();
        let ref_b = r_mapped.tensor_bytes(&t.name).unwrap();
        assert_eq!(
            direct_b,
            ref_b,
            "tensor byte mismatch for tensor {}",
            t.name
        );
    }

    // 5. Verify exact logit parity from forward pass
    let direct_logits = logits(&reference, &PROMPT);
    let gguf_logits = logits(&r_gguf, &PROMPT);
    assert_close(&direct_logits, &gguf_logits, 1e-6, "direct vs gguf forward pass");
    assert_eq!(direct_logits, gguf_logits, "forward pass logits must match bit-for-bit");

    for d in [quant, reference] {
        let _ = fs::remove_dir_all(d);
    }
}

/// Writes the tiny NVFP4 model and its dequantized reference to `$FLLM_FIXTURE_OUT/{nvfp4,reference}` for trying the
/// command line by hand: `FLLM_FIXTURE_OUT=/tmp/fx cargo test -p fllm-engine --lib write_fixture -- --ignored`.
#[test]
#[ignore]
fn write_fixture_for_manual_runs() {
    let out = PathBuf::from(std::env::var_os("FLLM_FIXTURE_OUT").expect("set FLLM_FIXTURE_OUT"));
    let (quant, reference) = (out.join("nvfp4"), out.join("reference"));
    fs::create_dir_all(&quant).unwrap();
    fs::create_dir_all(&reference).unwrap();
    build_pair(Kind::Nvfp4, 1234, &quant, &reference);
}

// ---- mixture-of-experts checkpoints: the experts of a layer stay in their compact format as one stacked tensor ----

const EXPERTS: usize = 4;
const EXPERTS_USED: usize = 2;

/// The tiny Mixtral-style model: the attention linears and the experts (`w1` gate, `w3` up, `w2` down) in `kind`, the router in BF16.
/// The reference has the same values dequantized to F32.
fn build_moe_pair(kind: Kind, seed: u64, quant_dir: &Path, ref_dir: &Path) {
    let mut rng = Rng(seed);
    let mut q_entries = Vec::new();
    let mut r_entries = Vec::new();
    let both = |e: Entry, q: &mut Vec<Entry>, r: &mut Vec<Entry>| {
        r.push(Entry { name: e.name.clone(), dtype: e.dtype, shape: e.shape.clone(), bytes: e.bytes.clone() });
        q.push(e);
    };
    let embed: Vec<f32> = (0..VOCAB * HIDDEN).map(|_| rng.unit()).collect();
    both(bf16_entry("model.embed_tokens.weight", &[VOCAB, HIDDEN], &embed), &mut q_entries, &mut r_entries);
    let head: Vec<f32> = (0..VOCAB * HIDDEN).map(|_| 0.3 * rng.unit()).collect();
    both(bf16_entry("lm_head.weight", &[VOCAB, HIDDEN], &head), &mut q_entries, &mut r_entries);
    let norm: Vec<f32> = (0..HIDDEN).map(|_| 1.0 + 0.1 * rng.unit()).collect();
    both(bf16_entry("model.norm.weight", &[HIDDEN], &norm), &mut q_entries, &mut r_entries);
    for layer in 0..LAYERS {
        for n in ["input_layernorm", "post_attention_layernorm"] {
            let w: Vec<f32> = (0..HIDDEN).map(|_| 1.0 + 0.1 * rng.unit()).collect();
            both(bf16_entry(&format!("model.layers.{layer}.{n}.weight"), &[HIDDEN], &w), &mut q_entries, &mut r_entries);
        }
        let router: Vec<f32> = (0..EXPERTS * HIDDEN).map(|_| rng.unit()).collect();
        both(bf16_entry(&format!("model.layers.{layer}.block_sparse_moe.gate.weight"), &[EXPERTS, HIDDEN], &router), &mut q_entries, &mut r_entries);
        let mut linears = vec![
            (format!("model.layers.{layer}.self_attn.q_proj"), HEADS * HEAD_DIM, HIDDEN),
            (format!("model.layers.{layer}.self_attn.k_proj"), KV_HEADS * HEAD_DIM, HIDDEN),
            (format!("model.layers.{layer}.self_attn.v_proj"), KV_HEADS * HEAD_DIM, HIDDEN),
            (format!("model.layers.{layer}.self_attn.o_proj"), HIDDEN, HEADS * HEAD_DIM),
        ];
        for e in 0..EXPERTS {
            let base = format!("model.layers.{layer}.block_sparse_moe.experts.{e}");
            linears.push((format!("{base}.w1"), INTER, HIDDEN));
            linears.push((format!("{base}.w3"), INTER, HIDDEN));
            linears.push((format!("{base}.w2"), HIDDEN, INTER));
        }
        for (prefix, out, inp) in linears {
            let (entries, dense) = quantized_linear(&mut rng, kind, &prefix, out, inp);
            q_entries.extend(entries);
            r_entries.push(f32_entry(&format!("{prefix}.weight"), &[out, inp], &dense));
        }
        if let Kind::Nvfp4 = kind {
            q_entries.push(f32_entry(&format!("model.layers.{layer}.self_attn.k_proj.k_scale"), &[], &[0.5]));
        }
    }
    for (dir, quant) in [(quant_dir, Some(quantization_config(kind))), (ref_dir, None)] {
        write_support_files(dir, quant);
        let mut cfg: Value = serde_json::from_slice(&fs::read(dir.join("config.json")).unwrap()).unwrap();
        cfg["model_type"] = json!("mixtral");
        cfg["architectures"] = json!(["MixtralForCausalLM"]);
        cfg["num_local_experts"] = json!(EXPERTS);
        cfg["num_experts_per_tok"] = json!(EXPERTS_USED);
        fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
    }
    write_safetensors(&quant_dir.join("model.safetensors"), &q_entries);
    write_safetensors(&ref_dir.join("model.safetensors"), &r_entries);
}

fn tensor_dims(gguf: &Path) -> std::collections::BTreeMap<String, Vec<u64>> {
    let m = MappedGguf::open(gguf).unwrap();
    m.index().tensors.iter().map(|t| (t.name.clone(), t.dimensions.clone())).collect()
}

/// Convert the quantized and the reference MoE model, check the compact types of the stacked experts and compare the logits.
fn check_moe(kind: Kind, seed: u64, expert_type: u32, tolerance: f32) -> std::collections::BTreeMap<String, Vec<u64>> {
    let quant = scratch("moe-q");
    let reference = scratch("moe-r");
    build_moe_pair(kind, seed, &quant, &reference);
    let q_gguf = convert(&quant, OutputDType::Preserve);
    let r_gguf = convert(&reference, OutputDType::Preserve);
    let types = tensor_types(&q_gguf);
    for t in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
        for layer in 0..LAYERS {
            assert_eq!(types.get(&format!("blk.{layer}.{t}.weight")), Some(&expert_type), "{kind:?}: type of blk.{layer}.{t}");
        }
    }
    let got = logits(&q_gguf, &PROMPT);
    let want = logits(&r_gguf, &PROMPT);
    assert_close(&got, &want, tolerance, &format!("MoE {kind:?}"));
    // the expanded conversion (`--outtype f16`) agrees as well
    let expanded = convert(&quant, OutputDType::F16);
    assert_eq!(tensor_types(&expanded).get("blk.0.ffn_gate_exps.weight"), Some(&1), "{kind:?}: f16 expansion");
    assert_close(&logits(&expanded, &PROMPT), &want, tolerance.max(2e-2), &format!("MoE {kind:?} as f16"));
    let dims = tensor_dims(&q_gguf);
    let _ = fs::remove_dir_all(quant);
    let _ = fs::remove_dir_all(reference);
    dims
}

#[test]
fn nvfp4_experts_stay_4_bit_with_one_scale_per_expert() {
    let dims = check_moe(Kind::Nvfp4, 21, 40, 1e-4);
    // stacked: [inputs, rows, experts]; the per-tensor factor and the activation scale come once per expert
    assert_eq!(dims["blk.0.ffn_gate_exps.weight"], vec![HIDDEN as u64, INTER as u64, EXPERTS as u64]);
    assert_eq!(dims["blk.0.ffn_down_exps.weight"], vec![INTER as u64, HIDDEN as u64, EXPERTS as u64]);
    for t in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
        assert_eq!(dims[&format!("blk.1.{t}.scale")], vec![EXPERTS as u64], "{t}.scale");
        assert_eq!(dims[&format!("blk.1.{t}.input_scale")], vec![EXPERTS as u64], "{t}.input_scale");
    }
}

#[test]
fn int4_experts_stay_q4z() {
    let dims = check_moe(Kind::CtInt4(32), 23, 0x8000_0010, 5e-3);
    assert_eq!(dims["blk.0.ffn_up_exps.weight"], vec![HIDDEN as u64, INTER as u64, EXPERTS as u64]);
}

#[test]
fn block_wise_fp8_experts_convert_to_f16() {
    // no compact device format for the 128x128 tiles yet (docs/moe-formats.md): the experts are expanded, and must still be right
    check_moe(Kind::Fp8Block(32), 29, 1, 5e-3);
}

// ---- direct loading: `open_model` of a checkpoint equals the converted GGUF file ----

/// Every tensor (name, dimensions, type, bytes) and every metadata value of the two models is equal.
pub(crate) fn assert_same_model(direct: &MappedGguf, file: &MappedGguf, what: &str) {
    assert_eq!(direct.index().metadata, file.index().metadata, "{what}: metadata");
    assert_eq!(direct.index().tensors.len(), file.index().tensors.len(), "{what}: tensor count");
    for t in &direct.index().tensors {
        let r = file.index().tensors.iter().find(|r| r.name == t.name).unwrap_or_else(|| panic!("{what}: {} missing", t.name));
        assert_eq!((&t.dimensions, t.type_id, t.payload_len), (&r.dimensions, r.type_id, r.payload_len), "{what}: layout of {}", t.name);
        assert!(direct.tensor_bytes(&t.name).unwrap() == file.tensor_bytes(&t.name).unwrap(), "{what}: bytes of {}", t.name);
    }
}

#[test]
fn direct_load_equals_the_converted_file_for_every_quantized_layout() {
    for (kind, moe) in [
        (Kind::Nvfp4, false),
        (Kind::CtInt4(32), false),
        (Kind::CtInt8, false),
        (Kind::CtPacked8, false),
        (Kind::Fp8Block(32), false),
        (Kind::Nvfp4, true),
        (Kind::CtInt4(32), true),
        (Kind::Fp8Block(32), true),
    ] {
        let (quant, reference) = (scratch("dq"), scratch("dr"));
        if moe {
            build_moe_pair(kind, 9, &quant, &reference);
        } else {
            build_pair(kind, 9, &quant, &reference);
        }
        for dtype in [OutputDType::Preserve, OutputDType::F16] {
            let gguf = convert(&quant, dtype);
            let direct = crate::import::open_model(&quant).unwrap_or_else(|e| panic!("{kind:?}: {e}"));
            if matches!(dtype, OutputDType::Preserve) {
                assert_same_model(&direct, &MappedGguf::open(&gguf).unwrap(), &format!("{kind:?} moe={moe}"));
                // the engine computes the same logits from both
                assert_eq!(logits(&quant, &PROMPT), logits(&gguf, &PROMPT), "{kind:?} moe={moe}: logits");
            }
        }
        for d in [quant, reference] {
            let _ = fs::remove_dir_all(d);
        }
    }
}

#[test]
fn direct_load_of_unquantized_checkpoints_gives_the_same_logits_as_the_converted_file() {
    let (quant, reference) = (scratch("du-q"), scratch("du-r"));
    build_pair(Kind::Nvfp4, 13, &quant, &reference);
    for dtype in [OutputDType::Preserve, OutputDType::F16, OutputDType::F32] {
        let gguf = convert(&reference, dtype);
        // `open_model` on a bare directory loads with the default type (preserve); other types go through the checkpoint API
        let options = ConvertOptions { output_dtype: dtype, overwrite: false, spec_head: None };
        let direct = fllm_checkpoint::load_checkpoint_as_gguf(&reference, options).unwrap();
        assert_same_model(&direct, &MappedGguf::open(&gguf).unwrap(), &format!("{dtype:?}"));
        if matches!(dtype, OutputDType::Preserve) {
            assert_eq!(logits(&reference, &PROMPT), logits(&gguf, &PROMPT));
        }
    }
    for d in [quant, reference] {
        let _ = fs::remove_dir_all(d);
    }
}

// ---- the `fllm-rs serve` command itself ----

/// The release or debug binary next to the test executable, when it has been built (`cargo build --bin fllm-rs`).
fn built_binary() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?.parent()?;
    let bin = dir.join(if cfg!(windows) { "fllm-rs.exe" } else { "fllm-rs" });
    bin.is_file().then_some(bin)
}

fn http(port: u16, method: &str, path: &str, body: &str) -> Option<String> {
    use std::io::{Read, Write};
    let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream.set_read_timeout(Some(std::time::Duration::from_secs(30))).ok()?;
    let request = format!("{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len());
    stream.write_all(request.as_bytes()).ok()?;
    let mut response = String::new();
    stream.read_to_string(&mut response).ok()?;
    Some(response)
}

/// Starts `fllm-rs serve --model MODEL ...` on the CPU, asks for a greedy completion and returns (text, model ids, log).
fn serve_once(bin: &Path, cache: &Path, model: &Path, extra: &[&str]) -> (String, String, String) {
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let log_path = cache.join(format!("serve-{port}.log"));
    let log = fs::File::create(&log_path).unwrap();
    let mut child = std::process::Command::new(bin)
        .args(["serve", "--device", "cpu", "--ctx", "64", "--max-seqs", "2", "--host", "127.0.0.1", "--port", &port.to_string(), "--model"])
        .arg(model)
        .args(extra)
        .env("FLLM_CACHE", cache)
        .env("HF_HOME", cache)
        .env_remove("FLLM_NO_DIRECT_LOAD")
        .env_remove("FLLM_DIRECT_SAFETENSORS")
        .stdout(log.try_clone().unwrap())
        .stderr(log)
        .spawn()
        .unwrap();
    let mut ready = false;
    for _ in 0..120 {
        if child.try_wait().unwrap().is_some() {
            break;
        }
        if http(port, "GET", "/health", "").is_some_and(|r| r.starts_with("HTTP/1.1 200")) {
            ready = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    let logs = |path: &Path| fs::read_to_string(path).unwrap_or_default();
    if !ready {
        let _ = child.kill();
        panic!("the server did not start for {} {extra:?}:\n{}", model.display(), logs(&log_path));
    }
    let completion = http(port, "POST", "/v1/completions", r#"{"model":"x","prompt":"t5 t9 t3 t17","max_tokens":10,"temperature":0}"#).unwrap_or_default();
    let models = http(port, "GET", "/v1/models", "").unwrap_or_default();
    let _ = child.kill();
    let _ = child.wait();
    let text = completion.split("\"text\":\"").nth(1).and_then(|t| t.split('"').next()).unwrap_or_else(|| panic!("no completion in {completion}\n{}", logs(&log_path))).to_owned();
    (text, models, logs(&log_path))
}

#[test]
fn the_serve_command_loads_checkpoint_directories_directly_and_answers_like_the_converted_file() {
    let Some(bin) = built_binary() else {
        eprintln!("skipped: build the fllm-rs binary first (cargo build --bin fllm-rs)");
        return;
    };
    let (quant, reference) = (scratch("srv-q"), scratch("srv-r"));
    build_pair(Kind::Nvfp4, 71, &quant, &reference);
    let cache = scratch("srv-cache");
    let cache_files = |dir: &Path| fs::read_dir(dir.join("fllm-local")).map(|d| d.flatten().count()).unwrap_or(0);
    for (model, extra) in [(&quant, vec![]), (&reference, vec![]), (&reference, vec!["--quant", "q8_0"])] {
        let (direct_text, models, log) = serve_once(&bin, &cache, model, &extra);
        assert!(log.contains("directly"), "no direct load for {}:\n{log}", model.display());
        assert_eq!(cache_files(&cache), 0, "a GGUF file was written for {}", model.display());
        let name = model.file_name().unwrap().to_string_lossy().into_owned();
        assert!(models.contains(&name), "model id should be the directory name {name}: {models}");
        let mut converted = extra.clone();
        converted.push("--no-direct-load");
        let (converted_text, _, log) = serve_once(&bin, &cache, model, &converted);
        assert!(cache_files(&cache) > 0, "--no-direct-load must write the converted file:\n{log}");
        assert!(!direct_text.is_empty());
        assert_eq!(direct_text, converted_text, "{} {extra:?}", model.display());
        let _ = fs::remove_dir_all(cache.join("fllm-local"));
    }
    for d in [quant, reference, cache] {
        let _ = fs::remove_dir_all(d);
    }
}
