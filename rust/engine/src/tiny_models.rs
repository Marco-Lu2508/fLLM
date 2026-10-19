//! Tiny random models for whole-model tests of the device backends, CPU only to build.
//!
//! A model is written as an F32 safetensors checkpoint in the layout of the Hugging Face architecture (Llama, Qwen3, Mixtral,
//! Qwen3-MoE, Qwen3.5), converted to GGUF with `fllm_checkpoint` (optionally re-encoded with a quantization preset) and loaded by
//! the engine. [`compare_with_host`] then runs the same prompt and the same greedy continuation on the host path of the decoder and
//! on a device backend and compares the logits of every step.
//!
//! The host path (`RustDecoder` without a backend) is the reference; nothing in this file depends on a GPU, so the fixtures and the
//! comparison itself are exercised on every platform with the CPU backend standing in for the device
//! (`cpu_backend_matches_host_path`). The Metal tests (`backend/metal/whole_model_tests.rs`) call the same functions with the Metal
//! backend.

use crate::{backend::Backend, rust_decoder::RustDecoder, GgufModel};
use fllm_checkpoint::{convert_checkpoint, quantize_gguf, ConvertOptions, OutputDType, Preset};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

pub(crate) const HIDDEN: usize = 256;
pub(crate) const HEADS: usize = 4;
pub(crate) const KV_HEADS: usize = 2;
pub(crate) const HEAD_DIM: usize = 64;
pub(crate) const INTER: usize = 512;
pub(crate) const LAYERS: usize = 2;
pub(crate) const VOCAB: usize = 128;
pub(crate) const EXPERTS: usize = 4;
pub(crate) const TOP_K: usize = 2;
pub(crate) const MOE_INTER: usize = 256;

/// Architecture of a tiny model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Arch {
    Llama,
    Qwen3,
    Mixtral,
    Qwen3Moe,
    /// Qwen3.5 style hybrid: layer 0 gated DeltaNet, layer 1 full attention with an output gate, dense feed-forward
    Qwen35,
    K2Horizon,
    Deepseek,
    Exaone,
    ChatGLM,
}

// Gated DeltaNet geometry of the Qwen35 fixture
const LIN_K_HEADS: usize = 2;
const LIN_V_HEADS: usize = 4;
const LIN_K_DIM: usize = 32;
const LIN_V_DIM: usize = 32;
const LIN_CONV: usize = 4;

pub(crate) fn scratch(label: &str) -> PathBuf {
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!("fllm-tiny-{label}-{}-{n}", std::process::id()));
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
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

struct Entry {
    name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
}

/// A linear layer `out x inp` with weights of standard deviation about `gain / sqrt(inp)`.
fn linear(rng: &mut Rng, name: &str, out: usize, inp: usize, gain: f32) -> Entry {
    let amp = gain * 3f32.sqrt() / (inp as f32).sqrt();
    Entry { name: name.into(), shape: vec![out, inp], values: (0..out * inp).map(|_| amp * rng.unit()).collect() }
}

/// A norm weight around 1.
fn norm(rng: &mut Rng, name: &str, n: usize) -> Entry {
    Entry { name: name.into(), shape: vec![n], values: (0..n).map(|_| 1.0 + 0.1 * rng.unit()).collect() }
}

/// A norm weight around 0 (Qwen3.5 stores `w` for `x * (1 + w)`).
fn zero_norm(rng: &mut Rng, name: &str, n: usize) -> Entry {
    Entry { name: name.into(), shape: vec![n], values: (0..n).map(|_| 0.1 * rng.unit()).collect() }
}

/// One layer of the Qwen3.5 fixture: a gated DeltaNet mixer (`linear`) or full attention with a query gate, and a dense MLP.
fn qwen35_layer(rng: &mut Rng, entries: &mut Vec<Entry>, p: &str, linear_layer: bool) {
    entries.push(zero_norm(rng, &format!("{p}.input_layernorm.weight"), HIDDEN));
    entries.push(zero_norm(rng, &format!("{p}.post_attention_layernorm.weight"), HIDDEN));
    if linear_layer {
        let key_total = LIN_K_HEADS * LIN_K_DIM;
        let value_total = LIN_V_HEADS * LIN_V_DIM;
        let conv_dim = 2 * key_total + value_total;
        let a = format!("{p}.linear_attn");
        entries.push(linear(rng, &format!("{a}.in_proj_qkv.weight"), conv_dim, HIDDEN, 1.0));
        entries.push(linear(rng, &format!("{a}.in_proj_z.weight"), value_total, HIDDEN, 1.0));
        entries.push(linear(rng, &format!("{a}.in_proj_b.weight"), LIN_V_HEADS, HIDDEN, 1.0));
        entries.push(linear(rng, &format!("{a}.in_proj_a.weight"), LIN_V_HEADS, HIDDEN, 1.0));
        let conv_amp = 1.0 / (LIN_CONV as f32).sqrt();
        entries.push(Entry {
            name: format!("{a}.conv1d.weight"),
            shape: vec![conv_dim, 1, LIN_CONV],
            values: (0..conv_dim * LIN_CONV).map(|_| conv_amp * rng.unit()).collect(),
        });
        entries.push(Entry { name: format!("{a}.dt_bias"), shape: vec![LIN_V_HEADS], values: (0..LIN_V_HEADS).map(|_| 0.5 * rng.unit()).collect() });
        entries.push(Entry {
            name: format!("{a}.A_log"),
            shape: vec![LIN_V_HEADS],
            values: (0..LIN_V_HEADS).map(|_| 0.5 * (rng.unit() + 1.0) - 1.0).collect(),
        });
        entries.push(norm(rng, &format!("{a}.norm.weight"), LIN_V_DIM));
        entries.push(linear(rng, &format!("{a}.out_proj.weight"), HIDDEN, value_total, 1.0));
    } else {
        entries.push(linear(rng, &format!("{p}.self_attn.q_proj.weight"), 2 * HEADS * HEAD_DIM, HIDDEN, 1.0));
        entries.push(linear(rng, &format!("{p}.self_attn.k_proj.weight"), KV_HEADS * HEAD_DIM, HIDDEN, 1.0));
        entries.push(linear(rng, &format!("{p}.self_attn.v_proj.weight"), KV_HEADS * HEAD_DIM, HIDDEN, 1.0));
        entries.push(linear(rng, &format!("{p}.self_attn.o_proj.weight"), HIDDEN, HEADS * HEAD_DIM, 1.0));
        entries.push(zero_norm(rng, &format!("{p}.self_attn.q_norm.weight"), HEAD_DIM));
        entries.push(zero_norm(rng, &format!("{p}.self_attn.k_norm.weight"), HEAD_DIM));
    }
    entries.push(linear(rng, &format!("{p}.mlp.gate_proj.weight"), INTER, HIDDEN, 1.0));
    entries.push(linear(rng, &format!("{p}.mlp.up_proj.weight"), INTER, HIDDEN, 1.0));
    entries.push(linear(rng, &format!("{p}.mlp.down_proj.weight"), HIDDEN, INTER, 1.0));
}

fn write_safetensors(path: &Path, entries: &[Entry]) {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for e in entries {
        let bytes = e.values.len() * 4;
        header.insert(e.name.clone(), json!({"dtype": "F32", "shape": e.shape, "data_offsets": [offset, offset + bytes]}));
        offset += bytes;
    }
    let header = Value::Object(header).to_string();
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    for e in entries {
        out.extend(e.values.iter().flat_map(|v| v.to_le_bytes()));
    }
    fs::write(path, out).unwrap();
}

fn config_json(arch: Arch) -> Value {
    let mut cfg = json!({
        "hidden_size": HIDDEN, "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS, "num_hidden_layers": LAYERS,
        "intermediate_size": INTER, "vocab_size": VOCAB, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "max_position_embeddings": 128, "hidden_act": "silu", "tie_word_embeddings": false,
        "bos_token_id": 1, "eos_token_id": 2, "torch_dtype": "float32"
    });
    match arch {
        Arch::Llama => {
            cfg["model_type"] = "llama".into();
            cfg["architectures"] = json!(["LlamaForCausalLM"]);
        }
        Arch::Qwen3 => {
            cfg["model_type"] = "qwen3".into();
            cfg["architectures"] = json!(["Qwen3ForCausalLM"]);
            cfg["head_dim"] = HEAD_DIM.into();
        }
        Arch::Mixtral => {
            cfg["model_type"] = "mixtral".into();
            cfg["architectures"] = json!(["MixtralForCausalLM"]);
            cfg["num_local_experts"] = EXPERTS.into();
            cfg["num_experts_per_tok"] = TOP_K.into();
        }
        Arch::Qwen35 => {
            cfg["model_type"] = "qwen3_5".into();
            cfg["architectures"] = json!(["Qwen3_5ForCausalLM"]);
            cfg["head_dim"] = HEAD_DIM.into();
            cfg["layer_types"] = json!(["linear_attention", "full_attention"]);
            cfg["full_attention_interval"] = 2.into();
            cfg["attn_output_gate"] = true.into();
            cfg["partial_rotary_factor"] = 0.25.into();
            cfg["linear_num_key_heads"] = LIN_K_HEADS.into();
            cfg["linear_num_value_heads"] = LIN_V_HEADS.into();
            cfg["linear_key_head_dim"] = LIN_K_DIM.into();
            cfg["linear_value_head_dim"] = LIN_V_DIM.into();
            cfg["linear_conv_kernel_dim"] = LIN_CONV.into();
        }
        Arch::Qwen3Moe => {
            cfg["model_type"] = "qwen3_moe".into();
            cfg["architectures"] = json!(["Qwen3MoeForCausalLM"]);
            cfg["head_dim"] = HEAD_DIM.into();
            cfg["num_experts"] = EXPERTS.into();
            cfg["num_experts_per_tok"] = TOP_K.into();
            cfg["moe_intermediate_size"] = MOE_INTER.into();
            cfg["norm_topk_prob"] = true.into();
            cfg["decoder_sparse_step"] = 1.into();
            cfg["mlp_only_layers"] = json!([]);
        }
        Arch::K2Horizon => {
            cfg["model_type"] = "k2_horizon".into();
            cfg["architectures"] = json!(["K2HorizonForCausalLM"]);
            cfg["layernorm_num_groups"] = 2.into();
            cfg["query_key_norm"] = false.into();
        }
        Arch::Deepseek => {
            cfg["model_type"] = "deepseek".into();
            cfg["architectures"] = json!(["DeepseekForCausalLM"]);
            cfg["n_routed_experts"] = EXPERTS.into();
            cfg["num_experts_per_tok"] = TOP_K.into();
            cfg["n_shared_experts"] = 1.into();
            cfg["moe_intermediate_size"] = MOE_INTER.into();
            cfg["first_k_dense_replace"] = 1.into();
            cfg["scoring_func"] = "softmax".into();
            cfg["norm_topk_prob"] = false.into();
        }
        Arch::Exaone => {
            cfg["model_type"] = "exaone".into();
            cfg["architectures"] = json!(["ExaoneForCausalLM"]);
            cfg["num_layers"] = LAYERS.into();
        }
        Arch::ChatGLM => {
            cfg["model_type"] = "chatglm".into();
            cfg["architectures"] = json!(["ChatGLMForConditionalGeneration"]);
            cfg["num_layers"] = LAYERS.into();
            cfg["ffn_hidden_size"] = INTER.into();
            cfg["multi_query_attention"] = true.into();
            cfg["multi_query_group_num"] = KV_HEADS.into();
            cfg["kv_channels"] = HEAD_DIM.into();
            cfg["layernorm_epsilon"] = 1e-5.into();
            cfg["rmsnorm"] = true.into();
            cfg["add_qkv_bias"] = true.into();
            cfg["seq_length"] = 2048.into();
            cfg["original_rope"] = true.into();
        }
    }
    cfg
}

fn write_support_files(dir: &Path, cfg: &Value) {
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
    fs::write(dir.join("tokenizer_config.json"), json!({"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>"}).to_string()).unwrap();
}

/// Write the checkpoint of a tiny random model of `arch` into `dir`.
pub(crate) fn write_checkpoint(dir: &Path, arch: Arch, seed: u64) {
    let mut rng = Rng(seed | 1);
    let mut entries = Vec::new();
    let embed: Vec<f32> = (0..VOCAB * HIDDEN).map(|_| rng.unit()).collect();
    let emb_name = match arch { Arch::Exaone => "transformer.wte.weight", Arch::ChatGLM => "transformer.embedding.word_embeddings.weight", _ => "model.embed_tokens.weight" };
    entries.push(Entry { name: emb_name.into(), shape: vec![VOCAB, HIDDEN], values: embed });
    let head_name = if arch == Arch::ChatGLM { "transformer.output_layer.weight" } else { "lm_head.weight" };
    entries.push(linear(&mut rng, head_name, VOCAB, HIDDEN, 1.5));
    let norm_name = match arch { Arch::Exaone => "transformer.ln_f.weight", Arch::ChatGLM => "transformer.encoder.final_layernorm.weight", _ => "model.norm.weight" };
    entries.push(norm(&mut rng, norm_name, HIDDEN));
    let qwen_style = matches!(arch, Arch::Qwen3 | Arch::Qwen3Moe | Arch::Qwen35);
    if arch == Arch::Qwen35 {
        // zero-centred norm weights (the model computes x * (1 + w))
        entries.pop();
        entries.push(zero_norm(&mut rng, "model.norm.weight", HIDDEN));
    }
    for layer in 0..LAYERS {
        if arch == Arch::ChatGLM {
            let p = format!("transformer.encoder.layers.{layer}");
            entries.push(norm(&mut rng, &format!("{p}.input_layernorm.weight"), HIDDEN));
            entries.push(norm(&mut rng, &format!("{p}.post_attention_layernorm.weight"), HIDDEN));
            let qkv_rows = (HEADS + 2 * KV_HEADS) * HEAD_DIM;
            entries.push(linear(&mut rng, &format!("{p}.self_attention.query_key_value.weight"), qkv_rows, HIDDEN, 1.0));
            entries.push(Entry {
                name: format!("{p}.self_attention.query_key_value.bias"),
                shape: vec![qkv_rows],
                values: (0..qkv_rows).map(|_| rng.unit() * 0.01).collect(),
            });
            entries.push(linear(&mut rng, &format!("{p}.self_attention.dense.weight"), HIDDEN, HEADS * HEAD_DIM, 1.0));
            entries.push(linear(&mut rng, &format!("{p}.mlp.dense_h_to_4h.weight"), 2 * INTER, HIDDEN, 1.0));
            entries.push(linear(&mut rng, &format!("{p}.mlp.dense_4h_to_h.weight"), HIDDEN, INTER, 1.0));
            continue;
        }
        if arch == Arch::Exaone {
            let p = format!("transformer.h.{layer}");
            entries.push(norm(&mut rng, &format!("{p}.ln_1.weight"), HIDDEN));
            entries.push(norm(&mut rng, &format!("{p}.ln_2.weight"), HIDDEN));
            entries.push(linear(&mut rng, &format!("{p}.attn.attention.q_proj.weight"), HEADS * HEAD_DIM, HIDDEN, 1.0));
            entries.push(linear(&mut rng, &format!("{p}.attn.attention.k_proj.weight"), KV_HEADS * HEAD_DIM, HIDDEN, 1.0));
            entries.push(linear(&mut rng, &format!("{p}.attn.attention.v_proj.weight"), KV_HEADS * HEAD_DIM, HIDDEN, 1.0));
            entries.push(linear(&mut rng, &format!("{p}.attn.attention.out_proj.weight"), HIDDEN, HEADS * HEAD_DIM, 1.0));
            entries.push(linear(&mut rng, &format!("{p}.mlp.c_fc_0.weight"), INTER, HIDDEN, 1.0));
            entries.push(linear(&mut rng, &format!("{p}.mlp.c_fc_1.weight"), INTER, HIDDEN, 1.0));
            entries.push(linear(&mut rng, &format!("{p}.mlp.c_proj.weight"), HIDDEN, INTER, 1.0));
            continue;
        }
        let p = format!("model.layers.{layer}");
        if arch == Arch::Qwen35 {
            qwen35_layer(&mut rng, &mut entries, &p, layer == 0);
            continue;
        }
        entries.push(norm(&mut rng, &format!("{p}.input_layernorm.weight"), HIDDEN));
        entries.push(norm(&mut rng, &format!("{p}.post_attention_layernorm.weight"), HIDDEN));
        entries.push(linear(&mut rng, &format!("{p}.self_attn.q_proj.weight"), HEADS * HEAD_DIM, HIDDEN, 1.0));
        entries.push(linear(&mut rng, &format!("{p}.self_attn.k_proj.weight"), KV_HEADS * HEAD_DIM, HIDDEN, 1.0));
        entries.push(linear(&mut rng, &format!("{p}.self_attn.v_proj.weight"), KV_HEADS * HEAD_DIM, HIDDEN, 1.0));
        entries.push(linear(&mut rng, &format!("{p}.self_attn.o_proj.weight"), HIDDEN, HEADS * HEAD_DIM, 1.0));
        if qwen_style {
            entries.push(norm(&mut rng, &format!("{p}.self_attn.q_norm.weight"), HEAD_DIM));
            entries.push(norm(&mut rng, &format!("{p}.self_attn.k_norm.weight"), HEAD_DIM));
        }
        match arch {
            Arch::Qwen35 => unreachable!("handled by qwen35_layer"),
            Arch::Exaone | Arch::ChatGLM => unreachable!("handled above"),
            Arch::Llama | Arch::Qwen3 | Arch::K2Horizon => {
                entries.push(linear(&mut rng, &format!("{p}.mlp.gate_proj.weight"), INTER, HIDDEN, 1.0));
                entries.push(linear(&mut rng, &format!("{p}.mlp.up_proj.weight"), INTER, HIDDEN, 1.0));
                entries.push(linear(&mut rng, &format!("{p}.mlp.down_proj.weight"), HIDDEN, INTER, 1.0));
            }
            Arch::Mixtral => {
                entries.push(linear(&mut rng, &format!("{p}.block_sparse_moe.gate.weight"), EXPERTS, HIDDEN, 2.0));
                for e in 0..EXPERTS {
                    let q = format!("{p}.block_sparse_moe.experts.{e}");
                    entries.push(linear(&mut rng, &format!("{q}.w1.weight"), INTER, HIDDEN, 1.0));
                    entries.push(linear(&mut rng, &format!("{q}.w3.weight"), INTER, HIDDEN, 1.0));
                    entries.push(linear(&mut rng, &format!("{q}.w2.weight"), HIDDEN, INTER, 1.0));
                }
            }
            Arch::Qwen3Moe => {
                entries.push(linear(&mut rng, &format!("{p}.mlp.gate.weight"), EXPERTS, HIDDEN, 2.0));
                for e in 0..EXPERTS {
                    let q = format!("{p}.mlp.experts.{e}");
                    entries.push(linear(&mut rng, &format!("{q}.gate_proj.weight"), MOE_INTER, HIDDEN, 1.0));
                    entries.push(linear(&mut rng, &format!("{q}.up_proj.weight"), MOE_INTER, HIDDEN, 1.0));
                    entries.push(linear(&mut rng, &format!("{q}.down_proj.weight"), HIDDEN, MOE_INTER, 1.0));
                }
            }
            Arch::Deepseek => {
                if layer < 1 {
                    entries.push(linear(&mut rng, &format!("{p}.mlp.gate_proj.weight"), INTER, HIDDEN, 1.0));
                    entries.push(linear(&mut rng, &format!("{p}.mlp.up_proj.weight"), INTER, HIDDEN, 1.0));
                    entries.push(linear(&mut rng, &format!("{p}.mlp.down_proj.weight"), HIDDEN, INTER, 1.0));
                } else {
                    entries.push(linear(&mut rng, &format!("{p}.mlp.gate.weight"), EXPERTS, HIDDEN, 2.0));
                    for e in 0..EXPERTS {
                        let q = format!("{p}.mlp.experts.{e}");
                        entries.push(linear(&mut rng, &format!("{q}.gate_proj.weight"), MOE_INTER, HIDDEN, 1.0));
                        entries.push(linear(&mut rng, &format!("{q}.up_proj.weight"), MOE_INTER, HIDDEN, 1.0));
                        entries.push(linear(&mut rng, &format!("{q}.down_proj.weight"), HIDDEN, MOE_INTER, 1.0));
                    }
                    let s = format!("{p}.mlp.shared_experts");
                    entries.push(linear(&mut rng, &format!("{s}.gate_proj.weight"), MOE_INTER, HIDDEN, 1.0));
                    entries.push(linear(&mut rng, &format!("{s}.up_proj.weight"), MOE_INTER, HIDDEN, 1.0));
                    entries.push(linear(&mut rng, &format!("{s}.down_proj.weight"), HIDDEN, MOE_INTER, 1.0));
                }
            }
        }
    }
    write_support_files(dir, &config_json(arch));
    write_safetensors(&dir.join("model.safetensors"), &entries);
}

/// How the GGUF of a tiny model is encoded.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Encoding {
    F32,
    F16,
    /// `fllm-rs convert --outtype <preset>`: the F16 file re-encoded with the quantization preset
    Quant(&'static str),
}

/// Build a tiny model and return the path of its GGUF file (the directory is `dir`).
pub(crate) fn build_gguf(dir: &Path, arch: Arch, encoding: Encoding, seed: u64) -> PathBuf {
    let ckpt = dir.join("ckpt");
    fs::create_dir_all(&ckpt).unwrap();
    write_checkpoint(&ckpt, arch, seed);
    let dtype = match encoding {
        Encoding::F32 => OutputDType::F32,
        Encoding::F16 | Encoding::Quant(_) => OutputDType::F16,
    };
    let base = dir.join("model-base.gguf");
    convert_checkpoint(&ckpt, &base, ConvertOptions { output_dtype: dtype, overwrite: false, spec_head: None })
        .unwrap_or_else(|e| panic!("conversion of the tiny {arch:?} checkpoint failed: {e}"));
    match encoding {
        Encoding::Quant(name) => {
            let out = dir.join(format!("model-{name}.gguf"));
            let preset = Preset::parse(name).unwrap_or_else(|| panic!("unknown preset {name}"));
            quantize_gguf(&base, &out, preset).unwrap_or_else(|e| panic!("quantizing to {name} failed: {e}"));
            out
        }
        _ => base,
    }
}

/// What one decoding run produced: the greedy (or forced) tokens and the logits that selected them.
pub(crate) struct Run {
    pub tokens: Vec<i64>,
    pub logits: Vec<Vec<f32>>,
}

fn argmax(x: &[f32]) -> usize {
    x.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0
}

/// Prefill `prompt`, then decode `steps` tokens: the greedy ones, or `forced[i]` where given. `logits[i]` are the logits that
/// predict token `i`.
pub(crate) fn decode(dec: &RustDecoder, model: &GgufModel, prompt: &[i64], steps: usize, forced: Option<&[i64]>) -> Result<Run, String> {
    let mut key = vec![0u16; model.key_buffer_len()?];
    let mut value = vec![0u16; model.value_buffer_len()?];
    let stride = model.table_stride();
    let table: Vec<i64> = (0..stride as i64).collect();
    let rows = prompt.len();
    let tables: Vec<i64> = (0..rows).flat_map(|_| table.iter().copied()).collect();
    let lengths: Vec<i64> = (1..=rows as i64).collect();
    let vocab = model.config().vocab;
    let mut run = Run { tokens: Vec::new(), logits: Vec::new() };
    let first = dec.forward(prompt, &lengths, &tables, &mut key, &mut value, &[rows - 1], false)?;
    if first.len() != vocab {
        return Err(format!("prefill returned {} logits, expected {vocab}", first.len()));
    }
    run.logits.push(first);
    for i in 0..steps {
        let token = match forced {
            Some(f) => f[i],
            None => argmax(&run.logits[i]) as i64,
        };
        run.tokens.push(token);
        if i + 1 == steps {
            break;
        }
        let length = (rows + i + 1) as i64;
        let next = dec.forward(&[token], &[length], &table, &mut key, &mut value, &[0], false)?;
        if next.len() != vocab {
            return Err(format!("decode step {i} returned {} logits, expected {vocab}", next.len()));
        }
        run.logits.push(next);
    }
    Ok(run)
}

fn margin(x: &[f32]) -> f32 {
    let top = argmax(x);
    let second = x.iter().enumerate().filter(|(i, _)| *i != top).fold(f32::NEG_INFINITY, |m, (_, &v)| m.max(v));
    x[top] - second
}

/// Largest absolute difference over the largest absolute value of `want`, and where the largest difference is.
fn relative_difference(got: &[f32], want: &[f32]) -> (f32, usize) {
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let mut worst = (0f32, 0usize);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = if g.is_finite() { (g - w).abs() / scale } else { f32::INFINITY };
        if d > worst.0 || d.is_nan() {
            worst = (d, i);
        }
    }
    worst
}

pub(crate) const PROMPT: [i64; 6] = [1, 5, 9, 3, 17, 40];
pub(crate) const STEPS: usize = 16;

/// The tolerance from `FLLM_MODEL_TEST_TOL` (relative to the largest logit) or `default`.
pub(crate) fn tolerance(default: f32) -> f32 {
    std::env::var("FLLM_MODEL_TEST_TOL").ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Load `gguf` twice, once for the host path and once attached to `backend`, and compare:
///
/// 1. the host run is greedy; the device run is fed the host's tokens, and the logits of all 16 steps must agree within
///    `tol` (largest absolute difference relative to the largest absolute logit);
/// 2. a free-running greedy device run must produce the host's tokens, up to the first step where the host's two best logits are
///    closer than twice the tolerance (a different token there is a legitimate rounding outcome and ends the comparison).
///
/// Returns a one-line summary.
pub(crate) fn compare_with_host(label: &str, gguf: &Path, backend: Arc<dyn Backend>, tol: f32, max_host: (usize, usize)) -> Result<String, String> {
    let model = GgufModel::load(gguf.to_str().unwrap(), 64, 1, 2)?;
    let host = RustDecoder::load(&model, "cpu")?;
    let reference = decode(&host, &model, &PROMPT, STEPS, None)?;
    let scale = reference.logits.iter().flatten().fold(0f32, |m, v| m.max(v.abs()));
    if scale.is_nan() || scale <= 1e-3 || reference.logits.iter().flatten().any(|v| !v.is_finite()) {
        return Err(format!("{label}: the host logits are degenerate (largest {scale})"));
    }

    let mut device = RustDecoder::load(&model, "cpu")?;
    device.init_backend(Arc::clone(&backend)).map_err(|e| format!("{label}: attaching the {} backend failed: {e}", backend.name()))?;
    if !device.uses_accelerator() {
        return Err(format!("{label}: the decoder does not use the {} backend", backend.name()));
    }
    let (host_layers, host_ffn) = device.host_fallbacks();
    if host_layers > max_host.0 || host_ffn > max_host.1 {
        return Err(format!(
            "{label}: {host_layers} layer(s) and {host_ffn} feed-forward block(s) run on the host instead of the {} backend (allowed {} and {}); the log line \"layers run on the host between the device layers\" names the reason",
            backend.name(), max_host.0, max_host.1
        ));
    }
    let forced = decode(&device, &model, &PROMPT, STEPS, Some(&reference.tokens))?;
    let mut worst = (0f32, 0usize, 0usize);
    for (step, (g, w)) in forced.logits.iter().zip(&reference.logits).enumerate() {
        let (d, at) = relative_difference(g, w);
        if d > worst.0 || d.is_nan() {
            worst = (d, step, at);
        }
    }
    if worst.0.is_nan() || worst.0 > tol {
        let (_, step, at) = worst;
        return Err(format!(
            "{label}: logits differ by {:.3e} of the largest logit {scale:.3} (tolerance {tol:.1e}) at step {step}, vocabulary entry {at}: device {} host {}",
            worst.0, forced.logits[step][at], reference.logits[step][at]
        ));
    }

    let mut device = RustDecoder::load(&model, "cpu")?;
    device.init_backend(backend).map_err(|e| format!("{label}: {e}"))?;
    let free = decode(&device, &model, &PROMPT, STEPS, None)?;
    let mut same = 0usize;
    for i in 0..STEPS {
        if free.tokens[i] == reference.tokens[i] {
            same += 1;
            continue;
        }
        let gap = margin(&reference.logits[i]);
        if gap > 2.0 * tol * scale {
            return Err(format!(
                "{label}: greedy token {i} is {} on the device and {} on the host although the host margin is {gap:.3e} (tolerance {:.3e}); host tokens {:?}, device tokens {:?}",
                free.tokens[i], reference.tokens[i], 2.0 * tol * scale, reference.tokens, free.tokens
            ));
        }
        break;
    }
    Ok(format!(
        "{label}: {STEPS} steps, largest logit difference {:.3e} (tolerance {tol:.1e}), {same}/{STEPS} free-running greedy tokens equal, {host_layers} host layer(s), {host_ffn} host feed-forward block(s), host tokens {:?}",
        worst.0, reference.tokens
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;

    fn run(label: &str, arch: Arch, encoding: Encoding, tol: f32) {
        let dir = scratch(label);
        let gguf = build_gguf(&dir, arch, encoding, 0x9e37_79b9_7f4a_7c15);
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let result = compare_with_host(label, &gguf, backend, tolerance(tol), (usize::MAX, usize::MAX));
        let _ = fs::remove_dir_all(&dir);
        match result {
            Ok(line) => eprintln!("{line}"),
            Err(e) => panic!("{e}"),
        }
    }

    /// The fixtures, the converter and the comparison itself, with the CPU backend in the role of the device.
    #[test]
    fn cpu_backend_matches_host_path_llama_f32() {
        run("llama-f32", Arch::Llama, Encoding::F32, 1e-3);
    }

    #[test]
    fn cpu_backend_matches_host_path_chatglm_f32() {
        run("chatglm-f32", Arch::ChatGLM, Encoding::F32, 1e-3);
    }

    #[test]
    fn cpu_backend_matches_host_path_exaone_f32() {
        run("exaone-f32", Arch::Exaone, Encoding::F32, 1e-3);
    }

    #[test]
    fn cpu_backend_matches_host_path_k2_horizon() {
        run("k2-horizon", Arch::K2Horizon, Encoding::F32, 1e-3);
    }

    #[test]
    fn cpu_backend_matches_host_path_qwen3_q8_0() {
        run("qwen3-q8", Arch::Qwen3, Encoding::Quant("q8_0"), 8e-2);
    }

    /// The MoE fixtures: the CPU backend has no MoE kernels, so these only check that the converted models load and decode on the
    /// host path with finite, non-degenerate logits (the comparison with a device is in the device backends' own tests).
    fn host_only(label: &str, arch: Arch, encoding: Encoding) {
        let dir = scratch(label);
        let gguf = build_gguf(&dir, arch, encoding, 0x1234_5678_9abc_def1);
        let model = GgufModel::load(gguf.to_str().unwrap(), 64, 1, 2).unwrap();
        let host = RustDecoder::load(&model, "cpu").unwrap();
        let run = decode(&host, &model, &PROMPT, STEPS, None).unwrap();
        let again = decode(&host, &model, &PROMPT, STEPS, None).unwrap();
        let _ = fs::remove_dir_all(&dir);
        assert_eq!(run.tokens, again.tokens, "{label}: the host path is not deterministic");
        let top = run.logits.iter().flatten().fold(0f32, |m, v| m.max(v.abs()));
        assert!(top > 1e-3 && run.logits.iter().flatten().all(|v| v.is_finite()), "{label}: degenerate logits");
        assert!(run.tokens.iter().any(|t| *t != run.tokens[0]), "{label}: constant output {:?}", run.tokens);
    }

    #[test]
    fn mixtral_fixture_runs_on_the_host() {
        host_only("mixtral-f16", Arch::Mixtral, Encoding::F16);
    }

    #[test]
    fn qwen3_moe_fixture_runs_on_the_host_quantized() {
        host_only("qwen3moe-q4km", Arch::Qwen3Moe, Encoding::Quant("q4_k_m"));
    }

    #[test]
    fn qwen35_fixture_runs_on_the_host() {
        host_only("qwen35-f32", Arch::Qwen35, Encoding::F32);
    }

    #[test]
    fn qwen35_fixture_runs_on_the_host_quantized() {
        host_only("qwen35-q8", Arch::Qwen35, Encoding::Quant("q8_0"));
    }

    #[test]
    fn deepseek_fixture_runs_on_the_host() {
        host_only("deepseek-f32", Arch::Deepseek, Encoding::F32);
    }

    #[test]
    fn falcon_h1_fixture_runs_on_the_host() {
        for norm in [false, true] {
            let (config, specs) = crate::hybrid_tests::falcon_h1_specs(norm);
            crate::hybrid_tests::exercise_state(config, &specs, 0x5eed_0001);
        }
    }

    #[test]
    fn jamba_fixture_runs_on_the_host() {
        let (config, specs) = crate::hybrid_tests::jamba_specs();
        crate::hybrid_tests::exercise_state(config, &specs, 0x5eed_0002);
    }

    #[test]
    fn zamba2_fixture_runs_on_the_host() {
        let (config, specs) = crate::hybrid_tests::zamba2_specs();
        crate::hybrid_tests::exercise_state(config, &specs, 0x5eed_0003);
    }
}
