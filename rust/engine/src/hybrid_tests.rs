//! End-to-end checks of the hybrid state-space / attention families with tiny random checkpoints, CPU only: conversion
//! from Hugging Face tensor names, token-by-token decoding against one batched pass, and the snapshot / restore API of the
//! recurrent state (speculative-decoding rollback, prefix reuse).
//!
//! The comparison with the reference implementations (transformers, fla) lives in `scripts/verify_hybrid.py`; these tests
//! need no Python and no network.

use crate::{rust_decoder::RustDecoder, GgufModel};
use fllm_checkpoint::{convert_checkpoint, ConvertOptions, OutputDType};
use serde_json::{json, Value};
use std::{fs, path::Path};

const VOCAB: usize = 48;

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

/// How the values of a tensor are drawn.
#[derive(Clone, Copy)]
pub(crate) enum Init {
    /// Weights of projections.
    Small,
    /// Norm scales around one.
    Norm,
    /// `A_log`: logarithm of rates between 0.4 and 2.5.
    ALog,
    /// Biases and the skip connection.
    Bias,
}

pub(crate) struct Spec {
    pub name: String,
    pub shape: Vec<usize>,
    pub init: Init,
}

pub(crate) fn spec(name: &str, shape: &[usize], init: Init) -> Spec {
    Spec { name: name.into(), shape: shape.to_vec(), init }
}

/// Writes `config.json`, a minimal tokenizer and a random `model.safetensors` for `specs` into `dir`.
pub(crate) fn write_checkpoint(dir: &Path, mut config: Value, specs: &[Spec], seed: u64) {
    fs::create_dir_all(dir).unwrap();
    config["vocab_size"] = VOCAB.into();
    config["bos_token_id"] = 1.into();
    config["eos_token_id"] = 2.into();
    fs::write(dir.join("config.json"), config.to_string()).unwrap();
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
    fs::write(dir.join("tokenizer_config.json"), json!({"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>"}).to_string())
        .unwrap();
    let mut rng = Rng(seed | 1);
    let mut header = serde_json::Map::new();
    let mut body = Vec::new();
    for s in specs {
        let count: usize = s.shape.iter().product();
        let start = body.len();
        for _ in 0..count {
            let v = match s.init {
                Init::Small => 0.35 * rng.unit(),
                Init::Norm => 1.0 + 0.2 * rng.unit(),
                Init::ALog => (0.4 + 2.1 * (rng.unit() * 0.5 + 0.5)).ln(),
                Init::Bias => 0.5 * rng.unit(),
            };
            body.extend_from_slice(&v.to_le_bytes());
        }
        header.insert(s.name.clone(), json!({"dtype": "F32", "shape": s.shape, "data_offsets": [start, body.len()]}));
    }
    let header = Value::Object(header).to_string();
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&body);
    fs::write(dir.join("model.safetensors"), out).unwrap();
}

pub(crate) fn scratch(label: &str) -> std::path::PathBuf {
    let n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let p = std::env::temp_dir().join(format!("fllm-hybrid-{label}-{}-{n}", std::process::id()));
    fs::create_dir_all(&p).unwrap();
    p
}

/// A loaded model with a KV cache for one sequence.
pub(crate) struct Runner {
    pub decoder: RustDecoder,
    key: Vec<u16>,
    value: Vec<u16>,
    table: Vec<i64>,
}

impl Runner {
    pub(crate) fn open(gguf: &Path) -> Self {
        let mut model = GgufModel::load(gguf.to_str().unwrap(), 64, 1, 2).unwrap();
        model.set_kv_kind(1);
        let decoder = RustDecoder::load(&model, "cpu").unwrap();
        let table = (0..model.table_stride()).map(|n| n as i64).collect();
        Self {
            key: vec![0u16; model.key_buffer_len().unwrap()],
            value: vec![0u16; model.value_buffer_len().unwrap()],
            decoder,
            table,
        }
    }

    /// Logits of `tokens` at positions `start..`, as one batch.
    pub(crate) fn forward(&mut self, tokens: &[i64], start: usize) -> Vec<f32> {
        let stride = self.table.len();
        let lengths: Vec<i64> = (0..tokens.len()).map(|i| (start + i + 1) as i64).collect();
        let tables: Vec<i64> = self.table.iter().copied().cycle().take(stride * tokens.len()).collect();
        let rows: Vec<usize> = (0..tokens.len()).collect();
        self.decoder.forward(tokens, &lengths, &tables, &mut self.key, &mut self.value, &rows, false).unwrap()
    }

    /// The same tokens one forward pass each.
    pub(crate) fn one_by_one(&mut self, tokens: &[i64], start: usize) -> Vec<f32> {
        tokens.iter().enumerate().flat_map(|(i, t)| self.forward(&[*t], start + i)).collect()
    }

    pub(crate) fn first_block(&self) -> i64 {
        self.table[0]
    }
}

fn close(a: &[f32], b: &[f32], tolerance: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let worst = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale;
    assert!(worst <= tolerance, "{what}: largest logit difference {worst} (relative to {scale}) exceeds {tolerance}");
    assert!(b.iter().any(|v| v.abs() > 1e-3), "{what}: reference logits are all zero");
}

/// Batched against incremental decoding, then rollback through `snapshot_state` / `restore_state`.
pub(crate) fn exercise_state(config: Value, specs: &[Spec], seed: u64) {
    let dir = scratch("state");
    write_checkpoint(&dir, config, specs, seed);
    let gguf = dir.join("model.gguf");
    convert_checkpoint(&dir, &gguf, ConvertOptions { output_dtype: OutputDType::F32, overwrite: false, spec_head: None })
        .unwrap_or_else(|e| panic!("conversion failed: {e}"));
    let mut run = Runner::open(&gguf);
    assert!(run.decoder.has_recurrent_state());
    let first = run.first_block();
    let tokens: Vec<i64> = vec![5, 9, 3, 17, 40, 8, 21, 2, 30];

    let batched = run.forward(&tokens, 0);
    run.decoder.forget_state(first);
    let stepwise = run.one_by_one(&tokens, 0);
    close(&stepwise, &batched, 1e-4, "token by token vs one batch");

    // roll back after a prefix: another continuation, then the original one again
    run.decoder.forget_state(first);
    let prefix = 5;
    run.forward(&tokens[..prefix], 0);
    let saved = run.decoder.snapshot_state(first).unwrap().expect("state after the prefix");
    assert_eq!(saved.tokens(), Some(prefix));
    let tail = run.forward(&tokens[prefix..], prefix);
    let other: Vec<i64> = vec![11, 12, 13];
    run.decoder.restore_state(first, &saved).unwrap();
    let wrong_way = run.forward(&other, prefix);
    run.decoder.restore_state(first, &saved).unwrap();
    let again = run.forward(&tokens[prefix..], prefix);
    close(&again, &tail, 1e-6, "continuation after restore");
    run.decoder.restore_state(first, &saved).unwrap();
    let rerouted = run.forward(&other, prefix);
    close(&rerouted, &wrong_way, 1e-6, "second continuation after restore");

    // the rolled-back run equals a fresh run of prefix + other tokens
    run.decoder.forget_state(first);
    let fresh_tokens: Vec<i64> = tokens[..prefix].iter().chain(&other).copied().collect();
    let fresh = run.forward(&fresh_tokens, 0);
    close(&rerouted, &fresh[prefix * VOCAB..], 1e-4, "rolled back vs fresh");
    let _ = fs::remove_dir_all(dir);
}

/// Tensor list of a Falcon-H1 model: every layer has attention and Mamba-2 side by side.
fn falcon_h1_specs(norm: bool) -> (Value, Vec<Spec>) {
    let (hidden, layers, heads, kv, head_dim, inter) = (32usize, 2usize, 4usize, 2usize, 8usize, 48usize);
    let (inner, mheads, groups, state, kernel) = (32usize, 4usize, 2usize, 8usize, 4usize);
    let conv_dim = inner + 2 * groups * state;
    let config = json!({
        "model_type": "falcon_h1", "hidden_size": hidden, "num_hidden_layers": layers, "num_attention_heads": heads,
        "num_key_value_heads": kv, "head_dim": head_dim, "intermediate_size": inter, "rms_norm_eps": 1e-5,
        "max_position_embeddings": 64, "mamba_d_ssm": inner, "mamba_n_heads": mheads, "mamba_d_head": inner / mheads,
        "mamba_n_groups": groups, "mamba_d_state": state, "mamba_d_conv": kernel, "mamba_conv_bias": true,
        "mamba_rms_norm": norm, "mamba_norm_before_gate": true, "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        "tie_word_embeddings": false, "lm_head_multiplier": 0.7, "embedding_multiplier": 1.5, "mlp_multipliers": [0.8, 1.3],
        "key_multiplier": 0.9, "attention_out_multiplier": 1.2, "attention_in_multiplier": 0.6,
        "ssm_multipliers": [1.1, 0.9, 1.3, 0.7, 1.2], "ssm_in_multiplier": 0.8, "ssm_out_multiplier": 1.4
    });
    let mut s = vec![
        spec("model.embed_tokens.weight", &[VOCAB, hidden], Init::Small),
        spec("lm_head.weight", &[VOCAB, hidden], Init::Small),
        spec("model.final_layernorm.weight", &[hidden], Init::Norm),
    ];
    for l in 0..layers {
        let p = format!("model.layers.{l}");
        s.extend([
            spec(&format!("{p}.input_layernorm.weight"), &[hidden], Init::Norm),
            spec(&format!("{p}.pre_ff_layernorm.weight"), &[hidden], Init::Norm),
            spec(&format!("{p}.self_attn.q_proj.weight"), &[heads * head_dim, hidden], Init::Small),
            spec(&format!("{p}.self_attn.k_proj.weight"), &[kv * head_dim, hidden], Init::Small),
            spec(&format!("{p}.self_attn.v_proj.weight"), &[kv * head_dim, hidden], Init::Small),
            spec(&format!("{p}.self_attn.o_proj.weight"), &[hidden, heads * head_dim], Init::Small),
            spec(&format!("{p}.feed_forward.gate_proj.weight"), &[inter, hidden], Init::Small),
            spec(&format!("{p}.feed_forward.up_proj.weight"), &[inter, hidden], Init::Small),
            spec(&format!("{p}.feed_forward.down_proj.weight"), &[hidden, inter], Init::Small),
            spec(&format!("{p}.mamba.in_proj.weight"), &[inner + conv_dim + mheads, hidden], Init::Small),
            spec(&format!("{p}.mamba.conv1d.weight"), &[conv_dim, 1, kernel], Init::Small),
            spec(&format!("{p}.mamba.conv1d.bias"), &[conv_dim], Init::Bias),
            spec(&format!("{p}.mamba.dt_bias"), &[mheads], Init::Bias),
            spec(&format!("{p}.mamba.A_log"), &[mheads], Init::ALog),
            spec(&format!("{p}.mamba.D"), &[mheads], Init::Bias),
            spec(&format!("{p}.mamba.out_proj.weight"), &[hidden, inner], Init::Small),
        ]);
        if norm {
            s.push(spec(&format!("{p}.mamba.norm.weight"), &[inner], Init::Norm));
        }
    }
    (config, s)
}

#[test]
fn falcon_h1_decodes_incrementally_and_rolls_back() {
    for norm in [false, true] {
        let (config, specs) = falcon_h1_specs(norm);
        exercise_state(config, &specs, 0x5eed_0001);
    }
}

/// Tensor list of a Jamba model: Mamba-1 layers with dt/B/C norms, one attention layer, experts on every second layer.
fn jamba_specs() -> (Value, Vec<Spec>) {
    let (hidden, layers, heads, kv, head_dim, inter) = (32usize, 4usize, 4usize, 2usize, 8usize, 48usize);
    let (experts, state, kernel, rank) = (4usize, 8usize, 4usize, 4usize);
    let inner = 2 * hidden;
    let config = json!({
        "model_type": "jamba", "hidden_size": hidden, "num_hidden_layers": layers, "num_attention_heads": heads,
        "num_key_value_heads": kv, "intermediate_size": inter, "rms_norm_eps": 1e-6, "max_position_embeddings": 64,
        "num_experts": experts, "num_experts_per_tok": 2, "expert_layer_period": 2, "expert_layer_offset": 1,
        "attn_layer_period": 4, "attn_layer_offset": 2, "mamba_d_state": state, "mamba_d_conv": kernel,
        "mamba_expand": 2, "mamba_dt_rank": rank, "mamba_conv_bias": true, "mamba_proj_bias": false,
        "tie_word_embeddings": false
    });
    let mut s = vec![
        spec("model.embed_tokens.weight", &[VOCAB, hidden], Init::Small),
        spec("lm_head.weight", &[VOCAB, hidden], Init::Small),
        spec("model.final_layernorm.weight", &[hidden], Init::Norm),
    ];
    for l in 0..layers {
        let p = format!("model.layers.{l}");
        s.extend([
            spec(&format!("{p}.input_layernorm.weight"), &[hidden], Init::Norm),
            spec(&format!("{p}.pre_ff_layernorm.weight"), &[hidden], Init::Norm),
        ]);
        if l % 4 == 2 {
            s.extend([
                spec(&format!("{p}.self_attn.q_proj.weight"), &[heads * head_dim, hidden], Init::Small),
                spec(&format!("{p}.self_attn.k_proj.weight"), &[kv * head_dim, hidden], Init::Small),
                spec(&format!("{p}.self_attn.v_proj.weight"), &[kv * head_dim, hidden], Init::Small),
                spec(&format!("{p}.self_attn.o_proj.weight"), &[hidden, heads * head_dim], Init::Small),
            ]);
        } else {
            s.extend([
                spec(&format!("{p}.mamba.in_proj.weight"), &[2 * inner, hidden], Init::Small),
                spec(&format!("{p}.mamba.conv1d.weight"), &[inner, 1, kernel], Init::Small),
                spec(&format!("{p}.mamba.conv1d.bias"), &[inner], Init::Bias),
                spec(&format!("{p}.mamba.x_proj.weight"), &[rank + 2 * state, inner], Init::Small),
                spec(&format!("{p}.mamba.dt_proj.weight"), &[inner, rank], Init::Small),
                spec(&format!("{p}.mamba.dt_proj.bias"), &[inner], Init::Bias),
                spec(&format!("{p}.mamba.A_log"), &[inner, state], Init::ALog),
                spec(&format!("{p}.mamba.D"), &[inner], Init::Bias),
                spec(&format!("{p}.mamba.out_proj.weight"), &[hidden, inner], Init::Small),
                spec(&format!("{p}.mamba.dt_layernorm.weight"), &[rank], Init::Norm),
                spec(&format!("{p}.mamba.b_layernorm.weight"), &[state], Init::Norm),
                spec(&format!("{p}.mamba.c_layernorm.weight"), &[state], Init::Norm),
            ]);
        }
        if l % 2 == 1 {
            s.push(spec(&format!("{p}.feed_forward.router.weight"), &[experts, hidden], Init::Small));
            for e in 0..experts {
                s.extend([
                    spec(&format!("{p}.feed_forward.experts.{e}.gate_proj.weight"), &[inter, hidden], Init::Small),
                    spec(&format!("{p}.feed_forward.experts.{e}.up_proj.weight"), &[inter, hidden], Init::Small),
                    spec(&format!("{p}.feed_forward.experts.{e}.down_proj.weight"), &[hidden, inter], Init::Small),
                ]);
            }
        } else {
            s.extend([
                spec(&format!("{p}.feed_forward.gate_proj.weight"), &[inter, hidden], Init::Small),
                spec(&format!("{p}.feed_forward.up_proj.weight"), &[inter, hidden], Init::Small),
                spec(&format!("{p}.feed_forward.down_proj.weight"), &[hidden, inter], Init::Small),
            ]);
        }
    }
    (config, s)
}

#[test]
fn jamba_decodes_incrementally_and_rolls_back() {
    let (config, specs) = jamba_specs();
    exercise_state(config, &specs, 0x5eed_0002);
}

/// Tensor list of a Zamba2 model with three shared-block calls over two blocks (the third call repeats block 0, whose
/// weights the checkpoint stores once).
fn zamba2_specs() -> (Value, Vec<Spec>) {
    let (hidden, layers, heads, inter, rank) = (32usize, 6usize, 4usize, 48usize, 4usize);
    let (inner, mheads, groups, state, kernel) = (64usize, 4usize, 2usize, 8usize, 4usize);
    let conv_dim = inner + 2 * groups * state;
    let wide = 2 * hidden;
    let kinds = ["mamba", "hybrid", "mamba", "hybrid", "mamba", "hybrid"];
    let config = json!({
        "model_type": "zamba2", "hidden_size": hidden, "num_hidden_layers": layers, "layers_block_type": kinds,
        "num_attention_heads": heads, "num_key_value_heads": heads, "intermediate_size": inter, "rms_norm_eps": 1e-5,
        "max_position_embeddings": 64, "mamba_d_state": state, "mamba_d_conv": kernel, "mamba_expand": 2,
        "mamba_ngroups": groups, "n_mamba_heads": mheads, "num_mem_blocks": 2, "use_shared_attention_adapter": true,
        "adapter_rank": rank, "use_mem_rope": true, "time_step_min": 0.001, "hidden_act": "gelu",
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0}
    });
    let mut s = vec![
        spec("model.embed_tokens.weight", &[VOCAB, hidden], Init::Small),
        spec("model.final_layernorm.weight", &[hidden], Init::Norm),
    ];
    let mut call = 0usize;
    for (l, kind) in kinds.iter().enumerate() {
        let p = format!("model.layers.{l}");
        let m = if *kind == "hybrid" { format!("{p}.mamba_decoder") } else { p.clone() };
        s.extend([
            spec(&format!("{m}.input_layernorm.weight"), &[hidden], Init::Norm),
            spec(&format!("{m}.mamba.in_proj.weight"), &[inner + conv_dim + mheads, hidden], Init::Small),
            spec(&format!("{m}.mamba.conv1d.weight"), &[conv_dim, 1, kernel], Init::Small),
            spec(&format!("{m}.mamba.conv1d.bias"), &[conv_dim], Init::Bias),
            spec(&format!("{m}.mamba.dt_bias"), &[mheads], Init::Bias),
            spec(&format!("{m}.mamba.A_log"), &[mheads], Init::ALog),
            spec(&format!("{m}.mamba.D"), &[mheads], Init::Bias),
            spec(&format!("{m}.mamba.norm.weight"), &[inner], Init::Norm),
            spec(&format!("{m}.mamba.out_proj.weight"), &[hidden, inner], Init::Small),
        ]);
        if *kind != "hybrid" {
            continue;
        }
        s.push(spec(&format!("{p}.linear.weight"), &[hidden, hidden], Init::Small));
        // the checkpoint stores the shared weights in the first layer of each block and the adapters of every call
        // in the module of the block the call uses
        if call < 2 {
            let t = format!("{p}.shared_transformer");
            s.extend([
                spec(&format!("{t}.input_layernorm.weight"), &[wide], Init::Norm),
                spec(&format!("{t}.pre_ff_layernorm.weight"), &[hidden], Init::Norm),
                spec(&format!("{t}.self_attn.q_proj.weight"), &[wide, wide], Init::Small),
                spec(&format!("{t}.self_attn.k_proj.weight"), &[wide, wide], Init::Small),
                spec(&format!("{t}.self_attn.v_proj.weight"), &[wide, wide], Init::Small),
                spec(&format!("{t}.self_attn.o_proj.weight"), &[hidden, wide], Init::Small),
                spec(&format!("{t}.feed_forward.gate_up_proj.weight"), &[2 * inter, hidden], Init::Small),
                spec(&format!("{t}.feed_forward.down_proj.weight"), &[hidden, inter], Init::Small),
            ]);
        }
        let holder = [1usize, 3][call % 2];
        let t = format!("model.layers.{holder}.shared_transformer");
        for q in ["q", "k", "v"] {
            s.push(spec(&format!("{t}.self_attn.linear_{q}_adapter_list.{call}.0.weight"), &[rank, wide], Init::Small));
            s.push(spec(&format!("{t}.self_attn.linear_{q}_adapter_list.{call}.1.weight"), &[wide, rank], Init::Small));
        }
        s.push(spec(&format!("{t}.feed_forward.gate_up_proj_adapter_list.{call}.0.weight"), &[rank, hidden], Init::Small));
        s.push(spec(&format!("{t}.feed_forward.gate_up_proj_adapter_list.{call}.1.weight"), &[2 * inter, rank], Init::Small));
        call += 1;
    }
    (config, s)
}

#[test]
fn zamba2_decodes_incrementally_and_rolls_back() {
    let (config, specs) = zamba2_specs();
    exercise_state(config, &specs, 0x5eed_0003);
}
