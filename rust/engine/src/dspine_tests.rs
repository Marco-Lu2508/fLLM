//! End-to-end tests for DSpine (`DSpineDraftModel`), DFlash (`DFlashDraftModel`),
//! and `bailing_moe` GQA checkpoints through direct loading, conversion, `RustDecoder`,
//! and `InferenceRuntime`.

use crate::import::{is_direct, open_model};
use crate::runtime::InferenceRuntime;
use crate::sampling::SamplingParams;
use crate::speculative::SpeculativeMethod;
use crate::tiny_models::{
    build_gguf, scratch, Arch, Encoding, HEADS, HEAD_DIM, HIDDEN, INTER, KV_HEADS, LAYERS, VOCAB,
};
use crate::GgufModel;
use fllm_checkpoint::{convert_checkpoint, ConvertOptions, OutputDType};
use serde_json::json;
use std::fs;
use std::path::Path;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
    fn vec(&mut self, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|_| self.unit() * scale).collect()
    }
}

struct Entry {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn bf16_entry(name: &str, shape: &[usize], values: &[f32]) -> Entry {
    Entry {
        name: name.into(),
        dtype: "BF16",
        shape: shape.to_vec(),
        bytes: values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect(),
    }
}

fn f32_entry(name: &str, shape: &[usize], values: &[f32]) -> Entry {
    Entry {
        name: name.into(),
        dtype: "F32",
        shape: shape.to_vec(),
        bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

fn i64_scalar_entry(name: &str, value: i64) -> Entry {
    Entry {
        name: name.into(),
        dtype: "I64",
        shape: Vec::new(),
        bytes: value.to_le_bytes().to_vec(),
    }
}

fn write_safetensors(path: &Path, entries: &[Entry]) {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for e in entries {
        header.insert(
            e.name.clone(),
            json!({
                "dtype": e.dtype,
                "shape": e.shape,
                "data_offsets": [offset, offset + e.bytes.len()]
            }),
        );
        offset += e.bytes.len();
    }
    let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let pad = (8 - (header_bytes.len() % 8)) % 8;
    let total_header = header_bytes.len() + pad;
    let mut out = Vec::with_capacity(8 + total_header + offset);
    out.extend_from_slice(&(total_header as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.resize(8 + total_header, b' ');
    for e in entries {
        out.extend_from_slice(&e.bytes);
    }
    fs::write(path, out).unwrap();
}

fn write_minimal_tokenizer(dir: &Path, vocab: usize) {
    let mut tokens = Vec::with_capacity(vocab);
    for i in 0..vocab {
        tokens.push(format!("<t{i}>"));
    }
    let tok = json!({
        "version": "1.0",
        "model": {
            "type": "BPE",
            "vocab": tokens.iter().enumerate().map(|(i, t)| (t.clone(), json!(i))).collect::<serde_json::Map<String, serde_json::Value>>(),
            "merges": []
        }
    });
    fs::write(dir.join("tokenizer.json"), serde_json::to_vec(&tok).unwrap()).unwrap();
}

/// Writes a DSpine or DFlash draft checkpoint WITHOUT `tokenizer.json` matching `tiny_models` geometry.
fn write_draft_checkpoint(dir: &Path, with_bridge: bool, seed: u64) {
    fs::create_dir_all(dir).unwrap();
    let mut rng = Rng(seed | 1);
    let draft_layers = 2usize;
    let code_size = 8usize;
    let rank = 4usize;
    let rt = if with_bridge { 2usize } else { 0usize };
    let target_ids = vec![0usize, 1usize];

    let cfg = if with_bridge {
        json!({
            "architectures": ["DSpineDraftModel"],
            "model_type": "qwen3",
            "hidden_size": HIDDEN,
            "intermediate_size": INTER,
            "num_hidden_layers": draft_layers,
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "vocab_size": VOCAB,
            "max_position_embeddings": 256,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "block_size": 4,
            "num_target_layers": LAYERS,
            "sliding_window": 12,
            "code_bridge_second_pass": {
                "mode": "serial_table",
                "first_exact": draft_layers - 1,
                "head_topk": 4
            },
            "dflash_config": {
                "mask_token_id": VOCAB - 1,
                "target_layer_ids": target_ids,
                "draft_block_causal": true,
                "selected_prefix_terminal": false,
                "selected_prefix_feature_writes": true,
                "selected_prefix_feature_write_mode": "independent",
                "selected_prefix_feature_layers": [1, 2],
                "selected_prefix_terminal_residual_rank": rt,
                "selected_prefix_code_bridge": {
                    "code_size": code_size,
                    "rank": rank,
                    "eps": 1e-6
                }
            }
        })
    } else {
        json!({
            "architectures": ["DFlashDraftModel"],
            "model_type": "qwen3",
            "hidden_size": HIDDEN,
            "intermediate_size": INTER,
            "num_hidden_layers": draft_layers,
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "vocab_size": VOCAB,
            "max_position_embeddings": 256,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "block_size": 4,
            "num_target_layers": LAYERS,
            "dflash_config": {
                "mask_token_id": VOCAB - 1,
                "target_layer_ids": target_ids,
                "draft_block_causal": false
            }
        })
    };
    fs::write(dir.join("config.json"), serde_json::to_vec_pretty(&cfg).unwrap()).unwrap();

    let mut entries = vec![
        bf16_entry(
            "model.fc.weight",
            &[HIDDEN, target_ids.len() * HIDDEN],
            &rng.vec(HIDDEN * target_ids.len() * HIDDEN, 0.05),
        ),
        bf16_entry("model.hidden_norm.weight", &[HIDDEN], &vec![1.0; HIDDEN]),
        bf16_entry("model.norm.weight", &[HIDDEN], &vec![1.0; HIDDEN]),
    ];

    for i in 0..draft_layers {
        let p = format!("model.layers.{i}");
        entries.extend([
            bf16_entry(&format!("{p}.input_layernorm.weight"), &[HIDDEN], &vec![1.0; HIDDEN]),
            bf16_entry(&format!("{p}.post_attention_layernorm.weight"), &[HIDDEN], &vec![1.0; HIDDEN]),
            bf16_entry(&format!("{p}.self_attn.q_proj.weight"), &[HEADS * HEAD_DIM, HIDDEN], &rng.vec(HEADS * HEAD_DIM * HIDDEN, 0.05)),
            bf16_entry(&format!("{p}.self_attn.k_proj.weight"), &[KV_HEADS * HEAD_DIM, HIDDEN], &rng.vec(KV_HEADS * HEAD_DIM * HIDDEN, 0.05)),
            bf16_entry(&format!("{p}.self_attn.v_proj.weight"), &[KV_HEADS * HEAD_DIM, HIDDEN], &rng.vec(KV_HEADS * HEAD_DIM * HIDDEN, 0.05)),
            bf16_entry(&format!("{p}.self_attn.o_proj.weight"), &[HIDDEN, HEADS * HEAD_DIM], &rng.vec(HIDDEN * HEADS * HEAD_DIM, 0.05)),
            bf16_entry(&format!("{p}.self_attn.q_norm.weight"), &[HEAD_DIM], &vec![1.0; HEAD_DIM]),
            bf16_entry(&format!("{p}.self_attn.k_norm.weight"), &[HEAD_DIM], &vec![1.0; HEAD_DIM]),
            bf16_entry(&format!("{p}.mlp.gate_proj.weight"), &[INTER, HIDDEN], &rng.vec(INTER * HIDDEN, 0.05)),
            bf16_entry(&format!("{p}.mlp.up_proj.weight"), &[INTER, HIDDEN], &rng.vec(INTER * HIDDEN, 0.05)),
            bf16_entry(&format!("{p}.mlp.down_proj.weight"), &[HIDDEN, INTER], &rng.vec(HIDDEN * INTER, 0.05)),
        ]);
    }

    if with_bridge {
        entries.extend([
            i64_scalar_entry("draft_block_causal_version", 1),
            i64_scalar_entry("last_write_refine_version", 1),
            i64_scalar_entry("selected_prefix_bridge.independent_write_version", 1),
            i64_scalar_entry("selected_prefix_bridge.terminal_disabled_version", 1),
            bf16_entry("selected_prefix_bridge.codebook", &[VOCAB, code_size], &rng.vec(VOCAB * code_size, 0.1)),
            bf16_entry("selected_prefix_bridge.read_weight", &[code_size, HIDDEN], &rng.vec(code_size * HIDDEN, 0.05)),
            bf16_entry("selected_prefix_bridge.early_read_weights", &[draft_layers - 1, code_size, HIDDEN], &rng.vec((draft_layers - 1) * code_size * HIDDEN, 0.05)),
            bf16_entry("selected_prefix_bridge.feature_write_weights", &[draft_layers, HIDDEN, rank], &rng.vec(draft_layers * HIDDEN * rank, 0.05)),
            bf16_entry("selected_prefix_bridge.gate.weight", &[rank, code_size], &rng.vec(rank * code_size, 0.08)),
            bf16_entry("selected_prefix_bridge.gate.bias", &[rank], &rng.vec(rank, 0.1)),
            bf16_entry("selected_prefix_bridge.predecessor_gate.weight", &[rank, code_size], &rng.vec(rank * code_size, 0.08)),
            bf16_entry("selected_prefix_bridge.prev.weight", &[rank, code_size], &rng.vec(rank * code_size, 0.08)),
            bf16_entry("selected_prefix_bridge.out.weight", &[code_size, rank], &vec![0.0; code_size * rank]),
            bf16_entry("selected_prefix_bridge.terminal_residual_embedding", &[VOCAB, rt], &rng.vec(VOCAB * rt, 0.08)),
            bf16_entry("selected_prefix_bridge.terminal_residual_projection", &[rank, rt], &rng.vec(rank * rt, 0.08)),
        ]);
    }

    write_safetensors(&dir.join("model.safetensors"), &entries);
}

fn run_generate(rt: &mut InferenceRuntime, id: &str, prompt: &[i64], tokens: usize, params: SamplingParams) -> Vec<i64> {
    rt.submit_with_sampling(id.into(), prompt.to_vec(), tokens, params).unwrap();
    let mut out = Vec::new();
    let mut guard = 0usize;
    while out.len() < tokens {
        guard += 1;
        assert!(guard < 5_000, "stuck waiting for tokens on {id}");
        for e in rt.step().unwrap() {
            if e.request_id == id {
                out.push(e.token_id);
            }
        }
    }
    out
}

fn load_direct(dir: &Path) -> GgufModel {
    let prepared = crate::import::prepare(
        dir.to_str().unwrap(),
        &crate::import::ImportOptions {
            direct: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(is_direct(&prepared), "{} must be direct-loadable", prepared.display());
    let mapping = open_model(&prepared).unwrap();
    GgufModel::from_mapping(mapping.with_tensor_aliases(&[]).unwrap(), 64, 4, 2).unwrap()
}

#[test]
fn dspine_and_dflash_reproduce_plain_greedy_on_qwen3_and_qwen35_hybrid() {
    let dir = scratch("dspine-e2e");
    let target_qwen3 = build_gguf(&dir.join("t-qwen3"), Arch::Qwen3, Encoding::F32, 0x101);
    let target_qwen35 = build_gguf(&dir.join("t-qwen35"), Arch::Qwen35, Encoding::F32, 0x202);

    let dspine_ckpt = dir.join("dspine-ckpt");
    write_draft_checkpoint(&dspine_ckpt, true, 0x303);
    let dflash_ckpt = dir.join("dflash-ckpt");
    write_draft_checkpoint(&dflash_ckpt, false, 0x404);

    // Also verify explicit GGUF conversion for DSpine without tokenizer.json
    let dspine_gguf = dir.join("dspine.gguf");
    convert_checkpoint(&dspine_ckpt, &dspine_gguf, ConvertOptions {
        output_dtype: OutputDType::Preserve,
        overwrite: true,
        spec_head: None,
    }).unwrap();
    let dspine_gguf_model = GgufModel::load(dspine_gguf.to_str().unwrap(), 64, 4, 2).unwrap();
    assert_eq!(dspine_gguf_model.config().arch, "dspine");

    let prompt: Vec<i64> = vec![1, 5, 12, 23, 34, 45, 56, 67, 12, 23];
    let steps = 16usize;

    // 1. Qwen3 target + direct-loaded DSpineDraftModel (greedy & sampled)
    {
        let mut plain_rt = InferenceRuntime::with_speculative(
            GgufModel::load(target_qwen3.to_str().unwrap(), 64, 4, 2).unwrap(),
            4, 64, Vec::new(), 0, 8,
        ).unwrap();
        plain_rt.attach_compute("cpu").unwrap();
        let plain = run_generate(&mut plain_rt, "plain", &prompt, steps, SamplingParams::default());

        let mut rt = InferenceRuntime::with_speculative(
            GgufModel::load(target_qwen3.to_str().unwrap(), 64, 4, 2).unwrap(),
            4, 64, Vec::new(), 0, 8,
        ).unwrap();
        rt.enable_dspine(load_direct(&dspine_ckpt), None).unwrap();
        assert_eq!(rt.speculative_status().0, SpeculativeMethod::Dspine);
        assert_eq!(rt.speculative_status().1, 3); // block_size(4) - 1
        rt.attach_compute("cpu").unwrap();
        let spec = run_generate(&mut rt, "dspine-greedy", &prompt, steps, SamplingParams::default());
        assert_eq!(spec, plain, "DSpine greedy diverged on Qwen3 target");

        let sampled_params = SamplingParams {
            temperature: 0.8,
            top_k: 8,
            top_p: 0.9,
            min_p: 0.05,
            seed: Some(777),
            logprobs: Some(3),
            ..Default::default()
        };
        let sampled = run_generate(&mut rt, "dspine-sampled", &prompt, 10, sampled_params);
        assert_eq!(sampled.len(), 10);
        assert!(sampled.iter().all(|&t| (0..VOCAB as i64).contains(&t)));
    }

    // 2. Qwen3.5 hybrid target (GatedDeltaNet + Attention, rollback on reject) + DSpineDraftModel
    {
        let mut plain_rt = InferenceRuntime::with_speculative(
            GgufModel::load(target_qwen35.to_str().unwrap(), 64, 4, 2).unwrap(),
            4, 64, Vec::new(), 0, 8,
        ).unwrap();
        plain_rt.attach_compute("cpu").unwrap();
        let plain = run_generate(&mut plain_rt, "plain35", &prompt, steps, SamplingParams::default());

        let mut rt = InferenceRuntime::with_speculative(
            GgufModel::load(target_qwen35.to_str().unwrap(), 64, 4, 2).unwrap(),
            4, 64, Vec::new(), 0, 8,
        ).unwrap();
        rt.enable_dspine(dspine_gguf_model, Some(3)).unwrap();
        rt.attach_compute("cpu").unwrap();
        let spec = run_generate(&mut rt, "dspine35", &prompt, steps, SamplingParams::default());
        assert_eq!(spec, plain, "DSpine greedy diverged on Qwen3.5 GatedDeltaNet hybrid target");
    }

    // 3. Qwen3 target + direct-loaded DFlashDraftModel (no bridge, block_causal=false)
    {
        let mut plain_rt = InferenceRuntime::with_speculative(
            GgufModel::load(target_qwen3.to_str().unwrap(), 64, 4, 2).unwrap(),
            4, 64, Vec::new(), 0, 8,
        ).unwrap();
        plain_rt.attach_compute("cpu").unwrap();
        let plain = run_generate(&mut plain_rt, "plain-df", &prompt, steps, SamplingParams::default());

        let mut rt = InferenceRuntime::with_speculative(
            GgufModel::load(target_qwen3.to_str().unwrap(), 64, 4, 2).unwrap(),
            4, 64, Vec::new(), 0, 8,
        ).unwrap();
        rt.enable_dspine(load_direct(&dflash_ckpt), Some(2)).unwrap();
        assert_eq!(rt.speculative_status().0, SpeculativeMethod::Dflash);
        assert_eq!(rt.speculative_status().1, 2);
        rt.attach_compute("cpu").unwrap();
        let spec = run_generate(&mut rt, "dflash", &prompt, steps, SamplingParams::default());
        assert_eq!(spec, plain, "DFlash greedy diverged on Qwen3 target");
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn bailing_moe_gqa_checkpoint_converts_and_runs_through_runtime() {
    let dir = scratch("bailing-gqa");
    let ckpt = dir.join("ckpt");
    fs::create_dir_all(&ckpt).unwrap();
    let mut rng = Rng(0x5a5a);
    let (h, heads, kvh, hd, ff, layers, vocab, experts, topk, sff) = (64, 4, 2, 16, 128, 2, 64, 4, 2, 64);

    let cfg = json!({
        "model_type": "bailing_moe",
        "hidden_size": h,
        "intermediate_size": ff,
        "moe_intermediate_size": ff,
        "shared_expert_intermediate_size": sff,
        "num_shared_experts": 1,
        "num_experts": experts,
        "num_experts_per_tok": topk,
        "first_k_dense_replace": 1,
        "scoring_func": "sigmoid",
        "norm_topk_prob": true,
        "routed_scaling_factor": 2.0,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kvh,
        "head_dim": hd,
        "vocab_size": vocab,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.5
    });
    fs::write(ckpt.join("config.json"), serde_json::to_vec_pretty(&cfg).unwrap()).unwrap();
    write_minimal_tokenizer(&ckpt, vocab);

    let qkv_out = (heads + 2 * kvh) * hd;
    let mut entries = vec![
        f32_entry("model.word_embeddings.weight", &[vocab, h], &rng.vec(vocab * h, 0.1)),
        f32_entry("model.norm.weight", &[h], &vec![1.0; h]),
        f32_entry("lm_head.weight", &[vocab, h], &rng.vec(vocab * h, 0.1)),
    ];
    for i in 0..layers {
        let p = format!("model.layers.{i}");
        entries.extend([
            f32_entry(&format!("{p}.input_layernorm.weight"), &[h], &vec![1.0; h]),
            f32_entry(&format!("{p}.post_attention_layernorm.weight"), &[h], &vec![1.0; h]),
            f32_entry(&format!("{p}.attention.query_key_value.weight"), &[qkv_out, h], &rng.vec(qkv_out * h, 0.08)),
            f32_entry(&format!("{p}.attention.dense.weight"), &[h, heads * hd], &rng.vec(h * heads * hd, 0.08)),
            f32_entry(&format!("{p}.attention.query_layernorm.weight"), &[hd], &vec![1.0; hd]),
            f32_entry(&format!("{p}.attention.key_layernorm.weight"), &[hd], &vec![1.0; hd]),
        ]);
        if i == 0 {
            entries.extend([
                f32_entry(&format!("{p}.mlp.gate_proj.weight"), &[ff, h], &rng.vec(ff * h, 0.08)),
                f32_entry(&format!("{p}.mlp.up_proj.weight"), &[ff, h], &rng.vec(ff * h, 0.08)),
                f32_entry(&format!("{p}.mlp.down_proj.weight"), &[h, ff], &rng.vec(h * ff, 0.08)),
            ]);
        } else {
            entries.extend([
                f32_entry(&format!("{p}.mlp.gate.weight"), &[experts, h], &rng.vec(experts * h, 0.1)),
                f32_entry(&format!("{p}.mlp.gate.expert_bias"), &[experts], &rng.vec(experts, 0.05)),
                f32_entry(&format!("{p}.mlp.shared_experts.gate_proj.weight"), &[sff, h], &rng.vec(sff * h, 0.08)),
                f32_entry(&format!("{p}.mlp.shared_experts.up_proj.weight"), &[sff, h], &rng.vec(sff * h, 0.08)),
                f32_entry(&format!("{p}.mlp.shared_experts.down_proj.weight"), &[h, sff], &rng.vec(h * sff, 0.08)),
            ]);
            for e in 0..experts {
                entries.extend([
                    f32_entry(&format!("{p}.mlp.experts.{e}.gate_proj.weight"), &[ff, h], &rng.vec(ff * h, 0.08)),
                    f32_entry(&format!("{p}.mlp.experts.{e}.up_proj.weight"), &[ff, h], &rng.vec(ff * h, 0.08)),
                    f32_entry(&format!("{p}.mlp.experts.{e}.down_proj.weight"), &[h, ff], &rng.vec(h * ff, 0.08)),
                ]);
            }
        }
    }
    write_safetensors(&ckpt.join("model.safetensors"), &entries);

    let model = load_direct(&ckpt);
    assert_eq!(model.config().arch, "bailing_moe");
    let mut rt = InferenceRuntime::with_speculative(model, 4, 64, Vec::new(), 0, 8).unwrap();
    rt.attach_compute("cpu").unwrap();
    let out = run_generate(&mut rt, "bailing", &[1, 3, 5, 7], 6, SamplingParams::default());
    assert_eq!(out.len(), 6);
    assert!(out.iter().all(|&t| (0..vocab as i64).contains(&t)));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn dspine_backend_device_path_matches_host_linear() {
    let dir = scratch("dspine-gpu-path");
    let target_gguf = build_gguf(&dir.join("t-qwen3"), Arch::Qwen3, Encoding::F32, 0x7711);
    let dspine_ckpt = dir.join("dspine-ckpt");
    write_draft_checkpoint(&dspine_ckpt, true, 0x8822);
    let dspine_gguf = dir.join("dspine.gguf");
    convert_checkpoint(&dspine_ckpt, &dspine_gguf, ConvertOptions {
        output_dtype: OutputDType::Preserve,
        overwrite: true,
        spec_head: None,
    }).unwrap();

    let target_model = GgufModel::load(target_gguf.to_str().unwrap(), 64, 4, 2).unwrap();
    let draft_model = GgufModel::load(dspine_gguf.to_str().unwrap(), 64, 4, 2).unwrap();

    let prompt: Vec<i64> = vec![1, 4, 9, 16, 25];
    let lengths: Vec<i64> = (1..=prompt.len() as i64).collect();
    let stride = target_model.table_stride();
    let table: Vec<i64> = (0..stride as i64).collect();
    let tables: Vec<i64> = (0..prompt.len()).flat_map(|_| table.iter().copied()).collect();

    // Host execution baseline
    let host_dec = crate::rust_decoder::RustDecoder::load(&target_model, "cpu").unwrap();
    let mut host_eng = crate::dspine::DSpineEngine::new(&draft_model).unwrap();
    host_eng.attach_compute("cpu", None).unwrap();
    assert!(!host_eng.uses_accelerator());
    host_dec.set_layer_tap(&host_eng.cfg.target_layer_ids);
    let mut hk = vec![0u16; target_model.key_buffer_len().unwrap()];
    let mut hv = vec![0u16; target_model.value_buffer_len().unwrap()];
    host_dec.forward(&prompt, &lengths, &tables, &mut hk, &mut hv, &[prompt.len() - 1], false).unwrap();
    let taps_host = host_dec.take_layer_tap();
    host_eng.observe_target_hidden(1, prompt.len(), &taps_host).unwrap();
    let mut hs = crate::sampling::SamplingState::new(&SamplingParams::default(), 0x42);
    let (tok_host, dist_host) = host_eng
        .propose(1, 25, 3, 8, &SamplingParams::default(), &mut hs, &prompt, &host_dec)
        .unwrap();

    // Device backend execution (`CpuBackend` implementing `Backend` + `Stream` + `BufferPool`)
    let backend: std::sync::Arc<dyn crate::backend::Backend> =
        std::sync::Arc::new(crate::backend::cpu::CpuBackend::new());
    let mut dev_dec = crate::rust_decoder::RustDecoder::load(&target_model, "cpu").unwrap();
    dev_dec.init_backend(std::sync::Arc::clone(&backend)).unwrap();
    let mut dev_eng = crate::dspine::DSpineEngine::new(&draft_model).unwrap();
    dev_eng.attach_compute("gpu", Some(backend)).unwrap();
    assert!(dev_dec.uses_accelerator());
    assert!(dev_eng.uses_accelerator());
    dev_dec.set_layer_tap(&dev_eng.cfg.target_layer_ids);
    let mut dk = vec![0u16; target_model.key_buffer_len().unwrap()];
    let mut dv = vec![0u16; target_model.value_buffer_len().unwrap()];
    dev_dec.forward(&prompt, &lengths, &tables, &mut dk, &mut dv, &[prompt.len() - 1], false).unwrap();
    let taps_dev = dev_dec.take_layer_tap();
    dev_eng.observe_target_hidden(1, prompt.len(), &taps_dev).unwrap();
    let mut ds = crate::sampling::SamplingState::new(&SamplingParams::default(), 0x42);
    let (tok_dev, dist_dev) = dev_eng
        .propose(1, 25, 3, 8, &SamplingParams::default(), &mut ds, &prompt, &dev_dec)
        .unwrap();

    assert_eq!(tok_dev, tok_host);
    assert_eq!(dist_dev.len(), dist_host.len());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn ungated_and_newly_registered_architectures_run_on_backend() {
    let dir = scratch("all-gpu-models");
    let mut rng = Rng(0x9191);
    let backend: std::sync::Arc<dyn crate::backend::Backend> =
        std::sync::Arc::new(crate::backend::cpu::CpuBackend::new());

    // 1. LFM2-MoE (`lfm2moe`): no longer rejected by RustDecoder hardcoded CPU gate
    {
        let ckpt = dir.join("lfm2moe");
        fs::create_dir_all(&ckpt).unwrap();
        let (h, heads, kvh, hd, ff, layers, vocab, experts, topk) = (64, 4, 2, 16, 128, 2, 64, 4, 2);
        let cfg = json!({
            "model_type": "lfm2_moe",
            "hidden_size": h,
            "intermediate_size": ff,
            "moe_intermediate_size": ff,
            "num_experts": experts,
            "num_experts_per_tok": topk,
            "num_dense_layers": 1,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kvh,
            "max_position_embeddings": 128,
            "vocab_size": vocab,
            "conv_L_cache": 3,
            "layer_types": ["conv", "full_attention"],
            "use_expert_bias": true,
            "norm_topk_prob": true,
            "routed_scaling_factor": 1.0,
            "norm_eps": 1e-5,
            "rope_theta": 10000.0
        });
        fs::write(ckpt.join("config.json"), serde_json::to_vec_pretty(&cfg).unwrap()).unwrap();
        write_minimal_tokenizer(&ckpt, vocab);
        let mut entries = vec![
            f32_entry("model.embed_tokens.weight", &[vocab, h], &rng.vec(vocab * h, 0.1)),
            f32_entry("model.embedding_norm.weight", &[h], &vec![1.0; h]),
            f32_entry("lm_head.weight", &[vocab, h], &rng.vec(vocab * h, 0.1)),
        ];
        // layer 0: conv + dense MLP; layer 1: full_attention + MoE
        entries.extend([
            f32_entry("model.layers.0.operator_norm.weight", &[h], &vec![1.0; h]),
            f32_entry("model.layers.0.ffn_norm.weight", &[h], &vec![1.0; h]),
            f32_entry("model.layers.0.conv.in_proj.weight", &[3 * h, h], &rng.vec(3 * h * h, 0.08)),
            f32_entry("model.layers.0.conv.conv.weight", &[h, 1, 3], &rng.vec(h * 3, 0.1)),
            f32_entry("model.layers.0.conv.out_proj.weight", &[h, h], &rng.vec(h * h, 0.08)),
            f32_entry("model.layers.0.feed_forward.w1.weight", &[ff, h], &rng.vec(ff * h, 0.08)),
            f32_entry("model.layers.0.feed_forward.w3.weight", &[ff, h], &rng.vec(ff * h, 0.08)),
            f32_entry("model.layers.0.feed_forward.w2.weight", &[h, ff], &rng.vec(h * ff, 0.08)),
            f32_entry("model.layers.1.operator_norm.weight", &[h], &vec![1.0; h]),
            f32_entry("model.layers.1.ffn_norm.weight", &[h], &vec![1.0; h]),
            f32_entry("model.layers.1.self_attn.q_proj.weight", &[heads * hd, h], &rng.vec(heads * hd * h, 0.08)),
            f32_entry("model.layers.1.self_attn.k_proj.weight", &[kvh * hd, h], &rng.vec(kvh * hd * h, 0.08)),
            f32_entry("model.layers.1.self_attn.v_proj.weight", &[kvh * hd, h], &rng.vec(kvh * hd * h, 0.08)),
            f32_entry("model.layers.1.self_attn.out_proj.weight", &[h, heads * hd], &rng.vec(h * heads * hd, 0.08)),
            f32_entry("model.layers.1.self_attn.q_layernorm.weight", &[hd], &vec![1.0; hd]),
            f32_entry("model.layers.1.self_attn.k_layernorm.weight", &[hd], &vec![1.0; hd]),
            f32_entry("model.layers.1.feed_forward.gate.weight", &[experts, h], &rng.vec(experts * h, 0.1)),
            f32_entry("model.layers.1.feed_forward.expert_bias", &[experts], &rng.vec(experts, 0.05)),
        ]);
        for e in 0..experts {
            entries.extend([
                f32_entry(&format!("model.layers.1.feed_forward.experts.{e}.w1.weight"), &[ff, h], &rng.vec(ff * h, 0.08)),
                f32_entry(&format!("model.layers.1.feed_forward.experts.{e}.w3.weight"), &[ff, h], &rng.vec(ff * h, 0.08)),
                f32_entry(&format!("model.layers.1.feed_forward.experts.{e}.w2.weight"), &[h, ff], &rng.vec(h * ff, 0.08)),
            ]);
        }
        write_safetensors(&ckpt.join("model.safetensors"), &entries);
        let model = load_direct(&ckpt);
        assert_eq!(model.config().arch, "lfm2moe");
        let mut dec = crate::rust_decoder::RustDecoder::load(&model, "cpu").unwrap();
        dec.init_backend(std::sync::Arc::clone(&backend)).unwrap();
        assert!(dec.uses_accelerator());
        let mut key = vec![0u16; model.key_buffer_len().unwrap()];
        let mut value = vec![0u16; model.value_buffer_len().unwrap()];
        let table: Vec<i64> = (0..model.table_stride() as i64).collect();
        let logits = dec.forward(&[1], &[1], &table, &mut key, &mut value, &[0], false).unwrap();
        assert_eq!(logits.len(), vocab);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    // 2. DBRX (`dbrx`): LayerNorm + qkv_clip + MoE runs on Backend cleanly
    {
        let ckpt = dir.join("dbrx");
        fs::create_dir_all(&ckpt).unwrap();
        let (h, heads, kvh, hd, ff, layers, vocab, experts, topk) = (64, 4, 2, 16, 128, 1, 64, 4, 2);
        let cfg = json!({
            "model_type": "dbrx",
            "d_model": h,
            "n_layers": layers,
            "n_heads": heads,
            "max_seq_len": 128,
            "vocab_size": vocab,
            "attn_config": { "kv_n_heads": kvh, "clip_qkv": 8.0, "rope_theta": 10000.0 },
            "ffn_config": {
                "ffn_hidden_size": ff,
                "moe_num_experts": experts,
                "moe_top_k": topk,
                "ffn_act_fn": { "name": "silu" }
            }
        });
        fs::write(ckpt.join("config.json"), serde_json::to_vec_pretty(&cfg).unwrap()).unwrap();
        write_minimal_tokenizer(&ckpt, vocab);
        let qkv_out = (heads + 2 * kvh) * hd;
        let entries = vec![
            f32_entry("transformer.wte.weight", &[vocab, h], &rng.vec(vocab * h, 0.1)),
            f32_entry("transformer.norm_f.weight", &[h], &vec![1.0; h]),
            f32_entry("lm_head.weight", &[vocab, h], &rng.vec(vocab * h, 0.1)),
            f32_entry("transformer.blocks.0.norm_attn_norm.norm_1.weight", &[h], &vec![1.0; h]),
            f32_entry("transformer.blocks.0.norm_attn_norm.norm_2.weight", &[h], &vec![1.0; h]),
            f32_entry("transformer.blocks.0.norm_attn_norm.attn.Wqkv.weight", &[qkv_out, h], &rng.vec(qkv_out * h, 0.08)),
            f32_entry("transformer.blocks.0.norm_attn_norm.attn.out_proj.weight", &[h, heads * hd], &rng.vec(h * heads * hd, 0.08)),
            f32_entry("transformer.blocks.0.ffn.router.layer.weight", &[experts, h], &rng.vec(experts * h, 0.1)),
            f32_entry("transformer.blocks.0.ffn.experts.mlp.w1", &[experts * ff, h], &rng.vec(experts * ff * h, 0.08)),
            f32_entry("transformer.blocks.0.ffn.experts.mlp.v1", &[experts * ff, h], &rng.vec(experts * ff * h, 0.08)),
            f32_entry("transformer.blocks.0.ffn.experts.mlp.w2", &[experts * ff, h], &rng.vec(experts * ff * h, 0.08)),
        ];
        write_safetensors(&ckpt.join("model.safetensors"), &entries);
        let model = load_direct(&ckpt);
        assert_eq!(model.config().arch, "dbrx");
        // Verify the hardcoded `"dbrx requires --device cpu"` gate is gone: `auto` loads cleanly and
        // `init_backend` reaches stream capability negotiation (`CpuBackend` omits `supports_arch_ops`,
        // while CUDA/Metal/Vulkan/wgpu all enable `supports_arch_ops` + `supports_layer_norm`).
        if let Err(err) = crate::rust_decoder::RustDecoder::load(&model, "cpu")
            .unwrap()
            .init_backend(std::sync::Arc::clone(&backend))
        {
            assert!(
                !err.contains("accelerator integration is pending") && err.contains("QKV clipping"),
                "unexpected dbrx backend error: {err}"
            );
        }
        let dec = crate::rust_decoder::RustDecoder::load(&model, "auto").unwrap();
        let mut key = vec![0u16; model.key_buffer_len().unwrap()];
        let mut value = vec![0u16; model.value_buffer_len().unwrap()];
        let table: Vec<i64> = (0..model.table_stride() as i64).collect();
        let logits = dec.forward(&[1], &[1], &table, &mut key, &mut value, &[0], false).unwrap();
        assert_eq!(logits.len(), vocab);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    // 3. Newly enabled Llama/Qwen architectures converted by plan.rs (e.g. `telechat`, `step3`)
    for (model_type, is_moe) in [("telechat", false), ("step3", true)] {
        let ckpt = dir.join(model_type);
        fs::create_dir_all(&ckpt).unwrap();
        let (h, heads, kvh, hd, ff, layers, vocab, experts, topk) = (64, 4, 2, 16, 128, 1, 64, 4, 2);
        let cfg = if is_moe {
            json!({
                "model_type": model_type,
                "hidden_size": h,
                "intermediate_size": ff,
                "moe_intermediate_size": ff,
                "num_experts": experts,
                "num_experts_per_tok": topk,
                "num_hidden_layers": layers,
                "num_attention_heads": heads,
                "num_key_value_heads": kvh,
                "head_dim": hd,
                "vocab_size": vocab,
                "max_position_embeddings": 128,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0
            })
        } else {
            json!({
                "model_type": model_type,
                "hidden_size": h,
                "intermediate_size": ff,
                "num_hidden_layers": layers,
                "num_attention_heads": heads,
                "num_key_value_heads": kvh,
                "head_dim": hd,
                "vocab_size": vocab,
                "max_position_embeddings": 128,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0
            })
        };
        fs::write(ckpt.join("config.json"), serde_json::to_vec_pretty(&cfg).unwrap()).unwrap();
        write_minimal_tokenizer(&ckpt, vocab);
        let mut entries = vec![
            f32_entry("model.embed_tokens.weight", &[vocab, h], &rng.vec(vocab * h, 0.1)),
            f32_entry("model.norm.weight", &[h], &vec![1.0; h]),
            f32_entry("lm_head.weight", &[vocab, h], &rng.vec(vocab * h, 0.1)),
            f32_entry("model.layers.0.input_layernorm.weight", &[h], &vec![1.0; h]),
            f32_entry("model.layers.0.post_attention_layernorm.weight", &[h], &vec![1.0; h]),
            f32_entry("model.layers.0.self_attn.q_proj.weight", &[heads * hd, h], &rng.vec(heads * hd * h, 0.08)),
            f32_entry("model.layers.0.self_attn.k_proj.weight", &[kvh * hd, h], &rng.vec(kvh * hd * h, 0.08)),
            f32_entry("model.layers.0.self_attn.v_proj.weight", &[kvh * hd, h], &rng.vec(kvh * hd * h, 0.08)),
            f32_entry("model.layers.0.self_attn.o_proj.weight", &[h, heads * hd], &rng.vec(h * heads * hd, 0.08)),
        ];
        if is_moe {
            entries.push(f32_entry("model.layers.0.mlp.gate.weight", &[experts, h], &rng.vec(experts * h, 0.1)));
            for e in 0..experts {
                entries.extend([
                    f32_entry(&format!("model.layers.0.mlp.experts.{e}.gate_proj.weight"), &[ff, h], &rng.vec(ff * h, 0.08)),
                    f32_entry(&format!("model.layers.0.mlp.experts.{e}.up_proj.weight"), &[ff, h], &rng.vec(ff * h, 0.08)),
                    f32_entry(&format!("model.layers.0.mlp.experts.{e}.down_proj.weight"), &[h, ff], &rng.vec(h * ff, 0.08)),
                ]);
            }
        } else {
            entries.extend([
                f32_entry("model.layers.0.mlp.gate_proj.weight", &[ff, h], &rng.vec(ff * h, 0.08)),
                f32_entry("model.layers.0.mlp.up_proj.weight", &[ff, h], &rng.vec(ff * h, 0.08)),
                f32_entry("model.layers.0.mlp.down_proj.weight", &[h, ff], &rng.vec(h * ff, 0.08)),
            ]);
        }
        write_safetensors(&ckpt.join("model.safetensors"), &entries);
        let model = load_direct(&ckpt);
        assert_eq!(model.config().arch, model_type);
        let mut dec = crate::rust_decoder::RustDecoder::load(&model, "cpu").unwrap();
        dec.init_backend(std::sync::Arc::clone(&backend)).unwrap();
        assert!(dec.uses_accelerator());
        let mut key = vec![0u16; model.key_buffer_len().unwrap()];
        let mut value = vec![0u16; model.value_buffer_len().unwrap()];
        let table: Vec<i64> = (0..model.table_stride() as i64).collect();
        let logits = dec.forward(&[1], &[1], &table, &mut key, &mut value, &[0], false).unwrap();
        assert_eq!(logits.len(), vocab);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn dspine_device_snapshot_restore_preserves_tokens() {
    let dir = scratch("dspine-dev-snap");
    let target_gguf = build_gguf(&dir.join("t-qwen3"), Arch::Qwen3, Encoding::F32, 0x3311);
    let dspine_ckpt = dir.join("dspine-ckpt");
    write_draft_checkpoint(&dspine_ckpt, true, 0x4422);
    let dspine_gguf = dir.join("dspine.gguf");
    convert_checkpoint(&dspine_ckpt, &dspine_gguf, ConvertOptions {
        output_dtype: OutputDType::Preserve,
        overwrite: true,
        spec_head: None,
    }).unwrap();

    let target_model = GgufModel::load(target_gguf.to_str().unwrap(), 64, 4, 2).unwrap();
    let draft_model = GgufModel::load(dspine_gguf.to_str().unwrap(), 64, 4, 2).unwrap();
    let backend: std::sync::Arc<dyn crate::backend::Backend> =
        std::sync::Arc::new(crate::backend::cpu::CpuBackend::new());
    let mut dec = crate::rust_decoder::RustDecoder::load(&target_model, "cpu").unwrap();
    dec.init_backend(std::sync::Arc::clone(&backend)).unwrap();
    let mut eng = crate::dspine::DSpineEngine::new(&draft_model).unwrap();
    eng.attach_compute("gpu", Some(backend)).unwrap();

    let prompt: Vec<i64> = vec![1, 4, 9, 16];
    let lengths: Vec<i64> = (1..=prompt.len() as i64).collect();
    let table: Vec<i64> = (0..target_model.table_stride() as i64).collect();
    let tables: Vec<i64> = (0..prompt.len()).flat_map(|_| table.iter().copied()).collect();
    dec.set_layer_tap(&eng.cfg.target_layer_ids);
    let mut dk = vec![0u16; target_model.key_buffer_len().unwrap()];
    let mut dv = vec![0u16; target_model.value_buffer_len().unwrap()];
    dec.forward(&prompt, &lengths, &tables, &mut dk, &mut dv, &[prompt.len() - 1], false).unwrap();
    let taps = dec.take_layer_tap();
    eng.observe_target_hidden(1, prompt.len(), &taps).unwrap();

    let params = SamplingParams { temperature: 0.7, top_k: 8, ..SamplingParams::default() };
    let mut s1 = crate::sampling::SamplingState::new(&params, 0x99);
    let (tok1, _) = eng.propose(1, 16, 3, 8, &params, &mut s1, &prompt, &dec).unwrap();
    assert!(!tok1.is_empty());

    let snap = eng.snapshot_seq(1);
    let s_saved = s1.clone();
    let mut s_trial = s1.clone();
    let hist2 = [1, 4, 9, 16, tok1[0]];
    let (tok2_a, _) = eng.propose(1, tok1[0], 3, 7, &params, &mut s_trial, &hist2, &dec).unwrap();
    eng.restore_seq(1, snap);
    let mut s_replay = s_saved;
    let (tok2_b, _) = eng.propose(1, tok1[0], 3, 7, &params, &mut s_replay, &hist2, &dec).unwrap();
    assert_eq!(tok2_a, tok2_b);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn step37_plain_moe_and_value_moe_match_host_on_backend() {
    use fllm_checkpoint::gguf::{write_to, Meta, OutTensor};
    use fllm_checkpoint::safetensors::Payload;
    let dir = scratch("gpu-gaps-parity");
    let mut rng = Rng(0xABCD);
    let backend: std::sync::Arc<dyn crate::backend::Backend> =
        std::sync::Arc::new(crate::backend::cpu::CpuBackend::new());
    let f32_t = |name: &str, dims: Vec<u64>, data: &[f32]| {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let data_len = bytes.len() as u64;
        OutTensor {
            name: name.to_owned(),
            dims,
            source_dtype: "F32".into(),
            dtype: "F32".into(),
            data: Payload::Owned(std::sync::Arc::new(Vec::new()), 0..0),
            packed: None,
            data_len,
            source_for_output: None,
            add: 0.0,
            owned: Some(bytes),
            parts: Vec::new(),
        }
    };

    // 1. Step-3.7-Flash (`step3p5` -> `laguna`) with per-layer SwiGLU clamps (`swiglu_limits`, `swiglu_limits_shared`)
    {
        let ckpt = dir.join("step37");
        fs::create_dir_all(&ckpt).unwrap();
        let (h, heads, kvh, hd, ff, sff, layers, vocab, experts, topk) = (32, 4, 2, 8, 64, 32, 2, 48, 4, 2);
        let cfg = json!({
            "model_type": "step3p5",
            "hidden_size": h,
            "intermediate_size": ff,
            "moe_intermediate_size": ff,
            "shared_expert_intermediate_size": sff,
            "num_experts": experts,
            "num_experts_per_tok": topk,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kvh,
            "head_dim": hd,
            "vocab_size": vocab,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "mlp_layer_types": ["dense", "sparse"],
            "swiglu_limits": [0.0, 5.5],
            "swiglu_limits_shared": [0.0, 3.5]
        });
        fs::write(ckpt.join("config.json"), serde_json::to_vec_pretty(&cfg).unwrap()).unwrap();
        write_minimal_tokenizer(&ckpt, vocab);
        let mut entries = vec![
            f32_entry("model.embed_tokens.weight", &[vocab, h], &rng.vec(vocab * h, 0.1)),
            f32_entry("model.norm.weight", &[h], &vec![0.0; h]),
            f32_entry("lm_head.weight", &[vocab, h], &rng.vec(vocab * h, 0.1)),
        ];
        for l in 0..layers {
            let p = format!("model.layers.{l}");
            entries.extend([
                f32_entry(&format!("{p}.input_layernorm.weight"), &[h], &vec![0.0; h]),
                f32_entry(&format!("{p}.post_attention_layernorm.weight"), &[h], &vec![0.0; h]),
                f32_entry(&format!("{p}.self_attn.q_proj.weight"), &[heads * hd, h], &rng.vec(heads * hd * h, 0.08)),
                f32_entry(&format!("{p}.self_attn.k_proj.weight"), &[kvh * hd, h], &rng.vec(kvh * hd * h, 0.08)),
                f32_entry(&format!("{p}.self_attn.v_proj.weight"), &[kvh * hd, h], &rng.vec(kvh * hd * h, 0.08)),
                f32_entry(&format!("{p}.self_attn.o_proj.weight"), &[h, heads * hd], &rng.vec(h * heads * hd, 0.08)),
                f32_entry(&format!("{p}.self_attn.q_norm.weight"), &[hd], &vec![0.0; hd]),
                f32_entry(&format!("{p}.self_attn.k_norm.weight"), &[hd], &vec![0.0; hd]),
                f32_entry(&format!("{p}.self_attn.g_proj.weight"), &[heads, h], &rng.vec(heads * h, 0.08)),
            ]);
            if l == 0 {
                entries.extend([
                    f32_entry(&format!("{p}.mlp.gate_proj.weight"), &[ff, h], &rng.vec(ff * h, 0.08)),
                    f32_entry(&format!("{p}.mlp.up_proj.weight"), &[ff, h], &rng.vec(ff * h, 0.08)),
                    f32_entry(&format!("{p}.mlp.down_proj.weight"), &[h, ff], &rng.vec(h * ff, 0.08)),
                ]);
            } else {
                entries.extend([
                    f32_entry(&format!("{p}.moe.gate.weight"), &[experts, h], &rng.vec(experts * h, 0.1)),
                    f32_entry(&format!("{p}.moe.router_bias"), &[experts], &rng.vec(experts, 0.05)),
                    f32_entry(&format!("{p}.share_expert.gate_proj.weight"), &[sff, h], &rng.vec(sff * h, 0.08)),
                    f32_entry(&format!("{p}.share_expert.up_proj.weight"), &[sff, h], &rng.vec(sff * h, 0.08)),
                    f32_entry(&format!("{p}.share_expert.down_proj.weight"), &[h, sff], &rng.vec(h * sff, 0.08)),
                ]);
                for e in 0..experts {
                    entries.extend([
                        f32_entry(&format!("{p}.mlp.experts.{e}.gate_proj.weight"), &[ff, h], &rng.vec(ff * h, 0.08)),
                        f32_entry(&format!("{p}.mlp.experts.{e}.up_proj.weight"), &[ff, h], &rng.vec(ff * h, 0.08)),
                        f32_entry(&format!("{p}.mlp.experts.{e}.down_proj.weight"), &[h, ff], &rng.vec(h * ff, 0.08)),
                    ]);
                }
            }
        }
        write_safetensors(&ckpt.join("model.safetensors"), &entries);
        let model = load_direct(&ckpt);
        let moe_cfg = model.config().moe.as_ref().unwrap();
        assert_eq!(moe_cfg.routed_clamp(0), None);
        assert_eq!(moe_cfg.routed_clamp(1), Some(5.5));
        assert_eq!(moe_cfg.shared_clamp(1), Some(3.5));

        let host = crate::rust_decoder::RustDecoder::load(&model, "cpu").unwrap();
        let mut dev = crate::rust_decoder::RustDecoder::load(&model, "cpu").unwrap();
        dev.init_backend(std::sync::Arc::clone(&backend)).unwrap();
        assert!(dev.uses_accelerator());

        let mut hk = vec![0u16; model.key_buffer_len().unwrap()];
        let mut hv = vec![0u16; model.value_buffer_len().unwrap()];
        let mut dk = hk.clone();
        let mut dv = hv.clone();
        let table: Vec<i64> = (0..model.table_stride() as i64).collect();
        let hl = host.forward(&[1], &[1], &table, &mut hk, &mut hv, &[0], false).unwrap();
        let dl = dev.forward(&[1], &[1], &table, &mut dk, &mut dv, &[0], false).unwrap();
        let diff = hl.iter().zip(&dl).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(diff < 1e-4, "step37 device vs host diff={diff}");
    }

    // 2. K2 Horizon `ValueMoe` (`attn_v_exps.weight`) GPU execution & host parity
    {
        let gguf_path = dir.join("value_moe.gguf");
        let (h, heads, kvh, hd, ff, vocab, ve_count, ve_used) = (32, 2, 2, 16, 64, 32, 4, 2);
        let vr = kvh * hd;
        let meta = vec![
            ("general.architecture".into(), Meta::Str("qwen2".into())),
            ("qwen2.embedding_length".into(), Meta::U32(h as u32)),
            ("qwen2.block_count".into(), Meta::U32(1)),
            ("qwen2.feed_forward_length".into(), Meta::U32(ff as u32)),
            ("qwen2.attention.head_count".into(), Meta::U32(heads as u32)),
            ("qwen2.attention.head_count_kv".into(), Meta::U32(kvh as u32)),
            ("qwen2.attention.key_length".into(), Meta::U32(hd as u32)),
            ("qwen2.attention.value_expert_count".into(), Meta::U32(ve_count as u32)),
            ("qwen2.attention.value_expert_used_count".into(), Meta::U32(ve_used as u32)),
            ("qwen2.rope.dimension_count".into(), Meta::U32(hd as u32)),
            ("qwen2.attention.layer_norm_rms_epsilon".into(), Meta::F32(1e-5)),
            ("qwen2.rope.freq_base".into(), Meta::F32(10000.0)),
        ];
        let tensors = vec![
            f32_t("token_embd.weight", vec![h as u64, vocab as u64], &rng.vec(vocab * h, 0.1)),
            f32_t("output_norm.weight", vec![h as u64], &vec![1.0; h]),
            f32_t("output.weight", vec![h as u64, vocab as u64], &rng.vec(vocab * h, 0.1)),
            f32_t("blk.0.attn_norm.weight", vec![h as u64], &vec![1.0; h]),
            f32_t("blk.0.ffn_norm.weight", vec![h as u64], &vec![1.0; h]),
            f32_t("blk.0.attn_q.weight", vec![h as u64, (heads * hd) as u64], &rng.vec(heads * hd * h, 0.08)),
            f32_t("blk.0.attn_k.weight", vec![h as u64, (kvh * hd) as u64], &rng.vec(kvh * hd * h, 0.08)),
            f32_t("blk.0.attn_v.weight", vec![h as u64, (kvh * hd) as u64], &vec![0.0; kvh * hd * h]),
            f32_t("blk.0.attn_output.weight", vec![(heads * hd) as u64, h as u64], &rng.vec(h * heads * hd, 0.08)),
            f32_t("blk.0.attn_v_gate.weight", vec![h as u64, ve_count as u64], &rng.vec(ve_count * h, 0.1)),
            f32_t("blk.0.attn_v_exps.weight", vec![h as u64, (ve_count * vr) as u64], &rng.vec(ve_count * vr * h, 0.08)),
            f32_t("blk.0.ffn_gate.weight", vec![h as u64, ff as u64], &rng.vec(ff * h, 0.08)),
            f32_t("blk.0.ffn_up.weight", vec![h as u64, ff as u64], &rng.vec(ff * h, 0.08)),
            f32_t("blk.0.ffn_down.weight", vec![ff as u64, h as u64], &rng.vec(h * ff, 0.08)),
        ];
        {
            let mut file = fs::File::create(&gguf_path).unwrap();
            write_to(&mut file, &meta, &tensors).unwrap();
        }
        let model = GgufModel::load(gguf_path.to_str().unwrap(), 32, 2, 1).unwrap();
        let host = crate::rust_decoder::RustDecoder::load(&model, "cpu").unwrap();
        let mut dev = crate::rust_decoder::RustDecoder::load(&model, "cpu").unwrap();
        dev.init_backend(std::sync::Arc::clone(&backend)).unwrap();
        assert!(dev.uses_accelerator());

        let mut hk = vec![0u16; model.key_buffer_len().unwrap()];
        let mut hv = vec![0u16; model.value_buffer_len().unwrap()];
        let mut dk = hk.clone();
        let mut dv = hv.clone();
        let table: Vec<i64> = (0..model.table_stride() as i64).collect();
        let hl = host.forward(&[1], &[1], &table, &mut hk, &mut hv, &[0], false).unwrap();
        let dl = dev.forward(&[1], &[1], &table, &mut dk, &mut dv, &[0], false).unwrap();
        let diff = hl.iter().zip(&dl).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(diff < 1e-4, "value_moe device vs host diff={diff}");
    }

    // 3. Plain non-gated MoE (`nemotron_h_moe` / `e.gate = None`) GPU parity
    {
        let gguf_path = dir.join("plain_moe.gguf");
        let (h, heads, kvh, hd, ff, vocab, experts, topk) = (32, 2, 2, 16, 48, 32, 4, 2);
        let meta = vec![
            ("general.architecture".into(), Meta::Str("nemotron_h_moe".into())),
            ("nemotron_h_moe.embedding_length".into(), Meta::U32(h as u32)),
            ("nemotron_h_moe.block_count".into(), Meta::U32(1)),
            ("nemotron_h_moe.feed_forward_length".into(), Meta::U32(ff as u32)),
            ("nemotron_h_moe.expert_feed_forward_length".into(), Meta::U32(ff as u32)),
            ("nemotron_h_moe.expert_count".into(), Meta::U32(experts as u32)),
            ("nemotron_h_moe.expert_used_count".into(), Meta::U32(topk as u32)),
            ("nemotron_h_moe.attention.head_count".into(), Meta::U32(heads as u32)),
            ("nemotron_h_moe.attention.head_count_kv".into(), Meta::U32(kvh as u32)),
            ("nemotron_h_moe.attention.key_length".into(), Meta::U32(hd as u32)),
            ("nemotron_h_moe.rope.dimension_count".into(), Meta::U32(hd as u32)),
            ("nemotron_h_moe.attention.layer_norm_rms_epsilon".into(), Meta::F32(1e-5)),
            ("nemotron_h_moe.rope.freq_base".into(), Meta::F32(10000.0)),
            ("nemotron_h_moe.hybrid_override_pattern".into(), Meta::Str("E".into())),
        ];
        let tensors = vec![
            f32_t("token_embd.weight", vec![h as u64, vocab as u64], &rng.vec(vocab * h, 0.1)),
            f32_t("output_norm.weight", vec![h as u64], &vec![1.0; h]),
            f32_t("output.weight", vec![h as u64, vocab as u64], &rng.vec(vocab * h, 0.1)),
            f32_t("blk.0.attn_norm.weight", vec![h as u64], &vec![1.0; h]),
            f32_t("blk.0.ffn_gate_inp.weight", vec![h as u64, experts as u64], &rng.vec(experts * h, 0.1)),
            f32_t("blk.0.ffn_up_exps.weight", vec![h as u64, ff as u64, experts as u64], &rng.vec(experts * ff * h, 0.08)),
            f32_t("blk.0.ffn_down_exps.weight", vec![ff as u64, h as u64, experts as u64], &rng.vec(experts * h * ff, 0.08)),
        ];
        {
            let mut file = fs::File::create(&gguf_path).unwrap();
            write_to(&mut file, &meta, &tensors).unwrap();
        }
        let model = GgufModel::load(gguf_path.to_str().unwrap(), 32, 2, 1).unwrap();
        let host = crate::rust_decoder::RustDecoder::load(&model, "cpu").unwrap();
        let mut dev = crate::rust_decoder::RustDecoder::load(&model, "cpu").unwrap();
        dev.init_backend(std::sync::Arc::clone(&backend)).unwrap();
        assert!(dev.uses_accelerator());

        let mut hk = vec![0u16; model.key_buffer_len().unwrap()];
        let mut hv = vec![0u16; model.value_buffer_len().unwrap()];
        let mut dk = hk.clone();
        let mut dv = hv.clone();
        let table: Vec<i64> = (0..model.table_stride() as i64).collect();
        let hl = host.forward(&[1], &[1], &table, &mut hk, &mut hv, &[0], false).unwrap();
        let dl = dev.forward(&[1], &[1], &table, &mut dk, &mut dv, &[0], false).unwrap();
        let diff = hl.iter().zip(&dl).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(diff < 1e-4, "plain MoE device vs host diff={diff}");
    }

    let _ = fs::remove_dir_all(&dir);
}


