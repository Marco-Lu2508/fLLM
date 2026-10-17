use crate::{gguf, safetensors};
use std::{
    fs,
    io::Write,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

fn scratch() -> PathBuf {
    let n = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!("fllm-checkpoint-test-{}-{n}", std::process::id()));
    fs::create_dir_all(&p).unwrap();
    p
}
fn write_safe(path: &std::path::Path, header: &str, payload: &[u8]) {
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(payload);
    fs::write(path, out).unwrap();
}

#[test]
fn safetensors_checks_exact_shape_byte_count_and_reads_tensor() {
    let dir = scratch();
    let p = dir.join("model.safetensors");
    write_safe(
        &p,
        r#"{"weight":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#,
        &[1, 2, 3, 4],
    );
    let ts = safetensors::read_file(&p).unwrap();
    assert_eq!(ts[0].shape, vec![2]);
    assert_eq!(ts[0].payload.as_slice(), [1, 2, 3, 4]);
    write_safe(
        &p,
        r#"{"weight":{"dtype":"F16","shape":[3],"data_offsets":[0,4]}}"#,
        &[1, 2, 3, 4],
    );
    assert!(safetensors::read_file(&p)
        .unwrap_err()
        .contains("expected 6 bytes"));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn sharded_index_rejects_path_traversal_before_opening_file() {
    let dir = scratch();
    fs::write(
        dir.join("model.safetensors.index.json"),
        r#"{"weight_map":{"x":"../escape.safetensors"}}"#,
    )
    .unwrap();
    assert!(safetensors::read_checkpoint(&dir)
        .unwrap_err()
        .contains("unsafe shard path"));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn gguf_writer_emits_v3_header_and_aligned_tensor_payload() {
    let tensor = gguf::OutTensor {
        name: "w".into(),
        dims: vec![2],
        source_dtype: "F16".into(),
        dtype: "F16".into(),
        data: crate::safetensors::Payload::Owned(std::sync::Arc::new(vec![1, 2, 3, 4]), 0..4),
        packed: None,
        data_len: 4,
        source_for_output: None,
        add: 0.0,
        owned: None,
        parts: Vec::new(),
    };
    let mut bytes = Vec::new();
    gguf::write_to(
        &mut bytes,
        &[("general.alignment".into(), gguf::Meta::U32(32))],
        &[tensor],
    )
    .unwrap();
    assert_eq!(&bytes[..4], b"GGUF");
    assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 3);
    assert_eq!(bytes.len() % 32, 4); // aligned data base plus four-byte payload
}

#[test]
fn llama_qk_rows_convert_split_half_to_interleaved_rope_order() {
    assert_eq!(
        crate::convert::row_permutation("llama", "blk.0.attn_q.weight", 1, 1, 4).unwrap(),
        Some(vec![0, 2, 1, 3])
    );
    assert_eq!(
        crate::convert::row_permutation("qwen2", "blk.0.attn_q.weight", 1, 1, 4).unwrap(),
        None
    );
}

#[test]
fn restricted_torch_zip_reader_loads_float_state_dict_without_python() {
    use zip::{write::SimpleFileOptions, ZipWriter};
    let dir = scratch();
    let p = dir.join("pytorch_model.bin");
    let mut pickle = vec![0x80, 2, b'c'];
    pickle.extend_from_slice(b"collections\nOrderedDict\n");
    pickle.extend_from_slice(&[
        b'q', 0, b')', b'R', b'q', 1, b'(', b'X', 1, 0, 0, 0, b'w', b'c',
    ]);
    pickle.extend_from_slice(b"torch._utils\n_rebuild_tensor_v2\n");
    pickle.extend_from_slice(&[b'q', 2, b'(', b'(', b'X', 7, 0, 0, 0]);
    pickle.extend_from_slice(b"storage");
    pickle.push(b'c');
    pickle.extend_from_slice(b"torch\nFloatStorage\n");
    pickle.extend_from_slice(&[b'q', 3, b'X', 1, 0, 0, 0, b'0', b'X', 3, 0, 0, 0]);
    pickle.extend_from_slice(b"cpu");
    pickle.extend_from_slice(&[
        b'K', 2, b't', b'Q', b'K', 0, b'(', b'K', 2, b't', b'(', b'K', 1, b't', 0x89, b'h', 0,
        b')', b'R', b't', b'R', b'u', b'.',
    ]);
    let file = fs::File::create(&p).unwrap();
    let mut zip = ZipWriter::new(file);
    let opts = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    zip.start_file("archive/data.pkl", opts).unwrap();
    zip.write_all(&pickle).unwrap();
    zip.start_file("archive/data/0", opts).unwrap();
    zip.write_all(&[0, 0, 128, 63, 0, 0, 0, 64]).unwrap();
    zip.start_file("archive/byteorder", opts).unwrap();
    zip.write_all(b"little").unwrap();
    zip.finish().unwrap();
    let tensors = crate::torch_zip::read(&p).unwrap();
    assert_eq!(tensors.len(), 1);
    assert_eq!(tensors[0].name, "w");
    assert_eq!(tensors[0].dtype, "F32");
    assert_eq!(tensors[0].shape, vec![2]);
    assert_eq!(tensors[0].payload.as_slice(), [0, 0, 128, 63, 0, 0, 0, 64]);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn gptq_group_is_mapped_to_standard_hf_weight_and_decodes_in_bounded_chunks() {
    use crate::safetensors::{Payload, Tensor};
    use std::sync::Arc;
    let payload = |v: Vec<u8>| {
        let n = v.len();
        Payload::Owned(Arc::new(v), 0..n)
    };
    let word = |x: u32| x.to_le_bytes().to_vec();
    let mut qw = Vec::new();
    for _ in 0..8 {
        qw.extend(word(0x1111_1111));
    }
    let qz = word(0x4444_4444);
    let mut scales = Vec::new();
    for _ in 0..8 {
        scales.extend_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
    }
    let tensors = vec![
        Tensor {
            name: "model.layers.0.self_attn.q_proj.qweight".into(),
            dtype: "I32".into(),
            shape: vec![1, 8],
            payload: payload(qw),
        },
        Tensor {
            name: "model.layers.0.self_attn.q_proj.qzeros".into(),
            dtype: "I32".into(),
            shape: vec![1, 1],
            payload: payload(qz),
        },
        Tensor {
            name: "model.layers.0.self_attn.q_proj.scales".into(),
            dtype: "F16".into(),
            shape: vec![1, 8],
            payload: payload(scales),
        },
    ];
    let plan = crate::plan::Plan::from_config(&serde_json::json!({
        "model_type": "qwen2", "hidden_size": 8, "num_attention_heads": 2,
        "num_hidden_layers": 1, "intermediate_size": 16
    }))
    .unwrap();
    let (_, mut groups) = crate::convert::split_packed(
        tensors,
        crate::convert::Quantization::Gptq { group: 8 },
        None,
        &|n| {
            Ok(match plan.map(n)? {
                crate::plan::Mapped::Tensor { name, .. }
                | crate::plan::Mapped::Expert { name, .. } => Some(name),
                _ => None,
            })
        },
    )
    .unwrap();
    assert_eq!(groups.len(), 1);
    let (name, packed, dims, _) = groups.pop().unwrap();
    assert_eq!(name, "blk.0.attn_q.weight");
    assert_eq!(dims, vec![8, 8]);
    let mut chunk = [0f32; 5];
    packed.decode_f32_chunk(3, &mut chunk).unwrap();
    assert_eq!(chunk, [-4.0; 5]);
}

#[test]
fn tokenizer_pads_only_after_real_vocabulary_and_uses_config_special_ids() {
    let dir = scratch();
    let raw = r#"{"model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"d":3},"merges":[]},"added_tokens":[]}"#;
    fs::write(dir.join("tokenizer.json"), raw).unwrap();
    let config: serde_json::Value =
        serde_json::from_str(r#"{"bos_token_id":0,"eos_token_id":[1,3]}"#).unwrap();
    let (real, meta) = crate::convert::tokenizer(&dir, 6, &config).unwrap();
    assert_eq!(real, 4);
    let tokens = meta
        .iter()
        .find_map(|(k, v)| {
            if k == "tokenizer.ggml.tokens" {
                if let gguf::Meta::Strs(xs) = v {
                    Some(xs)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .unwrap();
    let kinds = meta
        .iter()
        .find_map(|(k, v)| {
            if k == "tokenizer.ggml.token_type" {
                if let gguf::Meta::U32s(xs) = v {
                    Some(xs)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .unwrap();
    assert_eq!(tokens.len(), 6);
    assert_ne!(tokens[4], tokens[5]);
    assert_eq!(&kinds[4..], [5, 5]);
    assert!(meta
        .iter()
        .any(|(k, v)| k == "tokenizer.ggml.bos_token_id" && matches!(v, gguf::Meta::U32(0))));
    assert!(meta
        .iter()
        .any(|(k, v)| k == "tokenizer.ggml.eos_token_id" && matches!(v, gguf::Meta::U32(1))));
    assert!(meta
        .iter()
        .any(|(k, v)| k == "tokenizer.huggingface.eos_token_ids"
            && matches!(v,gguf::Meta::U32s(ids) if ids==&vec![1,3])));
    assert!(meta.iter().any(
        |(k, v)| k == "tokenizer.huggingface.json" && matches!(v,gguf::Meta::Str(s) if s==raw)
    ));
    let bad = r#"{"model":{"type":"BPE","vocab":{"a":0,"b":2},"merges":[]}}"#;
    fs::write(dir.join("tokenizer.json"), bad).unwrap();
    assert!(crate::convert::tokenizer(&dir, 6, &config)
        .err()
        .unwrap()
        .contains("missing ids"));
    fs::write(dir.join("tokenizer.json"), raw).unwrap();
    let invalid_eos = serde_json::json!({"eos_token_id":[1,4]});
    assert!(crate::convert::tokenizer(&dir, 6, &invalid_eos)
        .err()
        .unwrap()
        .contains("outside tokenizer's real vocabulary"));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn unigram_padding_uses_low_finite_score_and_rejects_smaller_embedding_vocab() {
    let dir = scratch();
    fs::write(
        dir.join("tokenizer.json"),
        r#"{"model":{"type":"Unigram","vocab":[["a",-0.1],["b",-0.2]]}}"#,
    )
    .unwrap();
    let config = serde_json::json!({});
    let (_, meta) = crate::convert::tokenizer(&dir, 3, &config).unwrap();
    let scores = meta
        .iter()
        .find_map(|(k, v)| {
            if k == "tokenizer.ggml.scores" {
                if let gguf::Meta::F32s(xs) = v {
                    Some(xs)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .unwrap();
    assert_eq!(scores.len(), 3);
    assert_eq!(scores[2], -1.0e9);
    assert!(crate::convert::tokenizer(&dir, 1, &config)
        .err()
        .unwrap()
        .contains("exceeding config vocab_size"));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn empty_token_is_present_and_duplicate_empty_token_id_is_rejected() {
    let dir = scratch();
    fs::write(
        dir.join("tokenizer.json"),
        r#"{"model":{"type":"BPE","vocab":{"":0,"b":1},"merges":[]}}"#,
    )
    .unwrap();
    let config = serde_json::json!({});
    assert_eq!(crate::convert::tokenizer(&dir, 2, &config).unwrap().0, 2);
    fs::write(
        dir.join("tokenizer.json"),
        r#"{"model":{"type":"BPE","vocab":{"":0,"b":0},"merges":[]}}"#,
    )
    .unwrap();
    assert!(crate::convert::tokenizer(&dir, 2, &config)
        .err()
        .unwrap()
        .contains("duplicate tokenizer vocabulary id"));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn load_checkpoint_as_gguf_zero_copy_and_exact_parity() {
    let dir = scratch();
    let config = serde_json::json!({
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4,
        "intermediate_size": 4,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "num_hidden_layers": 1,
        "vocab_size": 4,
    });
    fs::write(dir.join("config.json"), config.to_string()).unwrap();
    fs::write(
        dir.join("tokenizer.json"),
        r#"{"model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"d":3},"merges":[]}}"#,
    )
    .unwrap();

    let tensors = [
        ("model.embed_tokens.weight", vec![4, 4]),
        ("model.layers.0.input_layernorm.weight", vec![4]),
        ("model.layers.0.self_attn.q_proj.weight", vec![4, 4]),
        ("model.layers.0.self_attn.k_proj.weight", vec![4, 4]),
        ("model.layers.0.self_attn.v_proj.weight", vec![4, 4]),
        ("model.layers.0.self_attn.o_proj.weight", vec![4, 4]),
        ("model.layers.0.post_attention_layernorm.weight", vec![4]),
        ("model.layers.0.mlp.gate_proj.weight", vec![4, 4]),
        ("model.layers.0.mlp.up_proj.weight", vec![4, 4]),
        ("model.layers.0.mlp.down_proj.weight", vec![4, 4]),
        ("model.norm.weight", vec![4]),
        ("lm_head.weight", vec![4, 4]),
    ];
    let mut header_map = serde_json::Map::new();
    let mut payload = Vec::new();
    let mut offset = 0usize;
    for (name, shape) in tensors {
        let num_elements: usize = shape.iter().product();
        let num_bytes = num_elements * 2;
        let tensor_bytes: Vec<u8> = (0..num_bytes).map(|i| ((offset + i) % 251 + 1) as u8).collect();
        header_map.insert(
            name.to_string(),
            serde_json::json!({
                "dtype": "F16",
                "shape": shape,
                "data_offsets": [offset, offset + num_bytes]
            }),
        );
        payload.extend_from_slice(&tensor_bytes);
        offset += num_bytes;
    }
    write_safe(
        &dir.join("model.safetensors"),
        &serde_json::Value::Object(header_map).to_string(),
        &payload,
    );

    let gguf_path = dir.join("converted.gguf");
    crate::convert_checkpoint(&dir, &gguf_path, crate::ConvertOptions::default()).unwrap();
    let ref_mapped = fllm_gguf_probe::MappedGguf::open(&gguf_path).unwrap();

    let direct_mapped =
        crate::load_checkpoint_as_gguf(&dir, crate::ConvertOptions::default()).unwrap();

    // Verify metadata parity
    assert_eq!(
        direct_mapped.index().metadata.get("general.architecture"),
        ref_mapped.index().metadata.get("general.architecture")
    );
    assert_eq!(direct_mapped.index().tensors.len(), ref_mapped.index().tensors.len());

    // Verify zero-copy for untouched tensor
    assert!(direct_mapped.extra_mappings().contains_key("token_embd.weight"));
    let mmap = direct_mapped.extra_mappings().get("token_embd.weight").unwrap();
    let embd_slice = direct_mapped.tensor_bytes("token_embd.weight").unwrap();
    let mmap_start = mmap.as_ptr() as usize;
    let mmap_end = mmap_start + mmap.len();
    let embd_start = embd_slice.as_ptr() as usize;
    assert!(embd_start >= mmap_start && embd_start + embd_slice.len() <= mmap_end);

    // Verify permuted tensor is in anon read-only memory, NOT inside safetensors file mmap
    let q_mmap = direct_mapped.extra_mappings().get("blk.0.attn_q.weight").unwrap();
    assert!(!std::sync::Arc::ptr_eq(mmap, q_mmap));
    let q_slice = direct_mapped.tensor_bytes("blk.0.attn_q.weight").unwrap();
    let q_start = q_slice.as_ptr() as usize;
    assert!(q_start < mmap_start || q_start >= mmap_end);

    // Verify bit-for-bit parity of all tensor bytes
    for t in direct_mapped.index().tensors.iter() {
        let b1 = direct_mapped.tensor_bytes(&t.name).unwrap();
        let b2 = ref_mapped.tensor_bytes(&t.name).unwrap();
        assert_eq!(b1, b2, "tensor byte mismatch for {}", t.name);
    }

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn spec_head_conversion_medusa_and_eagle() {
    use crate::{convert_checkpoint, ConvertOptions, OutputDType, SpecHeadKind};

    // 1. Test Medusa
    let dir = scratch();
    let p = dir.join("model.safetensors");
    let header = r#"{"medusa_heads.0.0.weight":{"dtype":"F16","shape":[2,2],"data_offsets":[0,8]},"medusa_heads.1.0.weight":{"dtype":"F16","shape":[2,2],"data_offsets":[8,16]}}"#;
    let payload = vec![0u8; 16];
    write_safe(&p, header, &payload);
    fs::write(dir.join("config.json"), r#"{"medusa_num_heads": 2}"#).unwrap();

    let out_medusa = dir.join("medusa.gguf");
    let rep = convert_checkpoint(
        &dir,
        &out_medusa,
        ConvertOptions {
            output_dtype: OutputDType::Preserve,
            overwrite: true,
            spec_head: Some(SpecHeadKind::Medusa),
        },
    )
    .unwrap();
    assert_eq!(rep.architecture, "spec_head_medusa");
    assert_eq!(rep.tensors_written, 2);
    assert!(out_medusa.is_file());

    // 2. Test Eagle / Eagle3
    let dir_eagle = scratch();
    let p_eagle = dir_eagle.join("model.safetensors");
    let header_eagle = r#"{"fc.weight":{"dtype":"F16","shape":[4,8],"data_offsets":[0,64]},"model.layers.0.self_attn.q_proj.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[64,96]}}"#;
    let payload_eagle = vec![0u8; 96];
    write_safe(&p_eagle, header_eagle, &payload_eagle);
    fs::write(dir_eagle.join("config.json"), r#"{"hidden_size": 4, "eagle_layers": [2, 4]}"#).unwrap();

    let out_eagle = dir_eagle.join("eagle.gguf");
    let rep_eagle = convert_checkpoint(
        &dir_eagle,
        &out_eagle,
        ConvertOptions {
            output_dtype: OutputDType::Preserve,
            overwrite: true,
            spec_head: Some(SpecHeadKind::Eagle3),
        },
    )
    .unwrap();
    assert_eq!(rep_eagle.architecture, "spec_head_eagle3");
    assert_eq!(rep_eagle.tensors_written, 2);
    assert!(out_eagle.is_file());

    fs::remove_dir_all(dir).unwrap();
    fs::remove_dir_all(dir_eagle).unwrap();
}
