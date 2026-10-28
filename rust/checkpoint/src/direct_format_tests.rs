//! Direct load of the checkpoint formats that expand or repack their weights while loading (MXFP4, bitsandbytes, MLX,
//! EXL2, EXL3, PyTorch ZIP weights): tensors, dimensions, types and metadata must equal the converted GGUF file byte for
//! byte. The fixtures are tiny Llama models whose linear layers carry random codes in each format's layout: only the
//! converter's handling is compared, so the codes need no meaning.
use crate::direct_tests::{assert_same, convert_to, scratch, tensor_names, write_file, Rng};
use crate::{load_checkpoint_as_gguf, load_checkpoint_quantized, ConvertOptions, OutputDType, Preset};
use fllm_gguf_probe::MappedGguf;
use half::{bf16, f16};
use serde_json::{json, Value};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
};

type Entry = (String, &'static str, Vec<usize>, Vec<u8>);

fn word(rng: &mut Rng) -> u32 {
    rng.unit();
    (rng.0 >> 32) as u32
}

fn random_bytes(rng: &mut Rng, n: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(n + 4);
    while v.len() < n {
        v.extend_from_slice(&word(rng).to_le_bytes());
    }
    v.truncate(n);
    v
}

fn f16_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(|v| f16::from_f32(v).to_le_bytes()).collect()
}

fn f32_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
}

/// Which tensors of the model a format quantizes.
#[derive(Clone, Copy)]
enum Scope {
    /// The attention and MLP linears.
    Linears,
    /// The linears, the token embedding and the output head.
    Everything,
}

/// Write the config, the tokenizer and a model in which `quantize(rng, prefix, out, in)` may replace a matrix by the
/// entries of a quantized layout (`None` keeps it dense, as bf16). `config` sets or overrides top-level config keys.
fn model(dir: &Path, seed: u64, config: &[(&str, Value)], scope: Scope, shards: usize, mut quantize: impl FnMut(&mut Rng, &str, usize, usize) -> Option<Vec<Entry>>) {
    let mut cfg = json!({
        "model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": crate::direct_tests::HIDDEN,
        "num_attention_heads": crate::direct_tests::HEADS, "num_key_value_heads": crate::direct_tests::KV_HEADS,
        "num_hidden_layers": crate::direct_tests::LAYERS, "intermediate_size": crate::direct_tests::INTER,
        "vocab_size": crate::direct_tests::VOCAB, "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "max_position_embeddings": 128,
        "hidden_act": "silu", "tie_word_embeddings": false, "bos_token_id": 1, "eos_token_id": 2
    });
    for (k, v) in config {
        cfg[*k] = v.clone();
    }
    fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
    let vocab: serde_json::Map<String, Value> = (0..crate::direct_tests::VOCAB).map(|i| (format!("t{i}"), json!(i))).collect();
    fs::write(dir.join("tokenizer.json"), json!({"version": "1.0", "added_tokens": [], "model": {"type": "BPE", "vocab": vocab, "merges": []}}).to_string()).unwrap();
    fs::write(dir.join("tokenizer_config.json"), json!({"bos_token": "t1", "eos_token": "t2"}).to_string()).unwrap();
    let mut rng = Rng(seed);
    let mut entries: Vec<Entry> = Vec::new();
    for (name, shape) in tensor_names() {
        let matrix = shape.len() == 2;
        let wanted = match scope {
            Scope::Linears => name.contains("_proj.weight"),
            Scope::Everything => matrix,
        };
        if wanted {
            let prefix = name.trim_end_matches(".weight");
            if let Some(quantized) = quantize(&mut rng, prefix, shape[0], shape[1]) {
                entries.extend(quantized);
                continue;
            }
        }
        let n: usize = shape.iter().product();
        let norm = name.contains("norm");
        let bytes = (0..n).flat_map(|_| bf16::from_f32(if norm { 1.0 + 0.1 * rng.unit() } else { 0.5 * rng.unit() }).to_le_bytes()).collect();
        entries.push((name, "BF16", shape, bytes));
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

/// The direct load equals the converted file for `outtype`, and `expect_type` is the GGUF type of `blk.0.ffn_up.weight`
/// (so that a test cannot pass by comparing two copies of an unconverted model).
fn assert_direct_equals_converted(dir: &Path, outtype: OutputDType, expect_type: u32, what: &str) {
    let options = ConvertOptions { output_dtype: outtype, overwrite: false, spec_head: None };
    let direct = load_checkpoint_as_gguf(dir, options).unwrap_or_else(|e| panic!("{what}: direct load failed: {e}"));
    let file = MappedGguf::open(convert_to(dir, &format!("ref-{outtype:?}.gguf"), outtype)).unwrap();
    assert_same(&direct, &file, what);
    let up = direct.index().tensors.iter().find(|t| t.name == "blk.0.ffn_up.weight").unwrap_or_else(|| panic!("{what}: no ffn_up"));
    assert_eq!(up.type_id, expect_type, "{what}: type of blk.0.ffn_up.weight");
}

const F16_TYPE: u32 = 1;
const F32_TYPE: u32 = 0;
const MXFP4_TYPE: u32 = 39;

// ---------------------------------------------------------------- MXFP4

fn mxfp4_ct_config() -> Value {
    json!({"quant_method": "compressed-tensors", "format": "mxfp4-pack-quantized", "ignore": ["lm_head"],
        "config_groups": {"group_0": {"targets": ["Linear"], "weights": {"num_bits": 4, "type": "float", "group_size": 32, "strategy": "group", "symmetric": true}}}})
}

fn mxfp4_ct(rng: &mut Rng, prefix: &str, out: usize, inp: usize) -> Option<Vec<Entry>> {
    let packed = random_bytes(rng, out * inp / 2);
    // E8M0 exponents around 2^-8, never the NaN code 0xff
    let scales: Vec<u8> = (0..out * inp / 32).map(|_| 110 + (word(rng) % 20) as u8).collect();
    Some(vec![
        (format!("{prefix}.weight_packed"), "U8", vec![out, inp / 2], packed),
        (format!("{prefix}.weight_scale"), "U8", vec![out, inp / 32], scales),
    ])
}

#[test]
fn mxfp4_compressed_tensors_loads_identically_to_the_converted_file() {
    // `preserve` keeps the 4-bit blocks, an explicit 16-bit output type expands them
    for (shards, outtype, expect) in [(1, OutputDType::Preserve, MXFP4_TYPE), (2, OutputDType::Preserve, MXFP4_TYPE), (1, OutputDType::F16, F16_TYPE)] {
        let dir = scratch("mxct");
        model(&dir, 21, &[("quantization_config", mxfp4_ct_config())], Scope::Linears, shards, mxfp4_ct);
        assert_direct_equals_converted(&dir, outtype, expect, &format!("mxfp4 ct, {shards} shards, {outtype:?}"));
        let _ = fs::remove_dir_all(dir);
    }
}

/// The gpt-oss checkpoint of the converter's own tests (`quant_method: mxfp4`, experts as `_blocks`/`_scales`).
#[test]
fn gpt_oss_mxfp4_checkpoint_loads_identically_to_the_converted_file() {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata").join("gptoss_mxfp4");
    let dir = scratch("gptoss");
    for name in ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"] {
        fs::copy(source.join(name), dir.join(name)).unwrap();
    }
    for outtype in [OutputDType::Preserve, OutputDType::F16] {
        let options = ConvertOptions { output_dtype: outtype, overwrite: false, spec_head: None };
        let direct = load_checkpoint_as_gguf(&dir, options).unwrap();
        let file = MappedGguf::open(convert_to(&dir, &format!("ref-{outtype:?}.gguf"), outtype)).unwrap();
        assert_same(&direct, &file, &format!("gpt-oss {outtype:?}"));
        let experts = direct.index().tensors.iter().find(|t| t.name == "blk.0.ffn_up_exps.weight").unwrap();
        assert_eq!(experts.type_id, MXFP4_TYPE, "the experts stay MXFP4");
    }
    let _ = fs::remove_dir_all(dir);
}

// ---------------------------------------------------------------- bitsandbytes

#[derive(Clone, Copy)]
enum Bnb {
    Nf4 { block: usize, double: bool },
    Fp4 { block: usize, double: bool },
    Int8,
}

fn bnb_config(kind: Bnb) -> Value {
    match kind {
        Bnb::Int8 => json!({"quant_method": "bitsandbytes", "load_in_8bit": true, "load_in_4bit": false}),
        Bnb::Nf4 { double, .. } => json!({"quant_method": "bitsandbytes", "load_in_4bit": true, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": double}),
        Bnb::Fp4 { double, .. } => json!({"quant_method": "bitsandbytes", "load_in_4bit": true, "bnb_4bit_quant_type": "fp4", "bnb_4bit_use_double_quant": double}),
    }
}

fn bnb(kind: Bnb) -> impl FnMut(&mut Rng, &str, usize, usize) -> Option<Vec<Entry>> {
    move |rng, prefix, out, inp| {
        let w = format!("{prefix}.weight");
        if let Bnb::Int8 = kind {
            return Some(vec![
                (w, "I8", vec![out, inp], random_bytes(rng, out * inp)),
                (format!("{prefix}.SCB"), "F32", vec![out], f32_bytes((0..out).map(|_| 0.5 + 0.5 * rng.unit().abs()))),
            ]);
        }
        let (name, block, double) = match kind {
            Bnb::Nf4 { block, double } => ("nf4", block, double),
            Bnb::Fp4 { block, double } => ("fp4", block, double),
            Bnb::Int8 => unreachable!(),
        };
        let n = out * inp;
        let blocks = n / block;
        let code = f32_bytes((0..16).map(|i| (i as f32 - 7.5) / 8.0));
        let mut entries = vec![
            (w.clone(), "U8", vec![n / 2, 1], random_bytes(rng, n / 2)),
            (format!("{w}.quant_map"), "F32", vec![16], code),
        ];
        let mut state = json!({"quant_type": name, "blocksize": block, "dtype": "bfloat16", "shape": [out, inp]});
        if double {
            let nested_block = 256usize;
            entries.push((format!("{w}.absmax"), "U8", vec![blocks], random_bytes(rng, blocks)));
            entries.push((format!("{w}.nested_absmax"), "F32", vec![blocks.div_ceil(nested_block)], f32_bytes((0..blocks.div_ceil(nested_block)).map(|_| 0.5 + 0.25 * rng.unit().abs()))));
            entries.push((format!("{w}.nested_quant_map"), "F32", vec![256], f32_bytes((0..256).map(|i| (i as f32 - 127.5) / 128.0))));
            state["nested_blocksize"] = json!(nested_block);
            state["nested_offset"] = json!(0.0625);
        } else {
            entries.push((format!("{w}.absmax"), "F32", vec![blocks], f32_bytes((0..blocks).map(|_| 0.05 + 0.05 * rng.unit().abs()))));
        }
        let blob = state.to_string().into_bytes();
        entries.push((format!("{w}.quant_state.bitsandbytes__{name}"), "U8", vec![blob.len()], blob));
        Some(entries)
    }
}

#[test]
fn bitsandbytes_checkpoints_load_identically_to_the_converted_file() {
    for (kind, label, shards) in [
        (Bnb::Nf4 { block: 64, double: false }, "nf4 bs64", 1),
        (Bnb::Nf4 { block: 128, double: true }, "nf4 double-quantized bs128", 2),
        (Bnb::Fp4 { block: 64, double: false }, "fp4 bs64", 1),
        (Bnb::Fp4 { block: 64, double: true }, "fp4 double-quantized bs64", 1),
        (Bnb::Int8, "int8", 1),
        (Bnb::Int8, "int8 sharded", 3),
    ] {
        let dir = scratch("bnb");
        model(&dir, 31, &[("quantization_config", bnb_config(kind))], Scope::Linears, shards, bnb(kind));
        assert_direct_equals_converted(&dir, OutputDType::Preserve, F16_TYPE, label);
        let _ = fs::remove_dir_all(dir);
    }
    // F32 output and quantizing on load take the same path
    let dir = scratch("bnbq");
    let kind = Bnb::Nf4 { block: 64, double: false };
    model(&dir, 32, &[("quantization_config", bnb_config(kind))], Scope::Linears, 1, bnb(kind));
    assert_direct_equals_converted(&dir, OutputDType::F32, F32_TYPE, "nf4 as f32");
    let options = ConvertOptions { output_dtype: OutputDType::F16, overwrite: false, spec_head: None };
    let preset = Preset::parse("q8_0").unwrap();
    let direct = load_checkpoint_quantized(&dir, options, Some(preset), None).unwrap();
    let stage = convert_to(&dir, "stage.gguf", OutputDType::F16);
    let quantized = dir.join("quantized.gguf");
    crate::quantize_gguf(&stage, &quantized, preset).unwrap();
    assert_same(&direct, &MappedGguf::open(&quantized).unwrap(), "nf4 with q8_0 on load");
    let _ = fs::remove_dir_all(dir);
}

// ---------------------------------------------------------------- MLX

fn mlx(bits: usize, group: usize) -> impl FnMut(&mut Rng, &str, usize, usize) -> Option<Vec<Entry>> {
    move |rng, prefix, out, inp| {
        let groups = out * inp / group;
        Some(vec![
            (format!("{prefix}.weight"), "U32", vec![out, inp * bits / 32], random_bytes(rng, out * inp * bits / 8)),
            (format!("{prefix}.scales"), "F16", vec![out, inp / group], f16_bytes((0..groups).map(|_| 0.01 + 0.01 * rng.unit().abs()))),
            (format!("{prefix}.biases"), "F16", vec![out, inp / group], f16_bytes((0..groups).map(|_| 0.05 * rng.unit()))),
        ])
    }
}

#[test]
fn mlx_quantized_checkpoints_load_identically_to_the_converted_file() {
    for (bits, group, scope, shards) in [
        (4, 64, Scope::Everything, 1),
        (4, 32, Scope::Linears, 2),
        (2, 32, Scope::Linears, 1),
        (3, 64, Scope::Everything, 1),
        (5, 32, Scope::Linears, 1),
        (6, 64, Scope::Linears, 1),
        (8, 32, Scope::Everything, 1),
    ] {
        let dir = scratch("mlx");
        model(&dir, 41, &[("quantization", json!({"group_size": group, "bits": bits}))], scope, shards, mlx(bits, group));
        assert_direct_equals_converted(&dir, OutputDType::Preserve, F16_TYPE, &format!("mlx {bits} bits, group {group}, {shards} shards"));
        let _ = fs::remove_dir_all(dir);
    }
    // MLX also writes the same dictionary as `quantization_config`
    let dir = scratch("mlx2");
    model(&dir, 42, &[("quantization_config", json!({"group_size": 64, "bits": 4}))], Scope::Linears, 1, mlx(4, 64));
    assert_direct_equals_converted(&dir, OutputDType::F32, F32_TYPE, "mlx 4 bits as f32");
    let _ = fs::remove_dir_all(dir);
}

// ---------------------------------------------------------------- EXL2 and EXL3

const EXL2_WIDTHS: [usize; 6] = [2, 3, 4, 5, 6, 8];

fn exl2(rng: &mut Rng, prefix: &str, out: usize, inp: usize) -> Option<Vec<Entry>> {
    let count = inp / 32;
    let mut starts = Vec::new();
    let mut rows = 0usize;
    for g in 0..count {
        starts.push((EXL2_WIDTHS[g % 6], rows));
        rows += EXL2_WIDTHS[g % 6];
    }
    // the stored input order is a permutation of the original one
    let mut perm: Vec<u32> = (0..inp as u32).collect();
    for i in (1..inp).rev() {
        perm.swap(i, word(rng) as usize % (i + 1));
    }
    Some(vec![
        (format!("{prefix}.q_weight"), "I32", vec![rows, out], random_bytes(rng, rows * out * 4)),
        (format!("{prefix}.q_groups"), "I16", vec![count * 2], starts.iter().flat_map(|&(b, s)| [b as u16, s as u16]).flat_map(u16::to_le_bytes).collect()),
        (format!("{prefix}.q_scale"), "I32", vec![count, out / 8], random_bytes(rng, count * out / 8 * 4)),
        (format!("{prefix}.q_scale_max"), "F16", vec![count], f16_bytes((0..count).map(|_| 64.0 + 64.0 * rng.unit().abs()))),
        (format!("{prefix}.q_invperm"), "I32", vec![inp], perm.iter().flat_map(|v| v.to_le_bytes()).collect()),
    ])
}

#[test]
fn exl2_checkpoints_load_identically_to_the_converted_file() {
    let cfg = json!({"quant_method": "exl2", "bits": 4.0});
    for (shards, outtype) in [(1, OutputDType::Preserve), (2, OutputDType::F16)] {
        let dir = scratch("exl2");
        model(&dir, 51, &[("quantization_config", cfg.clone())], Scope::Linears, shards, exl2);
        assert_direct_equals_converted(&dir, outtype, F16_TYPE, &format!("exl2, {shards} shards, {outtype:?}"));
        let _ = fs::remove_dir_all(dir);
    }
}

/// `codebook`: 0 default, 1 `mcg`, 2 `mul1`; `packed_signs` stores the sign vectors as bit sets (`su`/`sv`) instead of halves.
fn exl3(bits: usize, codebook: u8, packed_signs: bool) -> impl FnMut(&mut Rng, &str, usize, usize) -> Option<Vec<Entry>> {
    move |rng, prefix, out, inp| {
        let sign_half = |rng: &mut Rng, n: usize| f16_bytes((0..n).map(|_| if word(rng) & 1 == 0 { 1.0 } else { -1.0 }));
        let mut entries = vec![(format!("{prefix}.trellis"), "I16", vec![inp / 16, out / 16, 16 * bits], random_bytes(rng, inp / 16 * out / 16 * 16 * bits * 2))];
        if packed_signs {
            entries.push((format!("{prefix}.su"), "I16", vec![inp / 16], random_bytes(rng, inp / 8)));
            entries.push((format!("{prefix}.sv"), "I16", vec![out / 16], random_bytes(rng, out / 8)));
        } else {
            entries.push((format!("{prefix}.suh"), "F16", vec![inp], sign_half(rng, inp)));
            entries.push((format!("{prefix}.svh"), "F16", vec![out], sign_half(rng, out)));
        }
        match codebook {
            1 => entries.push((format!("{prefix}.mcg"), "I32", vec![], 0xcbac_1fedu32.to_le_bytes().to_vec())),
            2 => entries.push((format!("{prefix}.mul1"), "I32", vec![], 1u32.to_le_bytes().to_vec())),
            _ => {}
        }
        Some(entries)
    }
}

#[test]
fn exl3_checkpoints_load_identically_to_the_converted_file() {
    let cfg = json!({"quant_method": "exl3", "bits": 3.0});
    for (bits, codebook, packed_signs, shards) in [(2, 0, false, 1), (4, 1, true, 2), (3, 2, false, 1)] {
        let dir = scratch("exl3");
        model(&dir, 61, &[("quantization_config", cfg.clone())], Scope::Linears, shards, exl3(bits, codebook, packed_signs));
        assert_direct_equals_converted(&dir, OutputDType::Preserve, F16_TYPE, &format!("exl3 {bits} bits, codebook {codebook}, packed signs {packed_signs}, {shards} shards"));
        let _ = fs::remove_dir_all(dir);
    }
}

// ---------------------------------------------------------------- PyTorch ZIP weights

/// One tensor as `torch.save` writes it: a storage (key `index`) and `_rebuild_tensor_v2` over it.
fn pickle_tensor(p: &mut Vec<u8>, index: usize, storage: &str, shape: &[usize]) {
    let numel: usize = shape.iter().product();
    let int = |p: &mut Vec<u8>, v: usize| match v {
        0..=255 => p.extend_from_slice(&[b'K', v as u8]),
        256..=65535 => {
            p.push(b'M');
            p.extend_from_slice(&(v as u16).to_le_bytes());
        }
        _ => {
            p.push(b'J');
            p.extend_from_slice(&(v as u32).to_le_bytes());
        }
    };
    let text = |p: &mut Vec<u8>, s: &str| {
        p.push(b'X');
        p.extend_from_slice(&(s.len() as u32).to_le_bytes());
        p.extend_from_slice(s.as_bytes());
    };
    p.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n(("); // global, MARK of the arguments, MARK of the persistent id
    text(p, "storage");
    p.extend_from_slice(format!("ctorch\n{storage}\n").as_bytes());
    text(p, &index.to_string());
    text(p, "cpu");
    int(p, numel);
    p.extend_from_slice(b"tQ"); // the persistent id tuple becomes the storage
    int(p, 0);
    p.push(b'(');
    for d in shape {
        int(p, *d);
    }
    p.push(b't');
    p.push(b'(');
    let mut stride = 1usize;
    let mut strides = Vec::new();
    for d in shape.iter().rev() {
        strides.push(stride);
        stride *= d.max(&1);
    }
    for s in strides.iter().rev() {
        int(p, *s);
    }
    p.extend_from_slice(&[b't', 0x89, b'h', 0, b')', b'R', b't', b'R']); // requires_grad False, OrderedDict(), TUPLE, REDUCE
}

/// A `torch.save`-style state dict archive: `data.pkl`, one `data/<n>` member per tensor and `byteorder`.
fn write_torch_zip(path: &Path, entries: &[Entry], stored: bool, aligned: bool) {
    use zip::{write::SimpleFileOptions, CompressionMethod, ZipWriter};
    let mut pickle = vec![0x80, 2];
    pickle.extend_from_slice(b"ccollections\nOrderedDict\nq\x00)Rq\x01(");
    for (i, (name, dtype, shape, _)) in entries.iter().enumerate() {
        pickle.push(b'X');
        pickle.extend_from_slice(&(name.len() as u32).to_le_bytes());
        pickle.extend_from_slice(name.as_bytes());
        let storage = match *dtype {
            "BF16" => "BFloat16Storage",
            "F16" => "HalfStorage",
            "F32" => "FloatStorage",
            other => panic!("no torch storage for {other}"),
        };
        pickle_tensor(&mut pickle, i, storage, shape);
    }
    pickle.extend_from_slice(b"u.");
    let mut zip = ZipWriter::new(fs::File::create(path).unwrap());
    let mut opts = SimpleFileOptions::default().compression_method(if stored { CompressionMethod::Stored } else { CompressionMethod::Deflated });
    zip.start_file("archive/data.pkl", opts).unwrap();
    zip.write_all(&pickle).unwrap();
    if aligned {
        // torch aligns every storage to 64 bytes
        opts = opts.with_alignment(64);
    }
    for (i, (.., bytes)) in entries.iter().enumerate() {
        zip.start_file(format!("archive/data/{i}"), opts).unwrap();
        zip.write_all(bytes).unwrap();
    }
    zip.start_file("archive/byteorder", SimpleFileOptions::default().compression_method(CompressionMethod::Stored)).unwrap();
    zip.write_all(b"little").unwrap();
    zip.finish().unwrap();
}

/// A tiny Llama directory with its weights as `.bin` archive(s); returns the entries of every file.
fn torch_model(dir: &Path, seed: u64, files: &[&str], stored: bool, store: &'static str) -> usize {
    // the support files and the weights come from the same generator as the safetensors fixtures
    let staging = scratch("torchsrc");
    model(&staging, seed, &[], Scope::Linears, 1, |_, _, _, _| None);
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
        fs::copy(staging.join(name), dir.join(name)).unwrap();
    }
    let tensors = crate::safetensors::read_file(&staging.join("model.safetensors")).unwrap();
    let mut entries: Vec<Entry> = Vec::new();
    for t in tensors {
        let values: Vec<f32> = t.payload.as_slice().as_chunks::<2>().0.iter().map(|c| bf16::from_bits(u16::from_le_bytes(*c)).to_f32()).collect();
        let (dtype, bytes): (&'static str, Vec<u8>) = match store {
            "BF16" => ("BF16", t.payload.as_slice().to_vec()),
            "F16" => ("F16", f16_bytes(values)),
            _ => ("F32", f32_bytes(values)),
        };
        entries.push((t.name, dtype, t.shape.iter().map(|&d| d as usize).collect(), bytes));
    }
    let _ = fs::remove_dir_all(staging);
    let total = entries.len();
    let per = entries.len().div_ceil(files.len());
    let mut weight_map = serde_json::Map::new();
    for (chunk, file) in entries.chunks(per).zip(files) {
        write_torch_zip(&dir.join(file), chunk, stored, true);
        for (name, ..) in chunk {
            weight_map.insert(name.clone(), json!(file));
        }
    }
    if files.len() > 1 {
        fs::write(dir.join("pytorch_model.bin.index.json"), json!({"metadata": {}, "weight_map": weight_map}).to_string()).unwrap();
    }
    total
}

#[test]
fn pytorch_zip_weights_load_identically_to_the_converted_file() {
    for (files, stored, store, outtype, label) in [
        (vec!["pytorch_model.bin"], true, "BF16", OutputDType::Preserve, "one stored bf16 file"),
        (vec!["pytorch_model.bin"], false, "BF16", OutputDType::Preserve, "one deflated bf16 file"),
        (vec!["pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin"], true, "F16", OutputDType::Preserve, "two stored f16 shards"),
        (vec!["pytorch_model.bin"], true, "F32", OutputDType::F16, "stored f32 as f16"),
    ] {
        let dir = scratch("torch");
        torch_model(&dir, 71, &files, stored, store);
        let expect = if matches!(outtype, OutputDType::F16) || store == "F16" { F16_TYPE } else { 30 };
        assert_direct_equals_converted(&dir, outtype, expect, label);
        let _ = fs::remove_dir_all(dir);
    }
}

#[test]
fn stored_pytorch_storages_stay_slices_of_the_archive() {
    let dir = scratch("torchzc");
    torch_model(&dir, 72, &["pytorch_model.bin"], true, "BF16");
    let direct = load_checkpoint_as_gguf(&dir, ConvertOptions::default()).unwrap();
    let embd = direct.extra_mappings().get("token_embd.weight").unwrap();
    let bytes = direct.tensor_bytes("token_embd.weight").unwrap();
    let (m0, m1) = (embd.as_ptr() as usize, embd.as_ptr() as usize + embd.len());
    assert!(bytes.as_ptr() as usize >= m0 && bytes.as_ptr() as usize + bytes.len() <= m1);
    assert!(embd.len() as u64 > dir.join("pytorch_model.bin").metadata().unwrap().len() / 2, "the mapping is the archive itself");
    assert!(!Arc::ptr_eq(embd, direct.extra_mappings().get("blk.0.attn_q.weight").unwrap()), "rope-permuted weights are built");
    // deflated storages cannot be slices: everything is built into one anonymous map
    let packed = scratch("torchdf");
    torch_model(&packed, 72, &["pytorch_model.bin"], false, "BF16");
    let built = load_checkpoint_as_gguf(&packed, ConvertOptions::default()).unwrap();
    assert!(Arc::ptr_eq(built.extra_mappings().get("token_embd.weight").unwrap(), built.extra_mappings().get("blk.0.attn_q.weight").unwrap()));
    let _ = fs::remove_dir_all(dir);
    let _ = fs::remove_dir_all(packed);
}

#[test]
fn the_memory_budget_applies_to_every_expanded_format() {
    let over_budget = |dir: &Path, what: &str| {
        match load_checkpoint_quantized(dir, ConvertOptions::default(), None, Some(1)) {
            Err(crate::CheckpointError::OverBudget { needed, limit }) => assert!(needed > 1 && limit == 1, "{what}"),
            other => panic!("{what}: expected OverBudget, got {:?}", other.map(|_| ())),
        }
        assert!(load_checkpoint_quantized(dir, ConvertOptions::default(), None, Some(u64::MAX)).is_ok(), "{what}");
    };
    let mlx_dir = scratch("budget-mlx");
    model(&mlx_dir, 81, &[("quantization", json!({"group_size": 64, "bits": 4}))], Scope::Linears, 1, mlx(4, 64));
    over_budget(&mlx_dir, "mlx");
    let bnb_dir = scratch("budget-bnb");
    let kind = Bnb::Nf4 { block: 64, double: false };
    model(&bnb_dir, 82, &[("quantization_config", bnb_config(kind))], Scope::Linears, 1, bnb(kind));
    over_budget(&bnb_dir, "bitsandbytes");
    let exl_dir = scratch("budget-exl2");
    model(&exl_dir, 83, &[("quantization_config", json!({"quant_method": "exl2"}))], Scope::Linears, 1, exl2);
    over_budget(&exl_dir, "exl2");
    let bin_dir = scratch("budget-bin");
    torch_model(&bin_dir, 84, &["pytorch_model.bin"], false, "BF16");
    over_budget(&bin_dir, "deflated pytorch weights");
    for d in [mlx_dir, bnb_dir, exl_dir, bin_dir] {
        let _ = fs::remove_dir_all(d);
    }
}

/// A `data.pkl` that would run `os.system` (or similar) under the real unpickler; the restricted reader must refuse it
/// and must not run anything.
fn malicious_archive(dir: &Path, pickle: &[u8]) -> PathBuf {
    use zip::{write::SimpleFileOptions, CompressionMethod, ZipWriter};
    let path = dir.join("evil.bin");
    let mut zip = ZipWriter::new(fs::File::create(&path).unwrap());
    let opts = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
    zip.start_file("archive/data.pkl", opts).unwrap();
    zip.write_all(pickle).unwrap();
    zip.finish().unwrap();
    path
}

#[test]
fn malicious_pickles_are_refused_and_never_run() {
    let dir = scratch("evil");
    let marker = dir.join("pwned");
    let command = format!("touch {}", marker.display());
    let mut cases: Vec<(&str, Vec<u8>)> = Vec::new();
    // GLOBAL + REDUCE: os.system(command)
    for (module, function) in [("os", "system"), ("posix", "system"), ("subprocess", "check_output"), ("builtins", "eval"), ("builtins", "exec"), ("nt", "system")] {
        let mut p = vec![0x80, 2];
        p.extend_from_slice(format!("c{module}\n{function}\n").as_bytes());
        p.push(b'X');
        p.extend_from_slice(&(command.len() as u32).to_le_bytes());
        p.extend_from_slice(command.as_bytes());
        p.extend_from_slice(&[0x85, b'R', b'.']);
        cases.push(("global reduce", p));
    }
    // STACK_GLOBAL (protocol 4)
    let mut p = vec![0x80, 4];
    for s in ["os", "system"] {
        p.push(0x8c);
        p.push(s.len() as u8);
        p.extend_from_slice(s.as_bytes());
    }
    p.push(0x93);
    p.push(0x8c);
    p.push(command.len() as u8);
    p.extend_from_slice(command.as_bytes());
    p.extend_from_slice(&[0x85, b'R', b'.']);
    cases.push(("stack global", p));
    // INST, OBJ, NEWOBJ, NEWOBJ_EX, BUILD and EXT opcodes
    let mut inst = vec![0x80, 2, b'('];
    inst.extend_from_slice(b"ios\nsystem\n");
    inst.push(b'.');
    cases.push(("INST", inst));
    cases.push(("OBJ", vec![0x80, 2, b'(', b'N', b'o', b'.']));
    cases.push(("NEWOBJ", vec![0x80, 2, b'N', b')', 0x81, b'.']));
    cases.push(("NEWOBJ_EX", vec![0x80, 4, b'N', b')', b'}', 0x92, b'.']));
    cases.push(("EXT1", vec![0x80, 2, 0x82, 1, b'.']));
    cases.push(("BUILD with a callable", {
        let mut p = vec![0x80, 2, b'}'];
        p.extend_from_slice(b"cos\nsystem\n}b.");
        p
    }));
    // an allowlisted tensor rebuild is fine, but only for the torch functions: a global that merely ends in the same name is not
    let mut sneaky = vec![0x80, 2];
    sneaky.extend_from_slice(b"cevil._utils\n_rebuild_tensor_v2\n)R.");
    cases.push(("lookalike global", sneaky));
    for (label, pickle) in cases {
        let archive = malicious_archive(&dir, &pickle);
        let err = crate::torch_zip::read(&archive).expect_err(label);
        assert!(!marker.exists(), "{label}: the pickle ran code ({err})");
        let _ = fs::remove_file(archive);
    }
    // a legacy (non-ZIP) pickle file is refused before any parsing
    let legacy = dir.join("legacy.bin");
    let mut body = vec![0x80, 2];
    body.extend_from_slice(b"cos\nsystem\n");
    body.push(b'X');
    body.extend_from_slice(&(command.len() as u32).to_le_bytes());
    body.extend_from_slice(command.as_bytes());
    body.extend_from_slice(&[0x85, b'R', b'.']);
    fs::write(&legacy, body).unwrap();
    assert!(crate::torch_zip::read(&legacy).unwrap_err().contains("never executes"));
    assert!(!marker.exists());
    // the same refusal through the direct load of a checkpoint directory
    let model_dir = scratch("evilmodel");
    torch_model(&model_dir, 73, &["pytorch_model.bin"], true, "BF16");
    fs::copy(malicious_archive(&dir, &cases_for_direct(&command)), model_dir.join("pytorch_model.bin")).unwrap();
    assert!(load_checkpoint_as_gguf(&model_dir, ConvertOptions::default()).is_err());
    assert!(!marker.exists());
    let _ = fs::remove_dir_all(dir);
    let _ = fs::remove_dir_all(model_dir);
}

fn cases_for_direct(command: &str) -> Vec<u8> {
    let mut p = vec![0x80, 2];
    p.extend_from_slice(b"cos\nsystem\n");
    p.push(b'X');
    p.extend_from_slice(&(command.len() as u32).to_le_bytes());
    p.extend_from_slice(command.as_bytes());
    p.extend_from_slice(&[0x85, b'R', b'.']);
    p
}
