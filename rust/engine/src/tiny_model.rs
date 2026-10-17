//! A tiny random Llama model for tests of the serving layer: `fllm-rs make-tiny-model OUT.gguf`.
//!
//! The model has two layers, a hidden size of 64 and a vocabulary of one token per printable ASCII character (plus
//! newline and three special tokens), so any English prompt tokenizes and any generated token decodes to text. Its
//! weights are random, so its output is gibberish; it exists to exercise HTTP routes, streaming, sampling and
//! structured output on the CPU in seconds without downloading anything. The same seed always gives the same file.
use fllm_checkpoint::{convert_checkpoint, ConvertOptions, OutputDType};
use serde_json::{json, Value};
use std::{fs, path::Path};

const HIDDEN: usize = 64;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 16;
const INTER: usize = 128;
const LAYERS: usize = 2;
/// Trained context of the tiny model.
pub const CONTEXT: usize = 512;

/// Chat format of the tiny model: `role: content` lines and a final `assistant:`.
pub const CHAT_TEMPLATE: &str = "{%- for message in messages -%}{{ message['role'] }}: {{ message['content'] }}\n{% endfor -%}{%- if add_generation_prompt -%}assistant:{%- endif -%}";

/// Sampling defaults written to `generation_config.json`, to test that they reach the GGUF and the server.
pub fn default_generation_config() -> Value {
    json!({"temperature": 0.7, "top_p": 0.95, "top_k": 20})
}

pub struct Options {
    pub seed: u64,
    pub generation_config: Option<Value>,
    pub chat_template: String,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            seed: 0x0005_eed0_ff11,
            generation_config: Some(default_generation_config()),
            chat_template: CHAT_TEMPLATE.to_owned(),
        }
    }
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

fn tokens() -> Vec<String> {
    let mut pieces = vec!["<unk>".to_owned(), "<s>".to_owned(), "</s>".to_owned(), "\n".to_owned()];
    pieces.extend((0x20u8..0x7f).map(|b| (b as char).to_string()));
    pieces
}

/// Number of tokens in the vocabulary of the tiny model.
pub fn vocab_size() -> usize {
    tokens().len()
}

fn tensor(rng: &mut Rng, name: &str, shape: &[usize], scale: f32, offset: f32) -> (String, Vec<usize>, Vec<u8>) {
    let count: usize = shape.iter().product();
    let bytes = (0..count).flat_map(|_| (offset + scale * rng.unit()).to_le_bytes()).collect();
    (name.to_owned(), shape.to_vec(), bytes)
}

/// Writes a Hugging Face style checkpoint directory (config, tokenizer, `model.safetensors`).
pub fn write_checkpoint(dir: &Path, options: &Options) -> Result<(), String> {
    let io = |e: std::io::Error| e.to_string();
    fs::create_dir_all(dir).map_err(io)?;
    let pieces = tokens();
    let vocab = pieces.len();
    let config = json!({
        "model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": HIDDEN,
        "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS, "num_hidden_layers": LAYERS,
        "intermediate_size": INTER, "vocab_size": vocab, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
        "max_position_embeddings": CONTEXT, "hidden_act": "silu", "tie_word_embeddings": false,
        "bos_token_id": 1, "eos_token_id": 2, "torch_dtype": "float32"
    });
    fs::write(dir.join("config.json"), config.to_string()).map_err(io)?;
    let vocab_map: serde_json::Map<String, Value> =
        pieces.iter().enumerate().map(|(i, p)| (p.clone(), json!(i))).collect();
    let tokenizer = json!({
        "version": "1.0",
        "added_tokens": pieces.iter().take(3).enumerate().map(|(id, content)| json!({
            "id": id, "content": content, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true
        })).collect::<Vec<_>>(),
        "model": {"type": "BPE", "unk_token": "<unk>", "vocab": vocab_map, "merges": []},
        "decoder": {"type": "Fuse"}
    });
    fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).map_err(io)?;
    fs::write(
        dir.join("tokenizer_config.json"),
        json!({"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "add_bos_token": false,
               "chat_template": options.chat_template})
        .to_string(),
    )
    .map_err(io)?;
    if let Some(generation) = &options.generation_config {
        fs::write(dir.join("generation_config.json"), generation.to_string()).map_err(io)?;
    }

    let mut rng = Rng(options.seed | 1);
    let mut entries = vec![
        tensor(&mut rng, "model.embed_tokens.weight", &[vocab, HIDDEN], 1.0, 0.0),
        tensor(&mut rng, "lm_head.weight", &[vocab, HIDDEN], 0.3, 0.0),
        tensor(&mut rng, "model.norm.weight", &[HIDDEN], 0.1, 1.0),
    ];
    for layer in 0..LAYERS {
        let p = format!("model.layers.{layer}");
        entries.push(tensor(&mut rng, &format!("{p}.input_layernorm.weight"), &[HIDDEN], 0.1, 1.0));
        entries.push(tensor(&mut rng, &format!("{p}.post_attention_layernorm.weight"), &[HIDDEN], 0.1, 1.0));
        for (name, out, inp) in [
            ("self_attn.q_proj", HEADS * HEAD_DIM, HIDDEN),
            ("self_attn.k_proj", KV_HEADS * HEAD_DIM, HIDDEN),
            ("self_attn.v_proj", KV_HEADS * HEAD_DIM, HIDDEN),
            ("self_attn.o_proj", HIDDEN, HEADS * HEAD_DIM),
            ("mlp.gate_proj", INTER, HIDDEN),
            ("mlp.up_proj", INTER, HIDDEN),
            ("mlp.down_proj", HIDDEN, INTER),
        ] {
            entries.push(tensor(&mut rng, &format!("{p}.{name}.weight"), &[out, inp], 0.15, 0.0));
        }
    }
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for (name, shape, bytes) in &entries {
        header.insert(
            name.clone(),
            json!({"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + bytes.len()]}),
        );
        offset += bytes.len();
    }
    let header = Value::Object(header).to_string();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    for (_, _, bytes) in &entries {
        file.extend_from_slice(bytes);
    }
    fs::write(dir.join("model.safetensors"), file).map_err(io)
}

/// Builds the tiny model and converts it to a GGUF at `out`.
pub fn write_gguf(out: &Path, options: &Options) -> Result<(), String> {
    let scratch = std::env::temp_dir().join(format!(
        "fllm-tiny-model-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    ));
    let built = write_checkpoint(&scratch, options).and_then(|()| {
        convert_checkpoint(
            &scratch,
            out,
            ConvertOptions { output_dtype: OutputDType::Preserve, overwrite: true, spec_head: None },
        )
        .map(|_| ())
        .map_err(|e| e.to_string())
    });
    let _ = fs::remove_dir_all(&scratch);
    built
}

/// `fllm-rs make-tiny-model OUT.gguf [--seed N] [--no-generation-config]`
pub fn run(args: &[String]) -> Result<(), String> {
    let mut options = Options::default();
    let mut out = None;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--seed" => {
                options.seed = iter
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--seed needs an integer")?
            }
            "--no-generation-config" => options.generation_config = None,
            other if !other.starts_with('-') && out.is_none() => out = Some(other.to_owned()),
            other => return Err(format!("usage: fllm-rs make-tiny-model OUT.gguf [--seed N] [--no-generation-config] (got {other})")),
        }
    }
    let out = out.ok_or("usage: fllm-rs make-tiny-model OUT.gguf [--seed N] [--no-generation-config]")?;
    write_gguf(Path::new(&out), &options)?;
    println!("wrote {out}");
    Ok(())
}
