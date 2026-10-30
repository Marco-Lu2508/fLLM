//! GGUF model description: hyperparameters, paged KV geometry, RoPE tables and load-time LoRA
//! merging. Weights stay memory-mapped; the compute paths read them through `MappedGguf`.
use crate::compute;
use crate::config::ModelConfig;
use crate::rope::{self, RopeTables};
use fllm_gguf_probe::{MappedGguf, TensorInfo};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

const SUPPORTED_ARCHITECTURES: &[&str] = &[
    "dbrx",
    "lfm2",
    "lfm2moe",
    "qwen3",
    "qwen2",
    "qwen",
    "baichuan",
    "llama",
    "mistral",
    "gemma",
    "gemma2",
    "gemma3",
    "gemma4",
    "laguna",
    "mimo_v2",
    "gemma3n",
    "vaultgemma",
    "xverse_moe",
    "muse-glimmer",
    "olmo2",
    "granite",
    "phi3",
    "phimoe",
    "granitehybrid",
    "mamba",
    "cohere2",
    "cohere2_moe",
    "granitemoe",
    "stablelm",
    "olmo",
    "persimmon",
    "phi",
    "gpt2",
    "starcoder",
    "gpt_neo",
    "gptj",
    "opt",
    "falcon",
    "bloom",
    "mpt",
    "olmo_hybrid",
    "llama4",
    "minicpm",
    "minicpm3",
    "apertus",
    "hy_v3",
    "exaone_moe",
    "afmoe",
    "cwm",
    "hyperclovax",
    "flex_olmo",
    "nanochat",
    "jais2",
    "nemotron",
    "internlm2",
    "nemotron_h",
    "nemotron_h_moe",
    "bamba",
    "falcon-h1",
    "jamba",
    "zamba2",
    "kimi-linear",
    "mamba2",
    "qwen3next",
    "qwen35",
    "qwen35moe",
    "gpt-oss",
    "glm4moe",
    "dots1",
    "minimax-m2",
    "ernie4_5moe",
    "hunyuan-moe",
    "glm",
    "glm4",
    "chatglm",
    "seed_oss",
    "smollm3",
    "arcee",
    "ministral3",
    "ernie4_5",
    "helium",
    "bitnet",
    "hunyuan-dense",
    "exaone4",
    "olmo3",
    "command-r",
    "starcoder2",
    "gptneox",
    "qwen2moe",
    "qwen3moe",
    "olmoe",
    "deepseek",
    "deepseek2",
    "k2_horizon",
    "k2-horizon",
    "k2horizon",
    "k2",
    "g9v3",
];

/// llama.cpp architecture names for layouts this program implements under its own name. Only names whose metadata keys
/// and tensor names are the ones the converter writes for the target are listed:
/// * `exaone` (EXAONE 3.x) is the Llama graph with the Llama tensor names;
/// * `ernie4_5-moe` is `ernie4_5moe` with a hyphen;
/// * `phi2` has the tensors of `phi` (GGUFs with a fused `attn_qkv` tensor fail on the missing `attn_q`).
const ARCHITECTURE_ALIASES: &[(&str, &str)] = &[("exaone", "llama"), ("ernie4_5-moe", "ernie4_5moe"), ("phi2", "phi")];

fn canonical_architecture(mapping: MappedGguf) -> MappedGguf {
    let arch = mapping.index().metadata.get("general.architecture").and_then(Value::as_str).unwrap_or_default();
    match ARCHITECTURE_ALIASES.iter().find(|(from, _)| *from == arch) {
        Some((from, to)) => mapping.with_architecture_alias(from, to),
        None => mapping,
    }
}

fn tensor<'a>(mapping: &'a MappedGguf, name: &str) -> Result<&'a TensorInfo, String> {
    mapping
        .index()
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| format!("missing GGUF tensor {name}"))
}

fn has_tensor(mapping: &MappedGguf, name: &str) -> bool {
    mapping.index().tensors.iter().any(|t| t.name == name)
}

/// Whole tensor as f32 (norm vectors, biases, LoRA factors and merge bases).
fn tensor_f32(mapping: &MappedGguf, name: &str) -> Result<Vec<f32>, String> {
    let descriptor = tensor(mapping, name)?;
    let bytes = mapping.tensor_bytes(name)?;
    let count = descriptor
        .dimensions
        .iter()
        .try_fold(1usize, |acc, &d| {
            usize::try_from(d).ok().and_then(|dim| acc.checked_mul(dim))
        })
        .ok_or_else(|| format!("tensor {name} dimensions overflow"))?;
    compute::dequantize(bytes, descriptor.type_id, count)
        .map_err(|e| format!("cannot read tensor {name}: {e}"))
}

fn merge_lora_delta(
    weights: &mut [f32],
    a: &[f32],
    b: &[f32],
    rows: usize,
    cols: usize,
    rank: usize,
    scale: f32,
) -> Result<(), String> {
    if rank == 0
        || weights.len() != rows.checked_mul(cols).ok_or("LoRA shape overflow")?
        || a.len() != rank.checked_mul(cols).ok_or("LoRA shape overflow")?
        || b.len() != rows.checked_mul(rank).ok_or("LoRA shape overflow")?
    {
        return Err("LoRA matrix dimensions do not match".into());
    }
    if a.iter().chain(b).any(|v| !v.is_finite()) {
        return Err("LoRA weights must be finite".into());
    }
    for (row, destination) in weights.chunks_exact_mut(cols).enumerate() {
        for (r, source) in a.chunks_exact(cols).enumerate() {
            let factor = b[row * rank + r] * scale;
            for (value, delta) in destination.iter_mut().zip(source) {
                *value += factor * delta;
            }
        }
    }
    if weights.iter().any(|v| !v.is_finite()) {
        return Err("LoRA merge overflowed".into());
    }
    Ok(())
}

/// A memory-mapped GGUF decoder model plus the paged KV geometry shared with the scheduler.
/// KV buffers are owned by the caller so the scheduler can assign shared paged blocks.
pub struct GgufModel {
    mapping: MappedGguf,
    config: ModelConfig,
    max_context: usize,
    max_sequences: usize,
    block_size: usize,
    table_stride: usize,
    pool_blocks: usize,
    threads: i32,
    /// Workers for batches of many rows (0: the same as `threads`).
    prefill_threads: i32,
    rope: Arc<Vec<RopeTables>>,
    /// Base of each table of `rope`.
    rope_bases: Vec<f32>,
    chat_template: String,
    tool_chat_template: String,
    kv_kind: i32,
    /// The KV cache holds the MLA latents instead of the rebuilt per-head keys and values (`enable_mla_latent`).
    mla_latent: bool,
    adapted: HashMap<String, Vec<f32>>,
}

impl GgufModel {
    pub fn from_mapping(
        mapping: MappedGguf,
        max_context: usize,
        max_sequences: usize,
        threads: i32,
    ) -> Result<Self, String> {
        if max_context == 0 || max_sequences == 0 || threads <= 0 {
            return Err("context, sequence capacity, and thread count must be positive".into());
        }
        let mapping = canonical_architecture(mapping);
        let config = ModelConfig::from_gguf(&mapping)?;
        if !SUPPORTED_ARCHITECTURES.contains(&config.arch.as_str()) {
            return Err(format!(
                "unsupported GGUF architecture '{}' (supported: {})",
                config.arch,
                SUPPORTED_ARCHITECTURES.join(", ")
            ));
        }
        let chat_template = mapping
            .index()
            .metadata
            .get("tokenizer.chat_template")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let tool_chat_template = mapping
            .index()
            .metadata
            .get("tokenizer.chat_template.tool_use")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let block_size = 16usize;
        let table_stride = max_context.div_ceil(block_size);
        let pool_blocks = table_stride
            .checked_mul(max_sequences)
            .ok_or("KV block count overflow")?;
        let mut rope = vec![rope::build(&config.rope, config.rope.base, max_context)];
        if let Some(local) = config.rope.local_base {
            // local (sliding-window) layers never use the long-context scaling
            let mut plain = config.rope.clone();
            plain.scaling = crate::config::RopeScaling::None;
            plain.factors = None;
            if let Some(dims) = plain.local_dims {
                plain.dims = dims;
            }
            rope.push(rope::build(&plain, local, max_context));
        }
        let mut rope_bases = vec![config.rope.base];
        if let Some(local) = config.rope.local_base {
            rope_bases.push(local);
        }
        // layers with their own rotary base (Granite SWA) get a table each
        if let Some(bases) = &config.layer_rope_bases {
            for &base in bases.iter().filter(|b| **b > 0.0) {
                if !rope_bases.contains(&base) {
                    let mut plain = config.rope.clone();
                    plain.scaling = crate::config::RopeScaling::None;
                    plain.factors = None;
                    rope.push(rope::build(&plain, base, max_context));
                    rope_bases.push(base);
                }
            }
        }
        Ok(Self {
            mapping,
            config,
            max_context,
            max_sequences,
            block_size,
            table_stride,
            pool_blocks,
            threads,
            prefill_threads: 0,
            rope: Arc::new(rope),
            rope_bases,
            chat_template,
            tool_chat_template,
            kv_kind: 1,
            mla_latent: false,
            adapted: HashMap::new(),
        })
    }

    pub fn load(
        model_path: &str,
        max_context: usize,
        max_sequences: usize,
        threads: i32,
    ) -> Result<Self, String> {
        let mapping = crate::import::open_model(std::path::Path::new(model_path))?;
        Self::from_mapping(mapping, max_context, max_sequences, threads)
    }

    /// Another handle on the same model: the memory map and all tensors are shared, nothing is read again. Every rank of a parallel
    /// group loads its share through its own handle, in its own thread.
    pub fn duplicate(&self) -> Result<Self, String> {
        Ok(Self {
            mapping: self.mapping.with_tensor_aliases(&[])?,
            config: self.config.clone(),
            max_context: self.max_context,
            max_sequences: self.max_sequences,
            block_size: self.block_size,
            table_stride: self.table_stride,
            pool_blocks: self.pool_blocks,
            threads: self.threads,
            prefill_threads: self.prefill_threads,
            rope: Arc::clone(&self.rope),
            chat_template: self.chat_template.clone(),
            kv_kind: self.kv_kind,
            adapted: self.adapted.clone(),
            mla_latent: self.mla_latent,
            rope_bases: self.rope_bases.clone(),
            tool_chat_template: self.tool_chat_template.clone(),
        })
    }

    /// Names of every projection a LoRA adapter may modify.
    fn projection_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        for layer in 0..self.config.layers {
            for part in [
                "attn_q",
                "attn_k",
                "attn_v",
                "attn_output",
                "ffn_gate",
                "ffn_up",
                "ffn_down",
            ] {
                names.push(format!("blk.{layer}.{part}.weight"));
            }
        }
        names.push(if has_tensor(&self.mapping, "output.weight") {
            "output.weight".to_owned()
        } else {
            "token_embd.weight".to_owned()
        });
        names.retain(|name| has_tensor(&self.mapping, name));
        names
    }

    /// Load-time LoRA merge for linear projections. Adapted matrices are retained as F32; base
    /// GGUF files are never modified. This trades RAM for a simple runtime path and preserves
    /// adapters across batched execution.
    pub fn apply_lora(&mut self, path: &str, scale: f32) -> Result<usize, String> {
        if !scale.is_finite() {
            return Err("LoRA scale must be finite".into());
        }
        let adapter = MappedGguf::open(path).map_err(|e| e.to_string())?;
        let metadata = &adapter.index().metadata;
        if metadata.get("adapter.type").and_then(Value::as_str) != Some("lora") {
            return Err("adapter must be a GGUF LoRA file".into());
        }
        if metadata
            .get("general.architecture")
            .and_then(Value::as_str)
            .is_some_and(|arch| arch != self.config.arch)
        {
            return Err("LoRA architecture does not match the loaded model".into());
        }
        if metadata.contains_key("adapter.alora.invocation_tokens") {
            return Err(
                "activated LoRA invocation tokens are not supported by load-time merging".into(),
            );
        }
        let alpha = metadata
            .get("adapter.lora.alpha")
            .and_then(Value::as_f64)
            .ok_or("LoRA adapter is missing adapter.lora.alpha")? as f32;
        if !alpha.is_finite() || alpha < 0.0 {
            return Err("invalid LoRA alpha".into());
        }
        let known = self.projection_names();
        for adapter_tensor in &adapter.index().tensors {
            let base = adapter_tensor
                .name
                .strip_suffix(".lora_a")
                .or_else(|| adapter_tensor.name.strip_suffix(".lora_b"))
                .ok_or_else(|| format!("unsupported LoRA tensor {}", adapter_tensor.name))?;
            if base == "token_embd.weight" {
                return Err("embedding LoRA is not supported by the projection merge path".into());
            }
            if !known.iter().any(|name| name == base) {
                return Err(format!(
                    "LoRA target {base} is not a supported linear projection in this model"
                ));
            }
            let a = tensor(&adapter, &format!("{base}.lora_a"))?;
            let b = tensor(&adapter, &format!("{base}.lora_b"))?;
            let descriptor = tensor(&self.mapping, base)?;
            if a.dimensions.len() != 2
                || b.dimensions.len() != 2
                || a.dimensions[0] != descriptor.dimensions[0]
                || b.dimensions[1] != descriptor.dimensions[1]
                || a.dimensions[1] != b.dimensions[0]
                || a.dimensions[1] == 0
            {
                return Err(format!(
                    "incompatible LoRA rank or matrix dimensions for {base}"
                ));
            }
        }
        // Validate and compute every target before changing the model.
        let mut replacements = HashMap::<String, Vec<f32>>::new();
        for name in &known {
            let a_name = format!("{name}.lora_a");
            if !has_tensor(&adapter, &a_name) {
                continue;
            }
            let descriptor = tensor(&self.mapping, name)?;
            let rows = descriptor.dimensions[1] as usize;
            let cols = descriptor.dimensions[0] as usize;
            let rank = tensor(&adapter, &a_name)?.dimensions[1] as usize;
            let a = tensor_f32(&adapter, &a_name)?;
            let b = tensor_f32(&adapter, &format!("{name}.lora_b"))?;
            let mut weights = match self.adapted.get(name) {
                Some(values) => values.clone(),
                None => tensor_f32(&self.mapping, name)?,
            };
            merge_lora_delta(
                &mut weights,
                &a,
                &b,
                rows,
                cols,
                rank,
                scale * alpha / rank as f32,
            )?;
            replacements.insert(name.clone(), weights);
        }
        let count = replacements.len();
        if count == 0 {
            return Err("LoRA adapter contains no compatible projections".into());
        }
        self.adapted.extend(replacements);
        Ok(count)
    }

    pub fn mapping(&self) -> &MappedGguf {
        &self.mapping
    }
    /// Owned load-time adapter weights for a projection, if a LoRA touched it.
    pub fn adapted_projection(&self, name: &str) -> Option<Vec<f32>> {
        self.adapted.get(name).cloned()
    }
    pub fn chat_template(&self) -> &str {
        &self.chat_template
    }

    /// The named `tool_use` chat template (`tokenizer.chat_template.tool_use`), empty when the checkpoint has none.
    pub fn tool_chat_template(&self) -> &str {
        &self.tool_chat_template
    }
    /// Recurrent layers keep per-sequence state outside the KV cache, so prefix sharing and
    /// speculative rollback are unavailable.
    pub fn is_stateful(&self) -> bool {
        self.config.delta.is_some()
            || self.config.mamba.is_some()
            || self.config.mamba1.is_some()
            || self.config.hybrid.stateful()
            || self
                .config
                .shortconv
                .as_ref()
                .is_some_and(|s| s.layers.iter().any(|&conv| conv))
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
    pub fn model_arch(&self) -> &str {
        &self.config.arch
    }
    pub fn vocab_size(&self) -> usize {
        self.config.vocab
    }
    pub fn max_context(&self) -> usize {
        self.max_context
    }
    pub fn max_sequences(&self) -> usize {
        self.max_sequences
    }
    pub fn block_size(&self) -> usize {
        self.block_size
    }
    pub fn table_stride(&self) -> usize {
        self.table_stride
    }
    pub fn pool_blocks(&self) -> usize {
        self.pool_blocks
    }
    /// Switches a multi-head latent attention model to the latent KV cache (`FLLM_MLA=latent`, CUDA decoder only): one KV head of
    /// `kv_lora + rope` values per token and 64-token pages instead of per-head keys and values (see `crate::mla_latent`). The cache
    /// geometry in the configuration changes, so every consumer (memory plan, pools, copy of shared prompt blocks) sizes itself from it.
    /// Call before the memory plan; the CPU decoder cannot run a model in this state.
    pub fn enable_mla_latent(&mut self) -> Result<(), String> {
        if self.mla_latent {
            return Ok(());
        }
        let m = self.config.mla.clone().ok_or("the model has no multi-head latent attention")?;
        let (kv_heads, head_dim, value_dim) = crate::mla_latent::latent_geometry(&m);
        self.config.kv_heads = kv_heads;
        self.config.head_dim = head_dim;
        self.config.value_dim = value_dim;
        self.block_size = crate::mla_latent::LATENT_BLOCK_SIZE;
        self.table_stride = self.max_context.div_ceil(self.block_size);
        self.pool_blocks = self.table_stride.checked_mul(self.max_sequences).ok_or("KV block count overflow")?;
        self.mla_latent = true;
        Ok(())
    }
    pub fn mla_latent(&self) -> bool {
        self.mla_latent
    }
    /// Lowers the context window (the device cannot hold the KV cache of a longer sequence). The rope tables built for
    /// the larger window stay valid.
    pub fn limit_context(&mut self, tokens: usize) {
        let tokens = tokens.clamp(self.block_size, self.max_context);
        self.max_context = tokens;
        self.table_stride = tokens.div_ceil(self.block_size);
        self.pool_blocks = self.pool_blocks.min(self.table_stride.saturating_mul(self.max_sequences));
    }
    /// Shrinks the KV pool (never below one full-context sequence).
    pub fn set_pool_blocks(&mut self, blocks: usize) {
        self.pool_blocks = blocks.clamp(self.table_stride, self.pool_blocks.max(self.table_stride));
    }
    pub fn layer_count(&self) -> usize {
        self.config.layers
    }
    pub fn hidden_size(&self) -> usize {
        self.config.hidden
    }
    pub fn heads(&self) -> usize {
        self.config.heads
    }
    pub fn kv_heads(&self) -> usize {
        self.config.kv_heads
    }
    pub fn threads(&self) -> i32 {
        self.threads
    }
    /// Changes the CPU worker count; only effective before the runtime is built from this model.
    pub fn set_threads(&mut self, threads: i32) {
        self.threads = threads.max(1);
    }
    /// Workers for batches of many rows (prompts); never fewer than `threads`.
    pub fn prefill_threads(&self) -> i32 {
        self.prefill_threads.max(self.threads)
    }
    pub fn set_prefill_threads(&mut self, threads: i32) {
        self.prefill_threads = threads.max(0);
    }
    /// Rotary tables: index 0 for global layers, 1 (when present) for sliding-window layers.
    pub fn rope_tables(&self) -> &[RopeTables] {
        &self.rope
    }
    /// Index into `rope_tables()` of the table layer `layer` uses: its own base when the model lists one per layer, the
    /// sliding-window table for sliding-window layers, otherwise the global table.
    pub fn rope_index(&self, layer: usize) -> usize {
        if let Some(base) = self.config.layer_rope_bases.as_ref().and_then(|b| b.get(layer)).filter(|b| **b > 0.0) {
            return self.rope_bases.iter().position(|b| b == base).unwrap_or(0);
        }
        usize::from(self.config.sliding_window.is_some() && self.config.layer_window(layer).is_some() && self.rope.len() > 1)
    }
    pub fn rope_shared(&self) -> Arc<Vec<RopeTables>> {
        Arc::clone(&self.rope)
    }
    /// Packed KV cache formats (see `crate::kv_format`): 1 = f16 keys and values, 2 = q8_0 keys and values.
    pub fn set_kv_kind(&mut self, kv_kind: i32) {
        self.kv_kind = kv_kind;
    }
    pub fn set_kv_formats(&mut self, key: crate::kv_format::KvFormat, value: crate::kv_format::KvFormat) {
        self.kv_kind = crate::kv_format::pack_kind(key, value);
    }
    pub fn kv_kind(&self) -> i32 {
        self.kv_kind
    }
    /// (key, value) cache formats.
    pub fn kv_formats(&self) -> (crate::kv_format::KvFormat, crate::kv_format::KvFormat) {
        use crate::kv_format::KvFormat;
        crate::kv_format::unpack_kind(self.kv_kind).unwrap_or((KvFormat::F16, KvFormat::F16))
    }
    /// 16-bit words of KV storage of the first layer for `kv_heads` heads of width `dim` in `format`.
    pub fn kv_words_per_layer(&self, format: crate::kv_format::KvFormat, dim: usize) -> usize {
        let slots = self.pool_blocks * self.block_size;
        format.layer_words(slots, self.config.kv_heads, dim)
    }
    /// Offsets and sizes of every layer's region in the key and value pools.
    pub fn kv_pool_layout(&self) -> crate::kv_geometry::KvPoolLayout {
        self.kv_pool_layout_for(&self.config)
    }
    /// The same for another view of the configuration (one rank of a tensor-parallel group sees a share of the heads).
    pub fn kv_pool_layout_for(&self, config: &ModelConfig) -> crate::kv_geometry::KvPoolLayout {
        let (key, value) = self.kv_formats();
        crate::kv_geometry::KvPoolLayout::new(config, key, value, self.pool_blocks * self.block_size)
    }
    pub fn key_buffer_len(&self) -> Result<usize, String> {
        self.kv_pool_layout_checked().map(|l| l.key_total)
    }
    pub fn value_buffer_len(&self) -> Result<usize, String> {
        self.kv_pool_layout_checked().map(|l| l.value_total)
    }
    fn kv_pool_layout_checked(&self) -> Result<crate::kv_geometry::KvPoolLayout, String> {
        let slots = self.pool_blocks.checked_mul(self.block_size).ok_or("KV cache allocation size overflow")?;
        let widest = (0..self.config.layers)
            .map(|l| self.config.layer_kv_heads(l) * self.config.layer_head_dim(l).max(self.config.layer_value_dim(l)))
            .max()
            .unwrap_or(0);
        slots
            .checked_mul(widest)
            .and_then(|n| n.checked_mul(self.config.layers))
            .ok_or("KV cache allocation size overflow")?;
        Ok(self.kv_pool_layout())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lora_merge_computes_scaled_ba_without_modifying_adapter() {
        let mut base = [1.0, 2.0, 3.0, 4.0];
        merge_lora_delta(&mut base, &[2.0, 3.0], &[4.0, 5.0], 2, 2, 1, 0.5).unwrap();
        assert_eq!(base, [5.0, 8.0, 8.0, 11.5]);
        assert!(merge_lora_delta(&mut base, &[1.0], &[1.0], 2, 2, 1, 1.0).is_err());
    }
}
