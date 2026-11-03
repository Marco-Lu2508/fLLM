//! The planner for one node of identical GPUs (8x MI355X, 8x B200 / H200 / H100 ...) and large models: which mix of tensor (TP), expert (EP),
//! pipeline (PP) and attention-data (DP) parallelism, how much memory every rank needs, and a first-order estimate of the decode step.
//!
//! Pure arithmetic over a [`ModelShape`] and a [`GpuSpec`], so it is tested over many configurations without any hardware.
//! Nothing here is measured: the step times come from a roofline model (weights read from HBM + collectives over the link, see
//! [`estimate_step`]) and are only meant to rank layouts against each other. Every plan also says whether this build of fLLM can run the
//! layout today (`runnable_today`, with the reason when not): attention data parallelism and the replicated multi-head latent attention that
//! DeepSeek-class models want are planned but not implemented.
//!
//! * Dense models: TP over the node (all heads divide), PP only when TP alone does not fit or the sizes do not divide.
//! * Mixture-of-experts with grouped-query attention: attention TP + expert parallelism over the same group (hybrid TP x EP).
//! * Mixture-of-experts with multi-head latent attention (DeepSeek-V3/V4, Kimi K2/K3, GLM-5): the KV cache is one small latent per token that
//!   tensor parallelism would have to replicate on every rank, so the ideal layout is attention data parallelism + expert parallelism.
use super::plan::attention_split;

const GIB: f64 = (1u64 << 30) as f64;

/// How the GPUs of the node talk to each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fabric {
    NvLink,
    /// AMD Infinity Fabric (XGMI) mesh.
    Xgmi,
    Pcie,
}

impl Fabric {
    pub fn name(self) -> &'static str {
        match self {
            Fabric::NvLink => "NVLink",
            Fabric::Xgmi => "Infinity Fabric (XGMI)",
            Fabric::Pcie => "PCIe",
        }
    }
}

/// One GPU model. Figures are the vendor's published numbers (usable memory is what the driver reports, a little below the marketing size);
/// `link_gbps` is the per-GPU bandwidth in one direction to the rest of the node, used only for the relative ranking of layouts.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuSpec {
    pub name: &'static str,
    pub arch: &'static str,
    pub mem_gib: f64,
    pub hbm_gbps: f64,
    pub fabric: Fabric,
    pub link_gbps: f64,
    /// Latency of one small collective in microseconds (launch + synchronisation).
    pub collective_us: f64,
    pub fp8: bool,
    pub fp4: bool,
}

pub fn gpu_presets() -> Vec<GpuSpec> {
    vec![
        GpuSpec { name: "mi355x", arch: "gfx950", mem_gib: 288.0, hbm_gbps: 8000.0, fabric: Fabric::Xgmi, link_gbps: 448.0, collective_us: 15.0, fp8: true, fp4: true },
        GpuSpec { name: "mi325x", arch: "gfx942", mem_gib: 256.0, hbm_gbps: 6000.0, fabric: Fabric::Xgmi, link_gbps: 448.0, collective_us: 15.0, fp8: true, fp4: false },
        GpuSpec { name: "mi300x", arch: "gfx942", mem_gib: 192.0, hbm_gbps: 5300.0, fabric: Fabric::Xgmi, link_gbps: 448.0, collective_us: 15.0, fp8: true, fp4: false },
        GpuSpec { name: "b300", arch: "sm_103", mem_gib: 279.0, hbm_gbps: 8000.0, fabric: Fabric::NvLink, link_gbps: 900.0, collective_us: 8.0, fp8: true, fp4: true },
        GpuSpec { name: "b200", arch: "sm_100", mem_gib: 179.0, hbm_gbps: 8000.0, fabric: Fabric::NvLink, link_gbps: 900.0, collective_us: 8.0, fp8: true, fp4: true },
        GpuSpec { name: "h200", arch: "sm_90", mem_gib: 140.0, hbm_gbps: 4800.0, fabric: Fabric::NvLink, link_gbps: 450.0, collective_us: 8.0, fp8: true, fp4: false },
        GpuSpec { name: "h100", arch: "sm_90", mem_gib: 79.0, hbm_gbps: 3350.0, fabric: Fabric::NvLink, link_gbps: 450.0, collective_us: 8.0, fp8: true, fp4: false },
        GpuSpec { name: "a100", arch: "sm_80", mem_gib: 79.0, hbm_gbps: 2000.0, fabric: Fabric::NvLink, link_gbps: 300.0, collective_us: 8.0, fp8: false, fp4: false },
        GpuSpec { name: "l40s", arch: "sm_89", mem_gib: 44.0, hbm_gbps: 864.0, fabric: Fabric::Pcie, link_gbps: 32.0, collective_us: 25.0, fp8: true, fp4: false },
        GpuSpec { name: "rtx2000ada", arch: "sm_89", mem_gib: 7.5, hbm_gbps: 256.0, fabric: Fabric::Pcie, link_gbps: 16.0, collective_us: 25.0, fp8: true, fp4: false },
    ]
}

/// The preset that matches a device name such as `AMD Instinct MI355X` or `NVIDIA B200`; an unknown device gets a generic spec from its
/// reported memory and bandwidth.
pub fn gpu_from_label(label: &str, backend: &str, total_bytes: u64, speed_gbs: f64) -> GpuSpec {
    let flat: String = label.chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>().to_ascii_lowercase();
    // longest names first, so that "mi300x" is not taken for part of a longer name
    let mut presets = gpu_presets();
    presets.sort_by_key(|g| std::cmp::Reverse(g.name.len()));
    if let Some(found) = presets.into_iter().find(|g| flat.contains(g.name)) {
        return found;
    }
    let amd = backend == "hip";
    GpuSpec {
        name: "device",
        arch: "unknown",
        mem_gib: total_bytes as f64 / GIB,
        hbm_gbps: speed_gbs.max(50.0),
        fabric: Fabric::Pcie,
        link_gbps: if amd { 64.0 } else { 32.0 },
        collective_us: 25.0,
        fp8: false,
        fp4: false,
    }
}

pub fn gpu_preset(name: &str) -> Option<GpuSpec> {
    let wanted: String = name.chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>().to_ascii_lowercase();
    gpu_presets().into_iter().find(|g| g.name == wanted)
}

/// Storage format of the weights: bytes per parameter including block scales.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightFormat {
    Bf16,
    /// FP8 with 128x128 block scales.
    Fp8,
    /// NVFP4 / MXFP4: 4-bit values + one 8-bit scale per 16 (NVFP4) or 32 (MXFP4) values.
    Fp4,
    Q4K,
    Q8,
}

impl WeightFormat {
    pub fn bytes_per_param(self) -> f64 {
        match self {
            WeightFormat::Bf16 => 2.0,
            WeightFormat::Fp8 => 1.0 + 1.0 / 128.0,
            WeightFormat::Fp4 => 0.5 + 1.0 / 16.0,
            WeightFormat::Q4K => 0.5625,
            WeightFormat::Q8 => 1.0625,
        }
    }
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "bf16" | "fp16" | "f16" => Some(Self::Bf16),
            "fp8" | "f8" => Some(Self::Fp8),
            "fp4" | "nvfp4" | "mxfp4" => Some(Self::Fp4),
            "q4k" | "q4_k" | "q4_k_m" => Some(Self::Q4K),
            "q8" | "q8_0" => Some(Self::Q8),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            WeightFormat::Bf16 => "bf16",
            WeightFormat::Fp8 => "fp8",
            WeightFormat::Fp4 => "fp4",
            WeightFormat::Q4K => "q4_k",
            WeightFormat::Q8 => "q8_0",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MlaShape {
    pub q_lora: usize,
    pub kv_lora: usize,
    pub nope: usize,
    pub rope: usize,
    pub value: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MoeShape {
    pub experts: usize,
    pub used: usize,
    pub expert_ff: usize,
    /// Always-active shared experts (their width is `shared_ff`).
    pub shared_ff: usize,
    /// Leading layers with an ordinary dense FFN of width `dense_ff`.
    pub dense_layers: usize,
    pub dense_ff: usize,
    /// Only every `moe_every`-th layer after the dense ones is a mixture-of-experts layer (Llama 4: 2).
    pub moe_every: usize,
}

/// The sizes the planner needs; parameter counts are derived from them.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelShape {
    pub name: String,
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    /// Dense feed-forward width (dense models).
    pub ff: usize,
    pub mla: Option<MlaShape>,
    pub moe: Option<MoeShape>,
    pub tied_embeddings: bool,
}

/// Parameter counts by class.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ParamCounts {
    pub attention: f64,
    pub dense_ffn: f64,
    pub routed_experts: f64,
    pub shared_experts: f64,
    pub vocab: f64,
}

impl ParamCounts {
    pub fn total(&self) -> f64 {
        self.attention + self.dense_ffn + self.routed_experts + self.shared_experts + self.vocab
    }
}

impl ModelShape {
    pub fn moe_layers(&self) -> usize {
        match &self.moe {
            None => 0,
            Some(m) => {
                let after = self.layers.saturating_sub(m.dense_layers);
                after.div_ceil(m.moe_every.max(1))
            }
        }
    }

    fn attention_params_per_layer(&self) -> f64 {
        let h = self.hidden as f64;
        match &self.mla {
            Some(m) => {
                let heads = self.heads as f64;
                let qk = (m.nope + m.rope) as f64;
                let q = if m.q_lora > 0 { h * m.q_lora as f64 + m.q_lora as f64 * heads * qk } else { h * heads * qk };
                q + h * (m.kv_lora + m.rope) as f64 + m.kv_lora as f64 * heads * (m.nope + m.value) as f64 + heads * m.value as f64 * h
            }
            None => {
                let hd = self.head_dim as f64;
                h * (self.heads as f64 * hd + 2.0 * self.kv_heads as f64 * hd) + self.heads as f64 * hd * h
            }
        }
    }

    pub fn params(&self) -> ParamCounts {
        let h = self.hidden as f64;
        let layers = self.layers as f64;
        let attention = self.attention_params_per_layer() * layers;
        let vocab = self.vocab as f64 * h * if self.tied_embeddings { 1.0 } else { 2.0 };
        match &self.moe {
            None => ParamCounts { attention, dense_ffn: 3.0 * h * self.ff as f64 * layers, vocab, ..Default::default() },
            Some(m) => {
                let moe_layers = self.moe_layers() as f64;
                let dense_layers = layers - moe_layers;
                ParamCounts {
                    attention,
                    dense_ffn: 3.0 * h * m.dense_ff as f64 * dense_layers,
                    routed_experts: 3.0 * h * m.expert_ff as f64 * m.experts as f64 * moe_layers,
                    shared_experts: 3.0 * h * m.shared_ff as f64 * moe_layers,
                    vocab,
                }
            }
        }
    }

    /// Active parameters per token (what one decode token reads), for the summary line.
    pub fn active_params(&self) -> f64 {
        let p = self.params();
        match &self.moe {
            None => p.total(),
            Some(m) => p.total() - p.routed_experts * (1.0 - m.used as f64 / m.experts as f64),
        }
    }

    /// KV cache bytes per token and layer for the whole (unsplit) model.
    pub fn kv_bytes_per_token_layer(&self, kv_bytes: f64) -> f64 {
        match &self.mla {
            Some(m) => (m.kv_lora + m.rope) as f64 * kv_bytes,
            None => 2.0 * self.kv_heads as f64 * self.head_dim as f64 * kv_bytes,
        }
    }

    pub fn is_mla(&self) -> bool {
        self.mla.is_some()
    }

    /// Public configurations as remembered (rounded): meant for planning examples, check them against the model's own `config.json`.
    pub fn preset(name: &str) -> Option<Self> {
        let key: String = name.chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>().to_ascii_lowercase();
        let mla = |q_lora, kv_lora| Some(MlaShape { q_lora, kv_lora, nope: 128, rope: 64, value: 128 });
        let shape = match key.as_str() {
            "deepseekv3" | "deepseekr1" => Self {
                name: "DeepSeek-V3 / R1 (671B)".into(),
                layers: 61,
                hidden: 7168,
                heads: 128,
                kv_heads: 128,
                head_dim: 192,
                vocab: 129_280,
                ff: 18_432,
                mla: mla(1536, 512),
                moe: Some(MoeShape { experts: 256, used: 8, expert_ff: 2048, shared_ff: 2048, dense_layers: 3, dense_ff: 18_432, moe_every: 1 }),
                tied_embeddings: false,
            },
            "kimik2" => Self {
                name: "Kimi K2 (1T)".into(),
                layers: 61,
                hidden: 7168,
                heads: 64,
                kv_heads: 64,
                head_dim: 192,
                vocab: 163_840,
                ff: 18_432,
                mla: mla(1536, 512),
                moe: Some(MoeShape { experts: 384, used: 8, expert_ff: 2048, shared_ff: 2048, dense_layers: 1, dense_ff: 18_432, moe_every: 1 }),
                tied_embeddings: false,
            },
            "qwen3235b" | "qwen3moe" => Self {
                name: "Qwen3-235B-A22B".into(),
                layers: 94,
                hidden: 4096,
                heads: 64,
                kv_heads: 4,
                head_dim: 128,
                vocab: 151_936,
                ff: 12_288,
                mla: None,
                moe: Some(MoeShape { experts: 128, used: 8, expert_ff: 1536, shared_ff: 0, dense_layers: 0, dense_ff: 12_288, moe_every: 1 }),
                tied_embeddings: false,
            },
            "llama4maverick" => Self {
                name: "Llama 4 Maverick (400B)".into(),
                layers: 48,
                hidden: 5120,
                heads: 40,
                kv_heads: 8,
                head_dim: 128,
                vocab: 202_048,
                ff: 16_384,
                mla: None,
                moe: Some(MoeShape { experts: 128, used: 1, expert_ff: 8192, shared_ff: 8192, dense_layers: 0, dense_ff: 16_384, moe_every: 2 }),
                tied_embeddings: false,
            },
            "gptoss120b" => Self {
                name: "gpt-oss-120b".into(),
                layers: 36,
                hidden: 2880,
                heads: 64,
                kv_heads: 8,
                head_dim: 64,
                vocab: 201_088,
                ff: 2880,
                mla: None,
                moe: Some(MoeShape { experts: 128, used: 4, expert_ff: 2880, shared_ff: 0, dense_layers: 0, dense_ff: 2880, moe_every: 1 }),
                tied_embeddings: false,
            },
            "llama370b" | "llama3170b" => Self {
                name: "Llama 3.x 70B".into(),
                layers: 80,
                hidden: 8192,
                heads: 64,
                kv_heads: 8,
                head_dim: 128,
                vocab: 128_256,
                ff: 28_672,
                mla: None,
                moe: None,
                tied_embeddings: false,
            },
            "llama3405b" | "llama31405b" | "llama405b" => Self {
                name: "Llama 3.1 405B".into(),
                layers: 126,
                hidden: 16_384,
                heads: 128,
                kv_heads: 8,
                head_dim: 128,
                vocab: 128_256,
                ff: 53_248,
                mla: None,
                moe: None,
                tied_embeddings: false,
            },
            // a hypothetical trillion-parameter MoE with grouped-query attention, for sizing
            "moe1t" | "1tmoe" => Self {
                name: "1T-parameter MoE (GQA, 384 experts)".into(),
                layers: 62,
                hidden: 8192,
                heads: 64,
                kv_heads: 8,
                head_dim: 128,
                vocab: 160_000,
                ff: 24_576,
                mla: None,
                moe: Some(MoeShape { experts: 384, used: 8, expert_ff: 1792, shared_ff: 1792, dense_layers: 2, dense_ff: 24_576, moe_every: 1 }),
                tied_embeddings: false,
            },
            _ => return None,
        };
        Some(shape)
    }

    pub fn preset_names() -> &'static [&'static str] {
        &["deepseek-v3", "kimi-k2", "qwen3-235b", "llama4-maverick", "gpt-oss-120b", "llama3-70b", "llama3-405b", "moe-1t"]
    }

    /// From the configuration of a loaded model.
    pub fn from_config(cfg: &crate::config::ModelConfig) -> Self {
        let mla = cfg.mla.as_ref().map(|m| MlaShape { q_lora: m.q_lora.unwrap_or(0), kv_lora: m.kv_lora, nope: m.nope, rope: m.rope, value: m.value });
        let moe = cfg.moe.as_ref().map(|m| MoeShape {
            experts: m.experts,
            used: m.used,
            expert_ff: m.expert_ff,
            shared_ff: m.shared_ff * m.shared.max(usize::from(m.shared_ff > 0)),
            dense_layers: m.dense_layers,
            dense_ff: cfg.ff,
            moe_every: 1,
        });
        Self {
            name: cfg.arch.clone(),
            layers: cfg.layers,
            hidden: cfg.hidden,
            heads: cfg.heads,
            kv_heads: cfg.kv_heads,
            head_dim: cfg.head_dim,
            vocab: cfg.vocab,
            ff: cfg.ff,
            mla,
            moe,
            tied_embeddings: cfg.tied_embeddings,
        }
    }
}

/// What the plan has to achieve.
#[derive(Clone, Debug, PartialEq)]
pub struct Workload {
    /// Context length of one sequence.
    pub ctx: usize,
    /// Sequences that should fit in the KV cache at full context at the same time.
    pub seqs: usize,
    /// Decode tokens per step used for the step-time estimate (concurrent sequences).
    pub batch: usize,
    pub kv_bytes: f64,
    /// Fraction of the memory the plan may use (`--gpu-memory-utilization`).
    pub utilization: f64,
    /// Tokens of the largest prefill chunk / step (`--max-num-tokens`): sizes the activation buffers.
    pub max_num_tokens: usize,
    pub weights: WeightFormat,
    /// Measured bytes per parameter of a real model file (replaces the figure of `weights`).
    pub bpp_override: Option<f64>,
}

impl Workload {
    pub fn bpp(&self) -> f64 {
        self.bpp_override.unwrap_or_else(|| self.weights.bytes_per_param())
    }
}

impl Default for Workload {
    fn default() -> Self {
        Self { ctx: 32_768, seqs: 16, batch: 64, kv_bytes: 2.0, utilization: 0.90, max_num_tokens: 8192, weights: WeightFormat::Fp8, bpp_override: None }
    }
}

/// One layout of the node.
#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    pub world: usize,
    pub pp: usize,
    /// Devices per pipeline stage (`world / pp`).
    pub group: usize,
    /// Tensor-parallel size of the attention inside a stage group; `group / attn_tp` replicas serve different sequences (attention DP).
    pub attn_tp: usize,
    /// Expert parallelism over the group (`group / ep` = tensor split inside every expert).
    pub ep: usize,
}

impl Layout {
    pub fn attn_dp(&self) -> usize {
        self.group / self.attn_tp
    }
    pub fn moe_tp(&self) -> usize {
        self.group / self.ep
    }
    pub fn describe(&self) -> String {
        let mut s = format!("TP{} x PP{}", self.group, self.pp);
        if self.attn_tp != self.group {
            s = format!("attention TP{} x DP{}, {}", self.attn_tp, self.attn_dp(), s);
        }
        if self.ep > 1 {
            s.push_str(&format!(", EP{}", self.ep));
            if self.moe_tp() > 1 {
                s.push_str(&format!(" x expert-TP{}", self.moe_tp()));
            }
        }
        s
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RankMemoryPlan {
    pub weights: f64,
    pub activations: f64,
    pub workspace: f64,
    pub kv_budget: f64,
    pub usable: f64,
    /// Tokens of KV cache the node holds in total with this layout.
    pub kv_tokens: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub layout: Layout,
    pub memory: RankMemoryPlan,
    pub fits: bool,
    /// Estimated decode step in milliseconds for `Workload::batch` tokens (roofline; not measured).
    pub step_ms: f64,
    pub tokens_per_s: f64,
    pub runnable_today: bool,
    pub notes: Vec<String>,
}

fn divisors(n: usize) -> Vec<usize> {
    (1..=n).filter(|d| n.is_multiple_of(*d)).collect()
}

/// Whether this build can execute the layout, and why not.
pub fn runnable_today(shape: &ModelShape, l: &Layout) -> Result<(), String> {
    if shape.is_mla() && l.group > 1 {
        return Err("multi-head latent attention cannot be split inside a stage yet (pipeline parallelism only); attention data parallelism / replicated MLA with expert parallelism is planned".into());
    }
    if l.attn_tp != l.group {
        return Err("attention data parallelism (different sequences on different ranks of a group) is not implemented".into());
    }
    if shape.moe.is_none() && l.ep > 1 {
        return Err("expert parallelism needs a mixture-of-experts model".into());
    }
    if l.ep > 1 && l.moe_tp() > 1 {
        return Err("expert parallelism combined with tensor-split experts inside the group is not implemented (use all experts distributed or all split)".into());
    }
    Ok(())
}

/// Memory per rank for a layout.
pub fn rank_memory(shape: &ModelShape, gpu: &GpuSpec, w: &Workload, l: &Layout) -> RankMemoryPlan {
    let p = shape.params();
    let bpp = w.bpp();
    let stage_frac = 1.0 / l.pp as f64; // layers of this stage (the vocabulary parts are added to the first / last stage; the larger one is taken)
    let g = l.group as f64;
    // K/V projections are replicated when there are fewer KV heads than attention ranks (GQA/MQA)
    let attn = if shape.is_mla() {
        // the MLA down projections (compressed latents) are small; the head-wise up projections split with attn_tp
        p.attention * stage_frac * bpp / l.attn_tp as f64
    } else {
        let kv_rep = (l.attn_tp as f64 / shape.kv_heads as f64).max(1.0);
        let hd = shape.head_dim as f64;
        let h = shape.hidden as f64;
        let kv_params = 2.0 * h * shape.kv_heads as f64 * hd * shape.layers as f64;
        let rest = p.attention - kv_params;
        (rest / l.attn_tp as f64 + kv_params * kv_rep / l.attn_tp as f64) * stage_frac * bpp
    };
    let dense = p.dense_ffn * stage_frac * bpp / g;
    let experts = p.routed_experts * stage_frac * bpp / g; // ep x moe_tp = group: every rank holds 1/group of the expert weights
    let shared = p.shared_experts * stage_frac * bpp / g;
    // the output head (and the embedding rows the stage needs) are split by vocabulary over the group; the larger of first / last stage
    let vocab = p.vocab * bpp / g;
    let weights = attn + dense + experts + shared + vocab;
    let usable = gpu.mem_gib * GIB * w.utilization;
    // activations: hidden states and the intermediate buffers of one step of max_num_tokens, plus the MoE dispatch buffer
    let tokens = w.max_num_tokens as f64;
    let hidden = shape.hidden as f64;
    let mut activations = tokens * hidden * 2.0 * 16.0;
    if let Some(m) = &shape.moe {
        activations += tokens * m.used as f64 * hidden * 2.0 * 2.0 + tokens * m.used as f64 * m.expert_ff as f64 * 2.0 * 2.0 / l.moe_tp() as f64;
    }
    activations += tokens * (shape.vocab as f64 / g) * 4.0;
    let workspace = (1.5 * GIB).max(0.02 * gpu.mem_gib * GIB) + if l.group > 1 { 1.0 * GIB } else { 0.0 };
    let kv_budget = (usable - weights - activations - workspace).max(0.0);
    // KV bytes per token on one rank: the layers of its stage, the heads (or the whole latent) it caches
    let layers_stage = (shape.layers as f64 * stage_frac).ceil();
    let per_token_rank = match &shape.mla {
        Some(m) => (m.kv_lora + m.rope) as f64 * w.kv_bytes * layers_stage, // replicated over the attention TP ranks
        None => {
            let local_heads = (shape.kv_heads as f64 / l.attn_tp as f64).ceil();
            2.0 * local_heads * shape.head_dim as f64 * w.kv_bytes * layers_stage
        }
    };
    let kv_tokens = if per_token_rank > 0.0 { kv_budget / per_token_rank * l.attn_dp() as f64 } else { 0.0 };
    RankMemoryPlan { weights, activations, workspace, kv_budget, usable, kv_tokens }
}

/// Roofline estimate of one decode step of `w.batch` tokens, in seconds. Weight reads from HBM + collectives over the link + a fixed cost
/// per layer; compute is ignored (decode is memory-bound at these batch sizes). Only meant to rank layouts against each other.
pub fn estimate_step(shape: &ModelShape, gpu: &GpuSpec, w: &Workload, l: &Layout) -> f64 {
    let bpp = w.bpp();
    let p = shape.params();
    let b = w.batch as f64;
    let g = l.group as f64;
    let layers = shape.layers as f64;
    let hbm = gpu.hbm_gbps * 1e9;
    let link = gpu.link_gbps * 1e9;
    let coll = gpu.collective_us * 1e-6;
    let act_bytes = b * shape.hidden as f64 * 2.0;
    let allreduce = |n: usize, bytes: f64| -> f64 {
        if n <= 1 {
            0.0
        } else {
            coll + 2.0 * (n as f64 - 1.0) / n as f64 * bytes / link
        }
    };
    let moe_layers = shape.moe_layers() as f64;
    let attn_layer = p.attention / layers * bpp / l.attn_tp as f64;
    // KV cache read by the attention of one layer: the sequences of this attention replica, half the context on average
    let kv_token_layer = match &shape.mla {
        Some(m) => (m.kv_lora + m.rope) as f64 * w.kv_bytes,
        None => 2.0 * (shape.kv_heads as f64 / l.attn_tp as f64).ceil() * shape.head_dim as f64 * w.kv_bytes,
    };
    let kv_read = b * (w.ctx as f64 / 2.0) * kv_token_layer / l.attn_dp() as f64 / hbm;
    let mut per_layer_fixed = 12e-6 + kv_read;
    per_layer_fixed += allreduce(l.attn_tp, act_bytes);
    let mut total = 0.0;
    let dense_layers = layers - moe_layers;
    // dense layers (all of them for a dense model)
    if dense_layers > 0.0 {
        let ffn = if shape.moe.is_none() { p.dense_ffn / layers } else { p.dense_ffn / dense_layers.max(1.0) } * bpp / g;
        let t = (attn_layer + ffn) / hbm + per_layer_fixed + allreduce(l.group, act_bytes);
        total += dense_layers * t;
    }
    if let Some(m) = &shape.moe {
        if moe_layers > 0.0 {
            let per_expert = 3.0 * shape.hidden as f64 * m.expert_ff as f64 * bpp / l.moe_tp() as f64;
            let experts_per_rank = m.experts as f64 / l.ep as f64;
            // distinct experts a rank touches in one step (balls into bins over all b * used picks)
            let hit_fraction = 1.0 - (1.0 - m.used as f64 / m.experts as f64).powf(b);
            let experts_read = experts_per_rank * hit_fraction;
            let shared = p.shared_experts / moe_layers.max(1.0) * bpp / g;
            let mem = (attn_layer + shared + experts_read * per_expert) / hbm;
            // communication of the layer: attention part (counted in per_layer_fixed) + the MoE exchange
            let moe_comm = if l.attn_dp() > 1 {
                // attention DP: gather the tokens of the other replicas before the experts, reduce-scatter after
                2.0 * (coll + (g - 1.0) / g * act_bytes / link)
            } else if l.ep > 1 {
                // all-to-all dispatch + combine moves b * used rows split over the group; the all-reduce combine moves b rows twice
                let a2a = 2.0 * (coll + b * m.used as f64 * shape.hidden as f64 * 2.0 / g * (g - 1.0) / g / link);
                a2a.min(allreduce(l.group, act_bytes))
            } else {
                allreduce(l.group, act_bytes)
            };
            total += moe_layers * (mem + per_layer_fixed + moe_comm + allreduce(l.moe_tp(), act_bytes) * f64::from(u8::from(l.ep > 1 && l.moe_tp() > 1)));
        }
    }
    // pipeline: activations travel between stages and the stages take turns for one batch
    total += (l.pp as f64 - 1.0) * (coll + act_bytes / link);
    // output head
    total += p.vocab / 2.0 * bpp / g / hbm + allreduce(l.group, b * (shape.vocab as f64 / g) * 4.0 / 4.0);
    total
}

/// All layouts for `world` identical devices that divide the model (not yet filtered by memory).
pub fn candidate_layouts(shape: &ModelShape, world: usize) -> Vec<Layout> {
    let mut out = Vec::new();
    for pp in divisors(world) {
        if pp > shape.layers {
            continue;
        }
        let group = world / pp;
        for attn_tp in divisors(group) {
            if attention_split(shape.heads.max(1), if shape.is_mla() { shape.heads } else { shape.kv_heads }, attn_tp).is_err() {
                continue;
            }
            let eps: Vec<usize> = match &shape.moe {
                Some(_) => divisors(group),
                None => vec![1],
            };
            for ep in eps {
                // with expert parallelism over fewer ranks than the group the rest of the group splits every expert; both are allowed
                out.push(Layout { world, pp, group, attn_tp, ep });
            }
        }
    }
    out
}

pub fn plan_layout(shape: &ModelShape, gpu: &GpuSpec, w: &Workload, l: &Layout) -> Plan {
    let memory = rank_memory(shape, gpu, w, l);
    let needed_tokens = (w.ctx * w.seqs) as f64;
    let mut notes = Vec::new();
    let fits = memory.kv_budget > 0.0 && memory.kv_tokens >= w.ctx as f64;
    if fits && memory.kv_tokens < needed_tokens {
        notes.push(format!(
            "KV cache holds {:.0}k tokens, {} sequences of {} tokens would need {:.0}k (fewer sequences fit at full context)",
            memory.kv_tokens / 1e3,
            w.seqs,
            w.ctx,
            needed_tokens / 1e3
        ));
    }
    if !fits {
        notes.push(format!(
            "does not fit: weights {:.1} GiB + buffers {:.1} GiB of {:.1} GiB usable leave {:.1} GiB for the KV cache",
            memory.weights / GIB,
            (memory.activations + memory.workspace) / GIB,
            memory.usable / GIB,
            memory.kv_budget / GIB
        ));
    }
    if let Some(m) = &shape.moe {
        if !m.experts.is_multiple_of(l.ep) {
            notes.push(format!("{} experts do not divide {} ways: the first ranks hold one expert more", m.experts, l.ep));
        }
    }
    if shape.is_mla() && l.attn_tp > 1 {
        notes.push(format!("the MLA latent cache is replicated on all {} attention ranks: KV capacity does not grow with the group", l.attn_tp));
    }
    if !shape.is_mla() && l.attn_tp > shape.kv_heads {
        notes.push(format!("{} KV heads on {} attention ranks: each KV head is replicated on {} ranks", shape.kv_heads, l.attn_tp, l.attn_tp / shape.kv_heads));
    }
    if l.pp > 1 {
        notes.push("pipeline stages take turns for a single sequence: latency is the sum of the stages, throughput needs many concurrent requests".into());
    }
    let (runnable, why) = match runnable_today(shape, l) {
        Ok(()) => (true, None),
        Err(e) => (false, Some(e)),
    };
    if let Some(why) = why {
        notes.push(format!("not runnable in this build: {why}"));
    }
    let step = estimate_step(shape, gpu, w, l);
    Plan { layout: l.clone(), memory, fits, step_ms: step * 1e3, tokens_per_s: w.batch as f64 / step, runnable_today: runnable, notes }
}

/// Ranked plans (best first): layouts that fit come before those that do not, then by estimated step time.
pub fn plan_node(shape: &ModelShape, gpu: &GpuSpec, w: &Workload, world: usize) -> Vec<Plan> {
    let mut plans: Vec<Plan> = candidate_layouts(shape, world).iter().map(|l| plan_layout(shape, gpu, w, l)).collect();
    // estimates within a nanosecond are ties: prefer fewer pipeline stages, then more expert parallelism, then more attention data parallelism
    let key = |p: &Plan| (p.step_ms * 1e6).round() as i64;
    plans.sort_by(|a, b| {
        b.fits
            .cmp(&a.fits)
            .then(key(a).cmp(&key(b)))
            .then(a.layout.pp.cmp(&b.layout.pp))
            .then(b.layout.ep.cmp(&a.layout.ep))
            .then(b.layout.attn_dp().cmp(&a.layout.attn_dp()))
    });
    plans
}

/// The best plan overall (the ideal) and the best one this build can run.
pub fn best_plans(shape: &ModelShape, gpu: &GpuSpec, w: &Workload, world: usize) -> (Option<Plan>, Option<Plan>) {
    let plans = plan_node(shape, gpu, w, world);
    let ideal = plans.iter().find(|p| p.fits).cloned();
    let today = plans.iter().find(|p| p.fits && p.runnable_today).cloned();
    (ideal, today)
}

/// `--tensor-parallel auto` once the model is known: tensor-parallel size, pipeline stages and whether the experts are distributed.
/// Falls back to `None` (the caller keeps its device-based choice) when no layout fits.
pub fn auto_layout(shape: &ModelShape, gpu: &GpuSpec, w: &Workload, world: usize) -> Option<(usize, usize, bool)> {
    let (_, today) = best_plans(shape, gpu, w, world);
    today.map(|p| (p.layout.group, p.layout.pp, p.layout.ep > 1))
}

/// `--tensor-parallel auto` for a loaded model on the listed identical GPUs: chooses tensor-parallel size, pipeline stages and expert
/// parallelism from the layouts this build can run. Returns the text for the log; leaves `par` alone when nothing fits (the caller then
/// reports the memory error as usual) or the devices are not identical.
pub fn apply_auto(
    par: &mut super::ParallelConfig,
    cfg: &crate::config::ModelConfig,
    weight_bytes: u64,
    list: &[(String, usize)],
    utilization: f64,
    ctx: usize,
) -> Result<Option<String>, String> {
    if list.len() < 2 || !list.iter().all(|d| d.0 == "cuda" || d.0 == "hip") {
        return Ok(None);
    }
    let devices: Vec<crate::devices::PlanningDevice> =
        list.iter().map(|(b, i)| crate::devices::planning_device(b, *i, utilization)).collect::<Result<_, _>>()?;
    let first = &devices[0];
    if devices.iter().any(|d| d.total_bytes.abs_diff(first.total_bytes) > first.total_bytes / 50) {
        return Ok(None); // different cards: the layer split handles those
    }
    let gpu = gpu_from_label(&first.label, &first.backend, first.total_bytes, first.speed_gbs);
    let shape = ModelShape::from_config(cfg);
    let params = shape.params().total();
    if params <= 0.0 {
        return Ok(None);
    }
    let w = Workload { ctx, utilization, bpp_override: Some(weight_bytes as f64 / params), ..Workload::default() };
    let Some((tp, pp, ep)) = auto_layout(&shape, &gpu, &w, list.len()) else {
        return Ok(None);
    };
    par.tp = tp;
    par.pp = pp;
    par.expert_parallel = ep && tp > 1;
    Ok(Some(format!(
        "--tensor-parallel auto: {} x {} -> tensor-parallel {tp} x pipeline-parallel {pp}{} (planner estimate, see `fllm-rs plan-cluster`)",
        list.len(),
        gpu.name,
        if par.expert_parallel { " with expert parallelism" } else { "" }
    )))
}

fn gib(bytes: f64) -> String {
    format!("{:.1} GiB", bytes / GIB)
}

pub fn render(shape: &ModelShape, gpu: &GpuSpec, w: &Workload, world: usize, top: usize) -> String {
    use std::fmt::Write;
    let p = shape.params();
    let mut s = String::new();
    let _ = writeln!(
        s,
        "{} on {}x {} ({}, {}, {} GiB, {:.0} GB/s HBM, {:.0} GB/s link): {:.0}B parameters ({:.0}B active), weights {} = {}",
        shape.name,
        world,
        gpu.name,
        gpu.arch,
        gpu.fabric.name(),
        gpu.mem_gib,
        gpu.hbm_gbps,
        gpu.link_gbps,
        p.total() / 1e9,
        shape.active_params() / 1e9,
        match w.bpp_override {
            Some(b) => format!("{} at {b:.2} bytes per parameter", w.weights.name()),
            None => w.weights.name().to_owned(),
        },
        gib(p.total() * w.bpp())
    );
    let _ = writeln!(s, "workload: {} sequences of {} tokens, decode batch {}, KV {} bytes/value, {:.0}% of memory", w.seqs, w.ctx, w.batch, w.kv_bytes, w.utilization * 100.0);
    let (ideal, today) = best_plans(shape, gpu, w, world);
    let line = |label: &str, plan: &Option<Plan>, s: &mut String| match plan {
        None => {
            let _ = writeln!(s, "{label}: nothing fits ({}x {} cannot hold this model with this workload)", world, gpu.name);
        }
        Some(p) => {
            let _ = writeln!(
                s,
                "{label}: {} | per rank weights {} + buffers {} | KV {} ({:.0}k tokens in the node) | est. decode step {:.2} ms ({:.0} tok/s at batch {}) [roofline estimate, not measured]",
                p.layout.describe(),
                gib(p.memory.weights),
                gib(p.memory.activations + p.memory.workspace),
                gib(p.memory.kv_budget),
                p.memory.kv_tokens / 1e3,
                p.step_ms,
                p.tokens_per_s,
                w.batch
            );
            for n in &p.notes {
                let _ = writeln!(s, "    - {n}");
            }
        }
    };
    line("ideal layout   ", &ideal, &mut s);
    line("runs in fLLM  ", &today, &mut s);
    if let (Some(i), Some(t)) = (&ideal, &today) {
        if i.layout == t.layout {
            let _ = writeln!(s, "(the ideal layout can run in this build)");
        }
    }
    if let Some(t) = &today {
        let mut flags = format!("--tensor-parallel {} --pipeline-parallel-size {}", t.layout.group, t.layout.pp);
        if t.layout.ep > 1 {
            flags.push_str(" --expert-parallel");
        }
        let _ = writeln!(s, "flags: {flags}");
    }
    let _ = writeln!(s, "ranking (first {top}):");
    for (i, p) in plan_node(shape, gpu, w, world).iter().take(top).enumerate() {
        let _ = writeln!(
            s,
            "  {:>2}. {:<52} {} weights {:>9}  KV {:>10}  step {:>7.2} ms  {}",
            i + 1,
            p.layout.describe(),
            if p.fits { "fits   " } else { "no fit " },
            gib(p.memory.weights),
            gib(p.memory.kv_budget),
            p.step_ms,
            if p.runnable_today { "runs today" } else { "planned" }
        );
    }
    s
}

pub const HELP: &str = "usage: fllm-rs plan-cluster --model NAME|PATH [--gpu mi355x] [--gpus 8] [--weights fp8|fp4|bf16|q4k|q8] [--ctx 32768] [--seqs 16]
                          [--batch 64] [--kv-bytes 2] [--max-num-tokens 8192] [--gpu-memory-utilization 0.9] [--top 8] [--native-fp8]
Plans how to spread a model over the GPUs of one node: tensor / expert / pipeline / attention-data parallelism, the memory every rank needs and a
roofline estimate of the decode step (an estimate to rank layouts, not a measurement). Prints the ideal layout and the best layout this build
can run, with the flags for it. MODEL is a preset (deepseek-v3, kimi-k2, qwen3-235b, llama4-maverick, gpt-oss-120b, llama3-70b, llama3-405b,
moe-1t) or a GGUF file (its configuration and the real tensor sizes are used). GPUs: mi355x, mi325x, mi300x, b300, b200, h200, h100, a100, l40s.";

pub fn run(args: &[String]) -> Result<(), String> {
    let mut model = None;
    let mut native_fp8 = false;
    let mut gpu = "mi355x".to_owned();
    let mut gpus = 8usize;
    let mut w = Workload::default();
    let mut top = 8usize;
    let mut i = 0;
    let value = |i: &mut usize, name: &str| -> Result<String, String> {
        *i += 1;
        args.get(*i).cloned().ok_or_else(|| format!("{name} needs a value"))
    };
    while i < args.len() {
        let flag = args[i].clone();
        match flag.as_str() {
            "--model" => model = Some(value(&mut i, &flag)?),
            "--gpu" => gpu = value(&mut i, &flag)?,
            "--gpus" => gpus = value(&mut i, &flag)?.parse().map_err(|_| "--gpus needs a number")?,
            "--weights" => w.weights = WeightFormat::parse(&value(&mut i, &flag)?).ok_or("--weights: bf16, fp8, fp4, q4k or q8")?,
            "--ctx" => w.ctx = value(&mut i, &flag)?.parse().map_err(|_| "--ctx needs a number")?,
            "--seqs" => w.seqs = value(&mut i, &flag)?.parse().map_err(|_| "--seqs needs a number")?,
            "--batch" => w.batch = value(&mut i, &flag)?.parse().map_err(|_| "--batch needs a number")?,
            "--kv-bytes" => w.kv_bytes = value(&mut i, &flag)?.parse().map_err(|_| "--kv-bytes needs a number (2 = fp16, 1 = fp8)")?,
            "--max-num-tokens" => w.max_num_tokens = value(&mut i, &flag)?.parse().map_err(|_| "--max-num-tokens needs a number")?,
            "--gpu-memory-utilization" => w.utilization = value(&mut i, &flag)?.parse().map_err(|_| "--gpu-memory-utilization needs a number")?,
            "--top" => top = value(&mut i, &flag)?.parse().map_err(|_| "--top needs a number")?,
            "--native-fp8" => native_fp8 = true,
            "--help" | "-h" => {
                println!("{HELP}");
                return Ok(());
            }
            other => return Err(format!("unknown option {other}\n{HELP}")),
        }
        i += 1;
    }
    let spec = model.ok_or_else(|| format!("--model is required\n{HELP}"))?;
    let gpu = gpu_preset(&gpu).ok_or_else(|| format!("unknown GPU '{gpu}' (known: {})", gpu_presets().iter().map(|g| g.name).collect::<Vec<_>>().join(", ")))?;
    let shape = if let Some(shape) = ModelShape::preset(&spec) {
        shape
    } else if std::path::Path::new(&spec).is_file() {
        let mapping = fllm_gguf_probe::MappedGguf::open(std::path::Path::new(&spec)).map_err(|e| format!("{spec}: {e}"))?;
        let cfg = crate::config::ModelConfig::from_gguf(&mapping)?;
        ModelShape::from_config(&cfg)
    } else {
        return Err(format!("'{spec}' is neither a preset ({}) nor a file", ModelShape::preset_names().join(", ")));
    };
    if gpu.fabric == Fabric::Xgmi && w.weights == WeightFormat::Fp8 && !native_fp8 {
        // the HIP backend of this build expands FP8 weights to F16 on load
        w.bpp_override = Some(WeightFormat::Bf16.bytes_per_param());
        eprintln!("note: this build expands FP8 weights to F16 on AMD GPUs, so the weights are planned at 2 bytes per parameter (--native-fp8 plans for a native FP8 kernel instead)");
    }
    if !gpu.fp8 && w.weights == WeightFormat::Fp8 || !gpu.fp4 && w.weights == WeightFormat::Fp4 {
        eprintln!("note: {} has no native {} matrix instructions; the weights would be converted on the fly", gpu.name, w.weights.name());
    }
    print!("{}", render(&shape, &gpu, &w, gpus, top));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(model: &str, gpu: &str, weights: WeightFormat, world: usize) -> (ModelShape, Vec<Plan>) {
        let shape = ModelShape::preset(model).unwrap();
        let w = Workload { weights, ..Workload::default() };
        let plans = plan_node(&shape, &gpu_preset(gpu).unwrap(), &w, world);
        (shape, plans)
    }

    #[test]
    fn parameter_counts_match_the_published_sizes() {
        let close = |shape: &str, expect_b: f64, tol: f64| {
            let b = ModelShape::preset(shape).unwrap().params().total() / 1e9;
            assert!((b - expect_b).abs() / expect_b < tol, "{shape}: {b:.1}B, expected about {expect_b}B");
        };
        close("deepseek-v3", 671.0, 0.03);
        close("kimi-k2", 1030.0, 0.05);
        close("qwen3-235b", 235.0, 0.03);
        close("llama3-70b", 70.6, 0.03);
        close("llama3-405b", 405.0, 0.03);
        close("gpt-oss-120b", 117.0, 0.06);
        close("llama4-maverick", 400.0, 0.06);
        let active = ModelShape::preset("deepseek-v3").unwrap().active_params() / 1e9;
        assert!((active - 37.0).abs() < 3.0, "DeepSeek-V3 activates about 37B, got {active:.1}B");
    }

    #[test]
    fn deepseek_v3_fp8_fits_eight_mi355x_with_attention_dp_and_expert_parallelism() {
        let (_, plans) = plan("deepseek-v3", "mi355x", WeightFormat::Fp8, 8);
        let best = &plans[0];
        assert!(best.fits, "{best:?}");
        assert_eq!(best.layout.pp, 1, "a single node should not need pipeline stages: {}", best.layout.describe());
        assert_eq!(best.layout.ep, 8);
        assert_eq!(best.layout.attn_tp, 1, "MLA wants attention data parallelism");
        assert!(!best.runnable_today);
        // ~660 GiB of FP8 weights over 8 ranks is ~83 GiB each
        assert!((70.0..100.0).contains(&(best.memory.weights / GIB)), "{}", best.memory.weights / GIB);
        // the layout this build can run is the pipeline
        let shape = ModelShape::preset("deepseek-v3").unwrap();
        let (ideal, today) = best_plans(&shape, &gpu_preset("mi355x").unwrap(), &Workload::default(), 8);
        assert_eq!(ideal.unwrap().layout.attn_dp(), 8);
        let today = today.unwrap();
        assert_eq!((today.layout.group, today.layout.pp), (1, 8));
    }

    #[test]
    fn deepseek_v3_bf16_does_not_fit_eight_h100_but_fp8_does_not_either() {
        let (_, bf16) = plan("deepseek-v3", "h100", WeightFormat::Bf16, 8);
        assert!(bf16.iter().all(|p| !p.fits));
        let (_, fp8) = plan("deepseek-v3", "h100", WeightFormat::Fp8, 8);
        // 671 GB FP8 over 8 x 79 GiB (632 GiB, 90%) leaves nothing: the planner must say so rather than invent room
        assert!(fp8.iter().all(|p| !p.fits), "{:?}", fp8[0]);
        let (_, fp4) = plan("deepseek-v3", "h100", WeightFormat::Fp4, 8);
        assert!(fp4.iter().any(|p| p.fits));
    }

    #[test]
    fn deepseek_v3_fp4_on_eight_b200() {
        let (_, plans) = plan("deepseek-v3", "b200", WeightFormat::Fp4, 8);
        assert!(plans[0].fits);
        assert_eq!(plans[0].layout.ep, 8);
        // FP8 on 8 x 179 GiB: weights ~660 GiB of 1432 GiB, fits
        let (_, fp8) = plan("deepseek-v3", "b200", WeightFormat::Fp8, 8);
        assert!(fp8[0].fits);
    }

    #[test]
    fn a_one_trillion_parameter_moe_needs_fp4_or_the_big_cards() {
        let (shape, plans) = plan("moe-1t", "mi355x", WeightFormat::Fp8, 8);
        assert!(shape.params().total() > 0.9e12);
        assert!(plans[0].fits, "1T FP8 = ~1 TB over 8 x 288 GiB must fit");
        assert_eq!(plans[0].layout.pp, 1);
        let (_, h100) = plan("moe-1t", "h100", WeightFormat::Fp8, 8);
        assert!(h100.iter().all(|p| !p.fits));
        let (_, b200_fp4) = plan("moe-1t", "b200", WeightFormat::Fp4, 8);
        assert!(b200_fp4[0].fits);
        // GQA attention: the layout runs in this build (attention TP + expert parallelism)
        let (ideal, today) = best_plans(&shape, &gpu_preset("mi355x").unwrap(), &Workload::default(), 8);
        let (ideal, today) = (ideal.unwrap(), today.unwrap());
        assert_eq!(today.layout.ep, 8);
        assert!(today.runnable_today && ideal.step_ms <= today.step_ms + 1e-9);
    }

    #[test]
    fn dense_models_use_tensor_parallelism_over_the_node() {
        let (_, plans) = plan("llama3-70b", "mi355x", WeightFormat::Bf16, 8);
        let best = &plans[0];
        assert!(best.fits);
        assert_eq!((best.layout.group, best.layout.pp, best.layout.ep), (8, 1, 1));
        assert!(best.runnable_today);
        // 405B in bf16 needs 810 GB: fits on 8 x 288 GiB with TP8
        let (_, big) = plan("llama3-405b", "mi355x", WeightFormat::Bf16, 8);
        assert!(big[0].fits);
        assert_eq!(big[0].layout.group, 8);
        // ... but not on 8 x H100 (632 GiB): the planner reports it
        let (_, h100) = plan("llama3-405b", "h100", WeightFormat::Bf16, 8);
        assert!(h100.iter().all(|p| !p.fits));
        let (_, h100_fp8) = plan("llama3-405b", "h100", WeightFormat::Fp8, 8);
        assert!(h100_fp8[0].fits);
    }

    #[test]
    fn gqa_moe_models_use_tp_attention_with_expert_parallelism() {
        for model in ["qwen3-235b", "llama4-maverick", "gpt-oss-120b"] {
            let (_, plans) = plan(model, "b200", WeightFormat::Fp8, 8);
            let best = plans.iter().find(|p| p.fits).unwrap_or_else(|| panic!("{model} must fit 8 x B200"));
            assert_eq!(best.layout.pp, 1, "{model}: {}", best.layout.describe());
            assert!(best.layout.ep > 1, "{model}: {}", best.layout.describe());
        }
    }

    #[test]
    fn kv_heads_fewer_than_ranks_are_replicated_and_noted() {
        let (_, plans) = plan("qwen3-235b", "mi355x", WeightFormat::Fp8, 8);
        let tp8 = plans.iter().find(|p| p.layout.group == 8 && p.layout.pp == 1 && p.layout.attn_tp == 8 && p.layout.ep == 8).unwrap();
        assert!(tp8.notes.iter().any(|n| n.contains("replicated on 2 ranks")), "{:?}", tp8.notes);
    }

    #[test]
    fn candidates_respect_the_head_counts() {
        let shape = ModelShape::preset("qwen3-235b").unwrap(); // 64 heads, 4 KV heads
        for l in candidate_layouts(&shape, 8) {
            assert!(64 % l.attn_tp == 0 && (4 % l.attn_tp == 0 || l.attn_tp % 4 == 0), "{}", l.describe());
            assert_eq!(l.pp * l.group, 8);
        }
        // 7 GPUs: heads do not divide by 7, only pp = 7 or tp = 1 remain
        let seven = candidate_layouts(&shape, 7);
        assert!(seven.iter().all(|l| l.attn_tp == 1));
    }

    #[test]
    fn plans_exist_for_every_gpu_count_and_preset() {
        for model in ModelShape::preset_names() {
            let shape = ModelShape::preset(model).unwrap();
            for gpu in gpu_presets() {
                for world in [1usize, 2, 4, 8, 16] {
                    for weights in [WeightFormat::Bf16, WeightFormat::Fp8, WeightFormat::Fp4] {
                        let w = Workload { weights, ..Workload::default() };
                        let plans = plan_node(&shape, &gpu, &w, world);
                        assert!(!plans.is_empty(), "{model} {} x{world}", gpu.name);
                        // fitting layouts come first and are sorted by step time
                        let mut seen_nofit = false;
                        for p in &plans {
                            assert!(!(seen_nofit && p.fits));
                            seen_nofit |= !p.fits;
                            assert!(p.step_ms.is_finite() && p.step_ms > 0.0 && p.memory.weights > 0.0);
                        }
                        // anything this build can run has attention TP over the whole group and no attention DP
                        if let Some(today) = best_plans(&shape, &gpu, &w, world).1 {
                            assert!(runnable_today(&shape, &today.layout).is_ok());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn auto_layout_picks_tensor_parallel_for_dense_and_pipeline_for_mla_today() {
        let gpu = gpu_preset("mi355x").unwrap();
        let w = Workload { weights: WeightFormat::Bf16, ..Workload::default() };
        let dense = auto_layout(&ModelShape::preset("llama3-70b").unwrap(), &gpu, &w, 8).unwrap();
        assert_eq!(dense, (8, 1, false));
        let w8 = Workload { weights: WeightFormat::Fp8, ..Workload::default() };
        let mla = auto_layout(&ModelShape::preset("deepseek-v3").unwrap(), &gpu, &w8, 8).unwrap();
        assert_eq!(mla, (1, 8, false));
        let moe = auto_layout(&ModelShape::preset("qwen3-235b").unwrap(), &gpu, &w8, 8).unwrap();
        assert_eq!((moe.0, moe.1, moe.2), (8, 1, true));
        // nothing fits: no answer
        let small = gpu_preset("rtx2000ada").unwrap();
        assert!(auto_layout(&ModelShape::preset("deepseek-v3").unwrap(), &small, &w8, 8).is_none());
    }

    #[test]
    fn more_batch_reads_more_experts_so_ep_beats_replication() {
        let shape = ModelShape::preset("qwen3-235b").unwrap();
        let gpu = gpu_preset("b200").unwrap();
        let w = Workload { weights: WeightFormat::Fp8, ..Workload::default() };
        let ep8 = estimate_step(&shape, &gpu, &w, &Layout { world: 8, pp: 1, group: 8, attn_tp: 8, ep: 8 });
        let pp8 = estimate_step(&shape, &gpu, &w, &Layout { world: 8, pp: 8, group: 1, attn_tp: 1, ep: 1 });
        assert!(ep8 < pp8, "tp8+ep8 {ep8} vs pp8 {pp8}");
    }

    #[test]
    fn the_rendered_plan_names_the_layout_and_the_flags() {
        let shape = ModelShape::preset("qwen3-235b").unwrap();
        let text = render(&shape, &gpu_preset("mi355x").unwrap(), &Workload::default(), 8, 5);
        assert!(text.contains("ideal layout"), "{text}");
        assert!(text.contains("--tensor-parallel 8 --pipeline-parallel-size 1 --expert-parallel"), "{text}");
        assert!(text.contains("not measured"));
    }

    #[test]
    fn gpu_names_are_found_loosely() {
        assert_eq!(gpu_preset("MI-355X").unwrap().arch, "gfx950");
        assert_eq!(gpu_preset("B200").unwrap().mem_gib, 179.0);
        assert!(gpu_preset("tpu").is_none());
    }
}
