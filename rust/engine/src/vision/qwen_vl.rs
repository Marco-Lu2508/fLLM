//! Qwen2-VL, Qwen2.5-VL and Qwen3-VL vision tower: dynamic-resolution patches, a ViT with 2D rotary
//! embeddings (window attention in Qwen2.5-VL; learned interpolated position embeddings and
//! DeepStack features in Qwen3-VL) and the patch merger that maps `merge x merge` neighbouring
//! patches to one language-model embedding.
//!
//! The weights are read from the `v.hf.*` tensors the converter copies out of the checkpoint (see
//! `fllm_checkpoint::vision_plan`) in their Hugging Face layout.
use super::mrope::{qwen_positions, MediaSpan, MropePositions};
use super::preprocess::{self, MediaLimits};
use super::{EncodedMedia, MediaKind, MultimodalEncoder};
use crate::compute::{self, LayerNorm, Linear, Tensor};
use fllm_gguf_probe::MappedGguf;
use rayon::prelude::*;
use serde_json::Value;

const PREFIX: &str = "v.hf.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Act {
    QuickGelu,
    Gelu,
    GeluTanh,
    Silu,
    Relu,
}

impl Act {
    fn parse(name: &str) -> Result<Self, String> {
        Ok(match name {
            "quick_gelu" => Self::QuickGelu,
            "gelu" => Self::Gelu,
            "gelu_pytorch_tanh" | "gelu_new" | "gelu_fast" => Self::GeluTanh,
            "silu" | "swish" => Self::Silu,
            "relu" => Self::Relu,
            other => return Err(format!("unsupported vision activation {other:?}")),
        })
    }
    fn apply(self, x: f32) -> f32 {
        match self {
            Self::QuickGelu => x / (1.0 + (-1.702 * x).exp()),
            Self::Gelu => compute::gelu(x),
            Self::GeluTanh => {
                0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
            }
            Self::Silu => x / (1.0 + (-x).exp()),
            Self::Relu => x.max(0.0),
        }
    }
}

/// Geometry and preprocessing parameters, from `fllm.vision.config`.
#[derive(Clone, Debug)]
pub struct QwenVlConfig {
    pub dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub mlp_hidden: usize,
    pub patch: usize,
    pub temporal: usize,
    pub merge: usize,
    pub out_dim: usize,
    /// Qwen2.5-VL: RMSNorm, gated MLP, window attention.
    pub v25: bool,
    /// Qwen3-VL: learned position embeddings, DeepStack, per-frame video with timestamps.
    pub v3: bool,
    /// Side of the square table of learned position embeddings (Qwen3-VL).
    pub pos_side: usize,
    /// Blocks whose output feeds a DeepStack merger (Qwen3-VL).
    pub deepstack: Vec<usize>,
    pub window_size: usize,
    pub fullatt: Vec<usize>,
    pub theta: f32,
    act: Act,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub min_pixels: usize,
    pub max_pixels: usize,
    /// Per-frame pixel bounds for video (the video processor's own, else the image bounds).
    pub video_min_pixels: usize,
    pub video_max_pixels: usize,
    pub image_token_id: i64,
    pub video_token_id: Option<i64>,
    pub tokens_per_second: f32,
}

fn triple(v: Option<&Value>, default: [f32; 3]) -> Result<[f32; 3], String> {
    let Some(v) = v else { return Ok(default) };
    let a = v.as_array().filter(|a| a.len() == 3).ok_or("invalid image mean/std")?;
    let mut out = [0.0; 3];
    for (o, x) in out.iter_mut().zip(a) {
        *o = x.as_f64().ok_or("invalid image mean/std")? as f32;
    }
    Ok(out)
}

impl QwenVlConfig {
    pub fn from_json(family: &str, config: &Value) -> Result<Self, String> {
        let (v25, v3) = match family {
            "qwen2_vl" => (false, false),
            "qwen2_5_vl" => (true, false),
            "qwen3_vl" => (false, true),
            other => return Err(format!("not a Qwen-VL family: {other}")),
        };
        let vision = config.get("vision").ok_or("fllm.vision.config has no vision section")?;
        let pre = config.get("preprocessor").unwrap_or(&Value::Null);
        let tokens = config.get("tokens").unwrap_or(&Value::Null);
        let uint = |key: &str| vision.get(key).and_then(Value::as_u64).map(|n| n as usize);
        let need = |key: &str| uint(key).filter(|&n| n > 0).ok_or_else(|| format!("vision_config.{key} missing"));
        let (dim, out_dim, mlp_hidden) = if v25 || v3 {
            (need("hidden_size")?, need("out_hidden_size")?, need("intermediate_size")?)
        } else {
            let dim = need("embed_dim")?;
            let ratio = vision.get("mlp_ratio").and_then(Value::as_f64).unwrap_or(4.0);
            (dim, need("hidden_size")?, (dim as f64 * ratio) as usize)
        };
        let heads = need("num_heads")?;
        let patch = uint("patch_size").unwrap_or(14);
        let temporal = uint("temporal_patch_size").unwrap_or(2);
        let merge = uint("spatial_merge_size").unwrap_or(2);
        let depth = need("depth")?;
        if dim % heads != 0 || (dim / heads) % 4 != 0 || depth > 128 || dim > 8192 || merge == 0 || temporal == 0 {
            return Err("unsupported Qwen-VL vision dimensions".into());
        }
        let fullatt = vision
            .get("fullatt_block_indexes")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect())
            .unwrap_or_default();
        let act = Act::parse(
            vision
                .get("hidden_act")
                .and_then(Value::as_str)
                .unwrap_or(if v25 {
                    "silu"
                } else if v3 {
                    "gelu_pytorch_tanh"
                } else {
                    "quick_gelu"
                }),
        )?;
        let positions = if v3 { need("num_position_embeddings")? } else { 0 };
        let pos_side = (positions as f64).sqrt().round() as usize;
        if v3 && pos_side * pos_side != positions {
            return Err("num_position_embeddings must be a square".into());
        }
        let deepstack: Vec<usize> = vision
            .get("deepstack_visual_indexes")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect())
            .unwrap_or_default();
        if deepstack.iter().any(|&i| i >= depth) {
            return Err("deepstack_visual_indexes beyond the depth of the tower".into());
        }
        let pixel_bound = |source: &Value, flat: &str, nested: &str, default: usize| {
            source
                .get(flat)
                .and_then(Value::as_u64)
                .or_else(|| source.get("size").and_then(|s| s.get(nested)).and_then(Value::as_u64))
                .map_or(default, |n| n as usize)
        };
        let video_pre = config.get("video_preprocessor").unwrap_or(&Value::Null);
        let (min_pixels, max_pixels) = (
            pixel_bound(pre, "min_pixels", "shortest_edge", 56 * 56),
            pixel_bound(pre, "max_pixels", "longest_edge", 28 * 28 * 1280),
        );
        Ok(Self {
            dim,
            depth,
            heads,
            mlp_hidden,
            patch,
            temporal,
            merge,
            out_dim,
            v25,
            v3,
            pos_side,
            deepstack,
            window_size: uint("window_size").unwrap_or(112),
            fullatt,
            theta: vision
                .get("rope_parameters")
                .and_then(|p| p.get("rope_theta"))
                .and_then(Value::as_f64)
                .unwrap_or(10000.0) as f32,
            act,
            mean: triple(pre.get("image_mean"), [0.481_454_66, 0.457_827_5, 0.408_210_73])?,
            std: triple(pre.get("image_std"), [0.268_629_54, 0.261_302_6, 0.275_777_1])?,
            min_pixels,
            max_pixels,
            video_min_pixels: pixel_bound(video_pre, "min_pixels", "shortest_edge", min_pixels),
            video_max_pixels: pixel_bound(video_pre, "max_pixels", "longest_edge", max_pixels),
            image_token_id: tokens
                .get("image_token_id")
                .and_then(Value::as_i64)
                .ok_or("config has no image_token_id")?,
            video_token_id: tokens.get("video_token_id").and_then(Value::as_i64),
            tokens_per_second: vision.get("tokens_per_second").and_then(Value::as_f64).unwrap_or(2.0) as f32,
        })
    }

    fn head_dim(&self) -> usize {
        self.dim / self.heads
    }
}

enum VisionNorm {
    Layer(LayerNorm),
    Rms { weight: Vec<f32>, eps: f32 },
}

impl VisionNorm {
    fn load(mapping: &MappedGguf, base: &str, dim: usize, rms: bool) -> Result<Self, String> {
        if rms {
            let weight = Tensor::load(mapping, &format!("{base}.weight"))?;
            if weight.shape != [dim] {
                return Err(format!("invalid norm dimensions: {base}"));
            }
            Ok(Self::Rms { weight: weight.data, eps: 1e-6 })
        } else {
            Ok(Self::Layer(LayerNorm::load(mapping, base, dim, 1e-6)?))
        }
    }
    fn apply(&self, x: &mut [f32]) {
        match self {
            Self::Layer(norm) => norm.apply(x),
            Self::Rms { weight, eps } => {
                let dim = weight.len();
                x.par_chunks_mut(dim).for_each(|row| {
                    let inv = (row.iter().map(|v| v * v).sum::<f32>() / dim as f32 + eps).sqrt().recip();
                    for (v, w) in row.iter_mut().zip(weight) {
                        *v *= inv * w;
                    }
                });
            }
        }
    }
}

enum Mlp {
    Plain { fc1: Linear, fc2: Linear },
    Gated { gate: Linear, up: Linear, down: Linear },
}

struct Block {
    norm1: VisionNorm,
    norm2: VisionNorm,
    qkv: Linear,
    proj: Linear,
    mlp: Mlp,
}

/// Patch merger: a norm over each token (or over each merged group when the weight is `merge^2`
/// times wider), then `merge x merge` consecutive rows are read as one row of the two-layer MLP.
struct Merger {
    norm: VisionNorm,
    fc1: Linear,
    fc2: Linear,
}

impl Merger {
    fn load(mapping: &MappedGguf, base: &str, names: [&str; 3], norm_dim: usize, merged: usize, out: usize, rms: bool) -> Result<Self, String> {
        Ok(Self {
            norm: VisionNorm::load(mapping, &format!("{base}.{}", names[0]), norm_dim, rms)?,
            fc1: Linear::load(mapping, &format!("{base}.{}", names[1]), merged, merged)?,
            fc2: Linear::load(mapping, &format!("{base}.{}", names[2]), out, merged)?,
        })
    }

    fn forward(&self, rows: &[f32]) -> Result<Vec<f32>, String> {
        let mut x = rows.to_vec();
        self.norm.apply(&mut x);
        let mut y = self.fc1.forward(&x)?;
        y.par_iter_mut().for_each(|v| *v = compute::gelu(*v));
        self.fc2.forward(&y)
    }
}

/// Output of the tower for one item.
pub struct TowerOutput {
    pub embeddings: Vec<Vec<f32>>,
    /// Per DeepStack level, one row per embedding.
    pub deepstack: Vec<Vec<Vec<f32>>>,
}

pub struct QwenVlEncoder {
    cfg: QwenVlConfig,
    patch_embed: Linear,
    /// Learned position table (rows x dim) of Qwen3-VL.
    pos_embed: Option<Vec<f32>>,
    blocks: Vec<Block>,
    merger: Merger,
    deepstack: Vec<(usize, Merger)>,
    pool: rayon::ThreadPool,
    family: &'static str,
}

impl QwenVlEncoder {
    /// Builds the tower from the `v.hf.*` tensors of the model GGUF; `config` is the parsed
    /// `fllm.vision.config`.
    pub fn load(mapping: &MappedGguf, family: &str, config: &Value, threads: usize, device: &str) -> Result<Self, String> {
        let cfg = QwenVlConfig::from_json(family, config)?;
        let n = |name: &str| format!("{PREFIX}{name}");
        let dim = cfg.dim;
        let patch_in = 3 * cfg.temporal * cfg.patch * cfg.patch;
        let patch_embed = Linear::load(mapping, &n("patch_embed.proj"), dim, patch_in)?;
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            let b = |name: &str| n(&format!("blocks.{i}.{name}"));
            let mlp = if cfg.v25 {
                Mlp::Gated {
                    gate: Linear::load(mapping, &b("mlp.gate_proj"), cfg.mlp_hidden, dim)?,
                    up: Linear::load(mapping, &b("mlp.up_proj"), cfg.mlp_hidden, dim)?,
                    down: Linear::load(mapping, &b("mlp.down_proj"), dim, cfg.mlp_hidden)?,
                }
            } else {
                let (first, second) = if cfg.v3 { ("mlp.linear_fc1", "mlp.linear_fc2") } else { ("mlp.fc1", "mlp.fc2") };
                Mlp::Plain {
                    fc1: Linear::load(mapping, &b(first), cfg.mlp_hidden, dim)?,
                    fc2: Linear::load(mapping, &b(second), dim, cfg.mlp_hidden)?,
                }
            };
            blocks.push(Block {
                norm1: VisionNorm::load(mapping, &b("norm1"), dim, cfg.v25)?,
                norm2: VisionNorm::load(mapping, &b("norm2"), dim, cfg.v25)?,
                qkv: Linear::load(mapping, &b("attn.qkv"), 3 * dim, dim)?,
                proj: Linear::load(mapping, &b("attn.proj"), dim, dim)?,
                mlp,
            });
        }
        let merged = dim * cfg.merge * cfg.merge;
        let names = if cfg.v3 { ["norm", "linear_fc1", "linear_fc2"] } else { ["ln_q", "mlp.0", "mlp.2"] };
        let merger = Merger::load(mapping, &n("merger"), names, dim, merged, cfg.out_dim, cfg.v25)?;
        let mut deepstack = Vec::new();
        for (k, &block) in cfg.deepstack.iter().enumerate() {
            let base = n(&format!("deepstack_merger_list.{k}"));
            // these mergers normalise each merged group, so their norm is `merge^2` times wider
            deepstack.push((block, Merger::load(mapping, &base, names, merged, merged, cfg.out_dim, false)?));
        }
        let pos_embed = if cfg.v3 {
            let table = Tensor::load(mapping, &n("pos_embed.weight"))?;
            if table.shape != [dim, cfg.pos_side * cfg.pos_side] {
                return Err("invalid vision position embedding table".into());
            }
            Some(table.data)
        } else {
            None
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads.clamp(1, 256))
            .build()
            .map_err(|e| e.to_string())?;
        let family = if cfg.v3 {
            "qwen3_vl"
        } else if cfg.v25 {
            "qwen2_5_vl"
        } else {
            "qwen2_vl"
        };
        let mut encoder = Self { cfg, patch_embed, pos_embed, blocks, merger, deepstack, pool, family };
        encoder.set_device(device)?;
        Ok(encoder)
    }

    pub fn config(&self) -> &QwenVlConfig {
        &self.cfg
    }

    fn set_device(&mut self, device: &str) -> Result<(), String> {
        if matches!(device, "cpu" | "auto") {
            return Ok(());
        }
        #[cfg(feature = "rust-gpu")]
        {
            let gpu = crate::gpu::GpuContext::new(device)?;
            self.patch_embed.enable_gpu(&gpu)?;
            for merger in std::iter::once(&mut self.merger).chain(self.deepstack.iter_mut().map(|d| &mut d.1)) {
                merger.fc1.enable_gpu(&gpu)?;
                merger.fc2.enable_gpu(&gpu)?;
            }
            for b in &mut self.blocks {
                let mlp: Vec<&mut Linear> = match &mut b.mlp {
                    Mlp::Plain { fc1, fc2 } => vec![fc1, fc2],
                    Mlp::Gated { gate, up, down } => vec![gate, up, down],
                };
                for linear in [&mut b.qkv, &mut b.proj].into_iter().chain(mlp) {
                    linear.enable_gpu(&gpu)?;
                }
            }
            Ok(())
        }
        #[cfg(not(feature = "rust-gpu"))]
        Err("vision GPU projections require --features rust-gpu".into())
    }

    /// Rotary angles of the patches of one item, `(rows x head_dim/2)` cosines and sines.
    fn rotary(&self, grid: [usize; 3]) -> (Vec<f32>, Vec<f32>) {
        let [t, gh, gw] = grid;
        let m = self.cfg.merge;
        let half = self.cfg.head_dim() / 2;
        let lanes = half / 2;
        let inverse: Vec<f32> = (0..lanes)
            .map(|i| 1.0 / self.cfg.theta.powf((2 * i) as f32 / half as f32))
            .collect();
        let mut cos = Vec::with_capacity(t * gh * gw * half);
        let mut sin = Vec::with_capacity(t * gh * gw * half);
        for _ in 0..t {
            for bh in 0..gh / m {
                for bw in 0..gw / m {
                    for mh in 0..m {
                        for mw in 0..m {
                            let (h, w) = ((bh * m + mh) as f32, (bw * m + mw) as f32);
                            for j in 0..half {
                                let position = if j < lanes { h } else { w };
                                let phase = position * inverse[j % lanes];
                                cos.push(phase.cos());
                                sin.push(phase.sin());
                            }
                        }
                    }
                }
            }
        }
        (cos, sin)
    }

    /// Learned position embeddings of Qwen3-VL resampled to the patch grid (bilinear, corners
    /// aligned), rows in merge-window order, repeated for every frame.
    fn interpolated_positions(&self, grid: [usize; 3]) -> Vec<f32> {
        let table = self.pos_embed.as_ref().expect("Qwen3-VL tower has a position table");
        let [t, gh, gw] = grid;
        let (m, side, dim) = (self.cfg.merge, self.cfg.pos_side, self.cfg.dim);
        // taps and weights along one axis
        let axis = |size: usize| -> Vec<(usize, usize, f32)> {
            (0..size)
                .map(|i| {
                    let src = i as f32 * (side - 1) as f32 / (size - 1).max(1) as f32;
                    let low = src.floor();
                    let lo = (low as usize).min(side - 1);
                    (lo, (lo + 1).min(side - 1), src - low)
                })
                .collect()
        };
        let (rows, cols) = (axis(gh), axis(gw));
        let mut frame = Vec::with_capacity(gh * gw * dim);
        for bh in 0..gh / m {
            for bw in 0..gw / m {
                for mh in 0..m {
                    for mw in 0..m {
                        let (y0, y1, fy) = rows[bh * m + mh];
                        let (x0, x1, fx) = cols[bw * m + mw];
                        let weights = [(y0, x0, (1.0 - fy) * (1.0 - fx)), (y0, x1, (1.0 - fy) * fx), (y1, x0, fy * (1.0 - fx)), (y1, x1, fy * fx)];
                        for d in 0..dim {
                            let mut v = 0.0;
                            for (y, x, w) in weights {
                                v += table[(y * side + x) * dim + d] * w;
                            }
                            frame.push(v);
                        }
                    }
                }
            }
        }
        let mut out = Vec::with_capacity(t * frame.len());
        for _ in 0..t {
            out.extend_from_slice(&frame);
        }
        out
    }

    /// Runs the tower over flattened patches (`rows x 3*temporal*patch*patch`, merge-window
    /// order) of one image or clip with the given patch grid; one embedding per merged group.
    pub fn encode_patches(&self, patches: &[f32], grid: [usize; 3]) -> Result<TowerOutput, String> {
        let cfg = &self.cfg;
        let [t, gh, gw] = grid;
        let m = cfg.merge;
        let rows = t * gh * gw;
        let patch_in = 3 * cfg.temporal * cfg.patch * cfg.patch;
        if rows == 0 || gh % m != 0 || gw % m != 0 || patches.len() != rows * patch_in || patches.iter().any(|v| !v.is_finite()) {
            return Err("invalid vision patches".into());
        }
        self.pool.install(|| {
            let dim = cfg.dim;
            let half = cfg.head_dim() / 2;
            let unit = m * m;
            let mut x = self.patch_embed.forward(patches)?;
            if cfg.v3 {
                compute::add(&mut x, &self.interpolated_positions(grid));
            }
            let (mut cos, mut sin) = self.rotary(grid);
            let frames: Vec<(usize, usize)> = (0..t).map(|i| (i * gh * gw, gh * gw)).collect();
            let (order, window_segments) = if cfg.v25 {
                let (order, segments) = window_layout(cfg, grid);
                let permute = |data: &[f32], width: usize| -> Vec<f32> {
                    let mut out = Vec::with_capacity(data.len());
                    for &u in &order {
                        out.extend_from_slice(&data[u * unit * width..(u + 1) * unit * width]);
                    }
                    out
                };
                x = permute(&x, dim);
                cos = permute(&cos, half);
                sin = permute(&sin, half);
                (order, segments)
            } else {
                (Vec::new(), Vec::new())
            };
            let mut features: Vec<Vec<Vec<f32>>> = Vec::new();
            for (i, block) in self.blocks.iter().enumerate() {
                let segments = if cfg.v25 && !cfg.fullatt.contains(&i) { &window_segments } else { &frames };
                let mut normed = x.clone();
                block.norm1.apply(&mut normed);
                let qkv = block.qkv.forward(&normed)?;
                let (mut q, mut k, mut v) = (vec![0.0; rows * dim], vec![0.0; rows * dim], vec![0.0; rows * dim]);
                for r in 0..rows {
                    q[r * dim..(r + 1) * dim].copy_from_slice(&qkv[r * 3 * dim..r * 3 * dim + dim]);
                    k[r * dim..(r + 1) * dim].copy_from_slice(&qkv[r * 3 * dim + dim..r * 3 * dim + 2 * dim]);
                    v[r * dim..(r + 1) * dim].copy_from_slice(&qkv[r * 3 * dim + 2 * dim..(r + 1) * 3 * dim]);
                }
                rotate(&mut q, cfg.heads, cfg.head_dim(), &cos, &sin);
                rotate(&mut k, cfg.heads, cfg.head_dim(), &cos, &sin);
                let attended = segment_attention(&q, &k, &v, cfg.heads, cfg.head_dim(), segments);
                compute::add(&mut x, &block.proj.forward(&attended)?);
                normed.clone_from(&x);
                block.norm2.apply(&mut normed);
                let hidden = match &block.mlp {
                    Mlp::Plain { fc1, fc2 } => {
                        let mut h = fc1.forward(&normed)?;
                        h.par_iter_mut().for_each(|v| *v = cfg.act.apply(*v));
                        fc2.forward(&h)?
                    }
                    Mlp::Gated { gate, up, down } => {
                        let mut g = gate.forward(&normed)?;
                        let u = up.forward(&normed)?;
                        g.par_iter_mut().zip(u.par_iter()).for_each(|(g, u)| *g = cfg.act.apply(*g) * u);
                        down.forward(&g)?
                    }
                };
                compute::add(&mut x, &hidden);
                if let Some((_, merger)) = self.deepstack.iter().find(|(block, _)| *block == i) {
                    features.push(to_rows(&merger.forward(&x)?, cfg.out_dim));
                }
            }
            let merged = self.merger.forward(&x)?;
            let out_dim = cfg.out_dim;
            let units = rows / unit;
            let mut embeddings = vec![Vec::new(); units];
            if cfg.v25 {
                for (position, &u) in order.iter().enumerate() {
                    embeddings[u] = merged[position * out_dim..(position + 1) * out_dim].to_vec();
                }
            } else {
                for (u, row) in embeddings.iter_mut().enumerate() {
                    *row = merged[u * out_dim..(u + 1) * out_dim].to_vec();
                }
            }
            if embeddings.iter().chain(features.iter().flatten()).flatten().any(|v| !v.is_finite()) {
                return Err("non-finite vision output".into());
            }
            Ok(TowerOutput { embeddings, deepstack: features })
        })
    }

    /// Pixel budget of one image: the processor's own bounds, further limited by the server.
    fn pixel_bounds(&self, kind: MediaKind) -> (usize, usize) {
        let limit = MediaLimits::get().max_image_pixels;
        let factor = self.cfg.patch * self.cfg.merge;
        let (min, max) = match kind {
            MediaKind::Image => (self.cfg.min_pixels, self.cfg.max_pixels),
            MediaKind::Video => (self.cfg.video_min_pixels, self.cfg.video_max_pixels),
        };
        let max = max.min(limit).max(factor * factor);
        (min.min(max), max)
    }

    fn encode_frames(&self, frames: &[image::RgbImage], kind: MediaKind, fps: Option<f32>) -> Result<EncodedMedia, String> {
        let first = frames.first().ok_or("no frames")?;
        let factor = self.cfg.patch * self.cfg.merge;
        let (min, max) = self.pixel_bounds(kind);
        let (h, w) = preprocess::smart_resize(first.height() as usize, first.width() as usize, factor, min, max)?;
        let resized: Vec<_> = frames.iter().map(|f| preprocess::resize(f, w as u32, h as u32)).collect();
        let (patches, grid) = preprocess::qwen_patches(&resized, self.cfg.patch, self.cfg.temporal, self.cfg.merge, self.cfg.mean, self.cfg.std)?;
        let tower = self.encode_patches(&patches, grid)?;
        let m = self.cfg.merge;
        let token_grid = [grid[0], grid[1] / m, grid[2] / m];
        let mut media = EncodedMedia::new(tower.embeddings, token_grid, kind);
        media.deepstack = tower.deepstack;
        media.fps = fps;
        if self.cfg.v3 && kind == MediaKind::Video {
            // every temporal patch is its own run of tokens, announced by the mean time of its frames
            let rate = fps.filter(|f| *f > 0.0).unwrap_or(2.0);
            let (tps, count) = (self.cfg.temporal, frames.len());
            media.runs = vec![token_grid[1] * token_grid[2]; token_grid[0]];
            media.timestamps = (0..token_grid[0])
                .map(|g| ((g * tps).min(count - 1) + (g * tps + tps - 1).min(count - 1)) as f32 / 2.0 / rate)
                .collect();
        }
        Ok(media)
    }
}

/// Attention windows of Qwen2.5-VL: the order in which merge units are visited and the
/// `(start, len)` token ranges of the windows in that order. Windows at the right and bottom
/// edge are smaller; frames never share a window.
fn window_layout(cfg: &QwenVlConfig, grid: [usize; 3]) -> (Vec<usize>, Vec<(usize, usize)>) {
    let [t, gh, gw] = grid;
    let m = cfg.merge;
    let (lh, lw) = (gh / m, gw / m);
    let size = (cfg.window_size / m / cfg.patch).max(1);
    let unit = m * m;
    let mut order = Vec::with_capacity(t * lh * lw);
    let mut segments = Vec::new();
    let mut at = 0;
    for ti in 0..t {
        for wy in 0..lh.div_ceil(size) {
            for wx in 0..lw.div_ceil(size) {
                let mut count = 0;
                for y in wy * size..((wy + 1) * size).min(lh) {
                    for x in wx * size..((wx + 1) * size).min(lw) {
                        order.push((ti * lh + y) * lw + x);
                        count += 1;
                    }
                }
                if count > 0 {
                    segments.push((at, count * unit));
                    at += count * unit;
                }
            }
        }
    }
    (order, segments)
}

/// `data` (rows of `width` values) as one vector per row.
fn to_rows(data: &[f32], width: usize) -> Vec<Vec<f32>> {
    data.chunks_exact(width).map(<[f32]>::to_vec).collect()
}

/// Rotates every head of `x` (`rows x heads*head_dim`) by the per-row angles with the
/// `rotate_half` pairing `(j, j + head_dim/2)`.
fn rotate(x: &mut [f32], heads: usize, head_dim: usize, cos: &[f32], sin: &[f32]) {
    let half = head_dim / 2;
    x.par_chunks_mut(heads * head_dim).enumerate().for_each(|(r, row)| {
        let (c, s) = (&cos[r * half..(r + 1) * half], &sin[r * half..(r + 1) * half]);
        for head in row.chunks_exact_mut(head_dim) {
            for j in 0..half {
                let (a, b) = (head[j], head[j + half]);
                head[j] = a * c[j] - b * s[j];
                head[j + half] = b * c[j] + a * s[j];
            }
        }
    });
}

/// Bidirectional attention inside each `(start, len)` segment of rows.
fn segment_attention(q: &[f32], k: &[f32], v: &[f32], heads: usize, head_dim: usize, segments: &[(usize, usize)]) -> Vec<f32> {
    let dim = heads * head_dim;
    let rows = q.len() / dim;
    let mut segment_of = vec![(0usize, 0usize); rows];
    for &(start, len) in segments {
        for item in &mut segment_of[start..start + len] {
            *item = (start, len);
        }
    }
    let scale = (head_dim as f32).sqrt().recip();
    let mut out = vec![0.0f32; q.len()];
    out.par_chunks_mut(dim).enumerate().for_each(|(i, row)| {
        let (start, len) = segment_of[i];
        let mut scores = vec![0.0f32; len];
        for h in 0..heads {
            let offset = h * head_dim;
            let query = &q[i * dim + offset..i * dim + offset + head_dim];
            for (j, s) in scores.iter_mut().enumerate() {
                let key = &k[(start + j) * dim + offset..(start + j) * dim + offset + head_dim];
                *s = compute::dot(query, key) * scale;
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for s in &mut scores {
                *s = (*s - max).exp();
                sum += *s;
            }
            let target = &mut row[offset..offset + head_dim];
            for (j, s) in scores.iter().enumerate() {
                let weight = *s / sum;
                let value = &v[(start + j) * dim + offset..(start + j) * dim + offset + head_dim];
                for (t, x) in target.iter_mut().zip(value) {
                    *t += weight * x;
                }
            }
        }
    });
    out
}

impl MultimodalEncoder for QwenVlEncoder {
    fn family(&self) -> &'static str {
        self.family
    }

    fn output_hidden(&self) -> usize {
        self.cfg.out_dim
    }

    fn marker(&self, kind: MediaKind) -> Option<&str> {
        match kind {
            MediaKind::Image => Some("<|image_pad|>"),
            MediaKind::Video => self.cfg.video_token_id.map(|_| "<|video_pad|>"),
        }
    }

    fn expand(&self, media: &EncodedMedia) -> String {
        match media.kind {
            MediaKind::Image => "<|image_pad|>".repeat(media.embeddings.len()),
            // Qwen3-VL separates the frames of a clip by their time stamps, each frame in its own
            // vision start / end pair (the `replace_video_token` rule of the reference processor)
            MediaKind::Video if self.cfg.v3 => {
                let per_frame = media.embeddings.len() / media.timestamps.len().max(1);
                media
                    .timestamps
                    .iter()
                    .map(|t| format!("<{t:.1} seconds><|vision_start|>{}<|vision_end|>", "<|video_pad|>".repeat(per_frame)))
                    .collect()
            }
            MediaKind::Video => "<|video_pad|>".repeat(media.embeddings.len()),
        }
    }

    fn token_id(&self, kind: MediaKind) -> Option<i64> {
        match kind {
            MediaKind::Image => Some(self.cfg.image_token_id),
            MediaKind::Video => self.cfg.video_token_id,
        }
    }

    fn encode_image(&self, bytes: &[u8]) -> Result<EncodedMedia, String> {
        let image = preprocess::decode_image(bytes)?;
        self.encode_frames(&[image], MediaKind::Image, None)
    }

    fn encode_video(&self, frames: &[Vec<u8>], fps: f32) -> Result<EncodedMedia, String> {
        let limits = MediaLimits::get();
        if frames.is_empty() || frames.len() > limits.max_video_frames {
            return Err(format!("a video needs 1 to {} frames", limits.max_video_frames));
        }
        if self.cfg.video_token_id.is_none() {
            return Err("this model has no video token".into());
        }
        let decoded = frames.iter().map(|f| preprocess::decode_image(f)).collect::<Result<Vec<_>, _>>()?;
        self.encode_frames(&decoded, MediaKind::Video, Some(fps))
    }

    fn mrope_layout(&self) -> bool {
        true
    }

    fn positions(&self, total: usize, spans: &[(usize, [usize; 3], &EncodedMedia)]) -> Result<Option<MropePositions>, String> {
        let spans: Vec<MediaSpan> = spans
            .iter()
            .map(|(start, grid, media)| MediaSpan {
                start: *start,
                grid: *grid,
                time_interval: match (media.kind, self.cfg.v25) {
                    (MediaKind::Video, true) => {
                        let fps = media.fps.filter(|f| *f > 0.0).unwrap_or(2.0);
                        self.cfg.temporal as f32 / fps * self.cfg.tokens_per_second
                    }
                    _ => 1.0,
                },
            })
            .collect();
        qwen_positions(total, &spans).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn config(v25: bool) -> Value {
        let vision = if v25 {
            json!({"depth": 3, "hidden_size": 32, "intermediate_size": 48, "num_heads": 2, "out_hidden_size": 64,
                   "patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2, "window_size": 56,
                   "fullatt_block_indexes": [2], "hidden_act": "silu"})
        } else {
            json!({"depth": 3, "embed_dim": 32, "hidden_size": 64, "num_heads": 2, "mlp_ratio": 2,
                   "patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2})
        };
        json!({"vision": vision, "preprocessor": {"min_pixels": 3136, "max_pixels": 1003520},
               "tokens": {"image_token_id": 8, "video_token_id": 9}})
    }

    #[test]
    fn config_maps_both_families() {
        let a = QwenVlConfig::from_json("qwen2_vl", &config(false)).unwrap();
        assert_eq!((a.dim, a.out_dim, a.mlp_hidden, a.v25), (32, 64, 64, false));
        let b = QwenVlConfig::from_json("qwen2_5_vl", &config(true)).unwrap();
        assert_eq!((b.dim, b.out_dim, b.mlp_hidden, b.v25), (32, 64, 48, true));
        assert_eq!(b.fullatt, vec![2]);
        assert_eq!(b.image_token_id, 8);
        assert!(QwenVlConfig::from_json("llava", &config(false)).is_err());
    }

    #[test]
    fn window_order_covers_every_unit_once() {
        let mut cfg = QwenVlConfig::from_json("qwen2_5_vl", &config(true)).unwrap();
        cfg.window_size = 56; // 56 / 2 / 14 = 2 merge units per window side
        let (gh, gw) = (6, 8); // 3 x 4 merge units: windows 2x2, 2x2, 1x2, 1x2 (edge windows are smaller)
        let tower = window_layout(&cfg, [1, gh, gw]);
        let mut seen = tower.0.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..12).collect::<Vec<_>>());
        assert_eq!(tower.1.iter().map(|s| s.1).sum::<usize>(), 48);
        assert_eq!(tower.1.iter().map(|s| s.1 / 4).collect::<Vec<_>>(), vec![4, 4, 2, 2]);
        // first window: units (0,0) (0,1) (1,0) (1,1) of the 3x4 unit grid
        assert_eq!(&tower.0[..4], &[0, 1, 4, 5]);
    }

    #[test]
    fn segment_attention_never_crosses_segments() {
        let (heads, head_dim) = (1, 2);
        let q = vec![0.0f32; 4 * 2];
        let k = vec![0.0f32; 4 * 2];
        let v: Vec<f32> = vec![1., 1., 3., 3., 10., 10., 30., 30.];
        let out = segment_attention(&q, &k, &v, heads, head_dim, &[(0, 2), (2, 2)]);
        // uniform weights inside each segment: the mean of its values
        assert_eq!(out, vec![2., 2., 2., 2., 20., 20., 20., 20.]);
    }
}
