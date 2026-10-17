#![allow(
    clippy::needless_range_loop,
    clippy::manual_div_ceil,
    clippy::chunks_exact_to_as_chunks
)]
//! SigLIP vision tower and projector for PaliGemma and Gemma 3 Vision.
//!
//! Features:
//! * Patch embedding (14x14 or configurable patch size)
//! * Learned position embeddings
//! * ViT transformer layers: LayerNorm, Multi-Head Self-Attention, LayerNorm, MLP (GELU)
//! * Post LayerNorm
//! * Projectors:
//!   - Linear projector for PaliGemma (`multi_modal_projector.linear`)
//!   - 2D Spatial pooling projector for Gemma 3 (2x2 spatial downsampling + linear projection)
//! * Bidirectional image attention for Gemma 3 (image tokens attend bidirectionally within their image span)
//! * Prefix-LM mask for PaliGemma (all prompt prefix tokens attend bidirectionally)

use super::{EncodedMedia, MediaKind, MultimodalEncoder};
use crate::compute::{self, LayerNorm, Linear, Tensor};
use fllm_gguf_probe::MappedGguf;
use serde_json::Value;

const PREFIX: &str = "v.hf.";

#[derive(Clone, Debug)]
pub struct SiglipConfig {
    pub family: String,
    pub hidden: usize,
    pub intermediate: usize,
    pub depth: usize,
    pub heads: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub out_dim: usize,
    pub pooling: bool,
    pub image_token_id: i64,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl SiglipConfig {
    pub fn from_json(family: &str, config: &Value) -> Result<Self, String> {
        let vision = config.get("vision").ok_or("config has no vision section")?;
        let tokens = config.get("tokens").unwrap_or(&Value::Null);

        let uint = |key: &str| vision.get(key).and_then(Value::as_u64).map(|n| n as usize);
        let hidden = uint("hidden_size").or_else(|| uint("embed_dim")).unwrap_or(1152);
        let intermediate = uint("intermediate_size").unwrap_or(4304);
        let depth = uint("num_hidden_layers").or_else(|| uint("depth")).unwrap_or(27);
        let heads = uint("num_attention_heads").or_else(|| uint("num_heads")).unwrap_or(16);
        let image_size = uint("image_size").unwrap_or(224);
        let patch_size = uint("patch_size").unwrap_or(14);
        let out_dim = uint("out_hidden_size")
            .or_else(|| config.get("text_config").and_then(|t| t.get("hidden_size")).and_then(Value::as_u64).map(|n| n as usize))
            .unwrap_or(2048);

        let pooling = family == "gemma3" || vision.get("pooling").and_then(Value::as_bool).unwrap_or(false);
        let image_token_id = tokens
            .get("image_token_id")
            .and_then(Value::as_i64)
            .or_else(|| config.get("image_token_id").and_then(Value::as_i64))
            .unwrap_or(257152);

        Ok(Self {
            family: family.to_owned(),
            hidden,
            intermediate,
            depth,
            heads,
            image_size,
            patch_size,
            out_dim,
            pooling,
            image_token_id,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        })
    }
}

pub struct SiglipLayer {
    pub norm1: LayerNorm,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub out: Linear,
    pub norm2: LayerNorm,
    pub fc1: Linear,
    pub fc2: Linear,
}

pub struct SiglipEncoder {
    pub config: SiglipConfig,
    pub patch: Linear,
    pub position: Tensor,
    pub layers: Vec<SiglipLayer>,
    pub post_norm: LayerNorm,
    pub projector: Linear,
    pub pool: rayon::ThreadPool,
}

fn has_tensor(mapping: &MappedGguf, name: &str) -> bool {
    mapping.index().tensors.iter().any(|t| t.name == name)
}

impl SiglipEncoder {
    pub fn load(
        mapping: &MappedGguf,
        family: &str,
        config: &Value,
        threads: usize,
        _device: &str,
    ) -> Result<Self, String> {
        let cfg = SiglipConfig::from_json(family, config)?;
        let hidden = cfg.hidden;
        let intermediate = cfg.intermediate;
        let patch_dim = 3 * cfg.patch_size * cfg.patch_size;

        let n = |rest: &str| format!("{PREFIX}{rest}");

        // Patch embedding: shape [hidden, patch_dim]
        let patch_base = if has_tensor(mapping, &n("embeddings.patch_embedding.weight")) {
            n("embeddings.patch_embedding")
        } else if has_tensor(mapping, &n("patch_embedding.weight")) {
            n("patch_embedding")
        } else {
            n("embeddings.patch_embedding")
        };
        let patch = Linear::load(mapping, &patch_base, hidden, patch_dim)?;

        // Position embedding: shape [num_patches, hidden]
        let pos_name = if has_tensor(mapping, &n("embeddings.position_embedding.weight")) {
            n("embeddings.position_embedding.weight")
        } else {
            n("position_embedding.weight")
        };
        let position = Tensor::load(mapping, &pos_name)?;

        let mut layers = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            let base = if has_tensor(mapping, &n(&format!("encoder.layers.{i}.layer_norm1.weight"))) {
                n(&format!("encoder.layers.{i}"))
            } else {
                n(&format!("layers.{i}"))
            };
            layers.push(SiglipLayer {
                norm1: LayerNorm::load(mapping, &format!("{base}.layer_norm1"), hidden, 1e-6)?,
                q: Linear::load(mapping, &format!("{base}.self_attn.q_proj"), hidden, hidden)?,
                k: Linear::load(mapping, &format!("{base}.self_attn.k_proj"), hidden, hidden)?,
                v: Linear::load(mapping, &format!("{base}.self_attn.v_proj"), hidden, hidden)?,
                out: Linear::load(mapping, &format!("{base}.self_attn.out_proj"), hidden, hidden)?,
                norm2: LayerNorm::load(mapping, &format!("{base}.layer_norm2"), hidden, 1e-6)?,
                fc1: Linear::load(mapping, &format!("{base}.mlp.fc1"), intermediate, hidden)?,
                fc2: Linear::load(mapping, &format!("{base}.mlp.fc2"), hidden, intermediate)?,
            });
        }

        let post_norm_base = if has_tensor(mapping, &n("post_layernorm.weight")) {
            n("post_layernorm")
        } else {
            n("encoder.post_layernorm")
        };
        let post_norm = LayerNorm::load(mapping, &post_norm_base, hidden, 1e-6)?;

        let proj_base = if has_tensor(mapping, &n("projector.linear.weight")) {
            n("projector.linear")
        } else {
            n("projector")
        };
        let proj_in = if cfg.pooling && has_tensor(mapping, &format!("{proj_base}.weight"))
            && mapping.index().tensors.iter().find(|t| t.name == format!("{proj_base}.weight")).is_some_and(|t| t.dimensions.first().copied() == Some(hidden as u64 * 4))
        {
            hidden * 4
        } else {
            hidden
        };
        let projector = Linear::load(mapping, &proj_base, cfg.out_dim, proj_in)?;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads.clamp(1, 256))
            .build()
            .map_err(|e| e.to_string())?;

        Ok(Self {
            config: cfg,
            patch,
            position,
            layers,
            post_norm,
            projector,
            pool,
        })
    }

    /// Encodes normalized RGB pixels into embeddings.
    pub fn encode_pixels(&self, pixels: &[f32], grid_w: usize, grid_h: usize) -> Result<EncodedMedia, String> {
        let patch_size = self.config.patch_size;
        let hidden = self.config.hidden;
        let num_patches = grid_w * grid_h;
        let patch_dim = 3 * patch_size * patch_size;

        if pixels.len() != grid_w * patch_size * grid_h * patch_size * 3 {
            return Err("invalid pixel buffer size for SigLIP".into());
        }

        self.pool.install(|| {
            let width = grid_w * patch_size;
            let height = grid_h * patch_size;
            let mut patches = Vec::with_capacity(num_patches * patch_dim);

            for gy in 0..grid_h {
                for gx in 0..grid_w {
                    for c in 0..3 {
                        for py in 0..patch_size {
                            for px in 0..patch_size {
                                let y = gy * patch_size + py;
                                let x = gx * patch_size + px;
                                patches.push(pixels[c * width * height + y * width + x]);
                            }
                        }
                    }
                }
            }

            // Patch embedding projection: [num_patches, hidden]
            let mut x = self.patch.forward(&patches)?;

            // Add position embeddings
            if self.position.data.len() >= num_patches * hidden {
                compute::add(&mut x, &self.position.data[..num_patches * hidden]);
            } else {
                return Err("position embedding table smaller than image patches".into());
            }

            // Transformer layers
            for l in &self.layers {
                let mut n = x.clone();
                l.norm1.apply(&mut n);
                let q = l.q.forward(&n)?;
                let k = l.k.forward(&n)?;
                let v = l.v.forward(&n)?;
                let a = compute::attention(&q, &k, &v, hidden, self.config.heads);
                compute::add(&mut x, &l.out.forward(&a)?);

                n = x.clone();
                l.norm2.apply(&mut n);
                let mut fc1 = l.fc1.forward(&n)?;
                for val in &mut fc1 {
                    *val = compute::gelu(*val);
                }
                compute::add(&mut x, &l.fc2.forward(&fc1)?);
            }

            self.post_norm.apply(&mut x);

            // Projector / Pooling
            let (projected, out_grid) = if self.config.pooling {
                let pool_w = grid_w / 2;
                let pool_h = grid_h / 2;
                if pool_w == 0 || pool_h == 0 {
                    return Err("image too small for 2x2 spatial pooling".into());
                }
                let mut pooled = Vec::with_capacity(pool_w * pool_h * hidden);
                for py in 0..pool_h {
                    for px in 0..pool_w {
                        let mut sum = vec![0.0f32; hidden];
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let idx = ((py * 2 + dy) * grid_w + px * 2 + dx) * hidden;
                                for i in 0..hidden {
                                    sum[i] += x[idx + i] * 0.25;
                                }
                            }
                        }
                        pooled.extend(sum);
                    }
                }
                let out = self.projector.forward(&pooled)?;
                (out, [1, pool_h, pool_w])
            } else {
                let out = self.projector.forward(&x)?;
                (out, [1, grid_h, grid_w])
            };

            let embeddings = projected
                .chunks_exact(self.config.out_dim)
                .map(<[f32]>::to_vec)
                .collect::<Vec<_>>();

            Ok(EncodedMedia::new(embeddings, out_grid, MediaKind::Image))
        })
    }
}

impl MultimodalEncoder for SiglipEncoder {
    fn family(&self) -> &'static str {
        match self.config.family.as_str() {
            "gemma3" => "gemma3",
            _ => "paligemma",
        }
    }

    fn output_hidden(&self) -> usize {
        self.config.out_dim
    }

    fn marker(&self, kind: MediaKind) -> Option<&str> {
        (kind == MediaKind::Image).then_some("<image>")
    }

    fn expand(&self, media: &EncodedMedia) -> String {
        "<image>".repeat(media.embeddings.len())
    }

    fn token_id(&self, kind: MediaKind) -> Option<i64> {
        (kind == MediaKind::Image).then_some(self.config.image_token_id)
    }

    fn encode_image(&self, bytes: &[u8]) -> Result<EncodedMedia, String> {
        let size = self.config.image_size;
        let rgb = image::DynamicImage::ImageRgb8(super::preprocess::decode_image(bytes)?)
            .resize_exact(size as u32, size as u32, image::imageops::FilterType::CatmullRom)
            .to_rgb8();

        let mut pixels = vec![0.0f32; size * size * 3];
        for (i, p) in rgb.pixels().enumerate() {
            for c in 0..3 {
                pixels[c * size * size + i] = (p[c] as f32 / 255.0 - self.config.mean[c]) / self.config.std[c];
            }
        }
        let grid = size / self.config.patch_size;
        self.encode_pixels(&pixels, grid, grid)
    }

    fn visible_lengths(&self, total: usize, spans: &[(usize, [usize; 3], &EncodedMedia)]) -> Result<Option<Vec<usize>>, String> {
        if self.config.family == "gemma3" {
            // Gemma 3: image tokens attend bidirectionally within their image span
            let mut vis = (1..=total).collect::<Vec<_>>();
            for (start, grid, _) in spans {
                let len = grid.iter().product::<usize>();
                let end = start + len;
                for r in *start..end {
                    if r < vis.len() {
                        vis[r] = end;
                    }
                }
            }
            Ok(Some(vis))
        } else {
            // PaliGemma: Prefix-LM attention mask (all prompt prefix tokens attend bidirectionally)
            Ok(Some(vec![total; total]))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn config_parses_paligemma_and_gemma3() {
        let pali = json!({
            "vision": {"hidden_size": 1152, "intermediate_size": 4304, "num_hidden_layers": 27, "num_attention_heads": 16, "image_size": 224, "patch_size": 14},
            "tokens": {"image_token_id": 257152}
        });
        let cfg = SiglipConfig::from_json("paligemma", &pali).unwrap();
        assert_eq!(cfg.hidden, 1152);
        assert_eq!(cfg.patch_size, 14);
        assert!(!cfg.pooling);

        let gemma = json!({
            "vision": {"hidden_size": 1152, "intermediate_size": 4304, "num_hidden_layers": 27, "num_attention_heads": 16, "image_size": 448, "patch_size": 14},
            "tokens": {"image_token_id": 257152}
        });
        let cfg2 = SiglipConfig::from_json("gemma3", &gemma).unwrap();
        assert_eq!(cfg2.image_size, 448);
        assert!(cfg2.pooling);
    }

    #[test]
    fn gemma3_bidirectional_image_attention_mask() {
        let dummy = SiglipConfig {
            family: "gemma3".into(),
            hidden: 64,
            intermediate: 128,
            depth: 1,
            heads: 2,
            image_size: 28,
            patch_size: 14,
            out_dim: 64,
            pooling: false,
            image_token_id: 100,
            mean: [0.5; 3],
            std: [0.5; 3],
        };
        // 10 tokens prompt: token 0 is BOS, tokens 1..5 is an image (len 4), tokens 5..9 are text
        let m = EncodedMedia::new(vec![vec![0.0; 64]; 4], [1, 2, 2], MediaKind::Image);
        let spans = vec![(1, [1, 2, 2], &m)];
        let enc = SiglipEncoder {
            config: dummy,
            patch: Linear::from_parts(Tensor { shape: vec![3 * 196, 64], data: vec![0.0; 64 * 3 * 196] }, 64, 3 * 196, None).unwrap(),
            position: Tensor { shape: vec![64, 4], data: vec![0.0; 4 * 64] },
            layers: Vec::new(),
            post_norm: LayerNorm { weight: vec![1.0; 64], bias: vec![0.0; 64], eps: 1e-6 },
            projector: Linear::from_parts(Tensor { shape: vec![64, 64], data: vec![0.0; 64 * 64] }, 64, 64, None).unwrap(),
            pool: rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap(),
        };
        let vis = enc.visible_lengths(10, &spans).unwrap().unwrap();
        assert_eq!(vis.len(), 10);
        assert_eq!(vis[0], 1); // BOS sees only itself
        assert_eq!(vis[1], 5); // Image token 1 sees up to end of image (5)
        assert_eq!(vis[2], 5); // Image token 2 sees up to end of image (5)
        assert_eq!(vis[3], 5); // Image token 3 sees up to end of image (5)
        assert_eq!(vis[4], 5); // Image token 4 sees up to end of image (5)
        assert_eq!(vis[5], 6); // Text token 5 sees causal (6)
        assert_eq!(vis[6], 7); // Text token 6 sees causal (7)
    }

    #[test]
    fn paligemma_prefix_lm_mask() {
        let dummy = SiglipConfig {
            family: "paligemma".into(),
            hidden: 64,
            intermediate: 128,
            depth: 1,
            heads: 2,
            image_size: 28,
            patch_size: 14,
            out_dim: 64,
            pooling: false,
            image_token_id: 100,
            mean: [0.5; 3],
            std: [0.5; 3],
        };
        let m = EncodedMedia::new(vec![vec![0.0; 64]; 4], [1, 2, 2], MediaKind::Image);
        let spans = vec![(1, [1, 2, 2], &m)];
        let enc = SiglipEncoder {
            config: dummy,
            patch: Linear::from_parts(Tensor { shape: vec![3 * 196, 64], data: vec![0.0; 64 * 3 * 196] }, 64, 3 * 196, None).unwrap(),
            position: Tensor { shape: vec![64, 4], data: vec![0.0; 4 * 64] },
            layers: Vec::new(),
            post_norm: LayerNorm { weight: vec![1.0; 64], bias: vec![0.0; 64], eps: 1e-6 },
            projector: Linear::from_parts(Tensor { shape: vec![64, 64], data: vec![0.0; 64 * 64] }, 64, 64, None).unwrap(),
            pool: rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap(),
        };
        let vis = enc.visible_lengths(8, &spans).unwrap().unwrap();
        assert_eq!(vis, vec![8; 8]); // all prefix tokens see the entire prefix (8)
    }
}
